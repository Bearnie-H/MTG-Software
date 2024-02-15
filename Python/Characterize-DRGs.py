#!/usr/bin/env python3

#   Author: Joseph Sadden
#   Date:   6th November, 2023

#   Script Purpose: This is the top-level script to perform the analysis of the
#                       DRG (Dorsal Root Ganglion) samples used by the
#                       Mend-the-Gap collaboration to quantify the relative
#                       performance of the different neural cell-culture media
#                       and conditions.

#   Import the necessary standard library modules
from __future__ import annotations

import argparse
import datetime
import math
import os
import platform
import sys
import traceback
import typing
#   ...

#   Import the necessary third-part modules
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import readlif
from readlif.reader import LifFile, LifImage
#   ...

#   Import the desired locally written modules
from MTG_Common import Logger
from MTG_Common import Utils
#   ...

#   Define the classes required by this program.
class Configuration():
    """
    Configuration

    This class represents the full set of configuration options and settings for
    this tool. This also contains the necessary logic to validate the provided
    settings, and to generate a configuration dump-file clearly documenting the
    full set of configuration and relevant environment options used during a
    given execution.
    """

    ### +++++ Begin Magic Methods +++++
    def __init__(self: Configuration, LogWriter: Logger.Logger = Logger.Discarder) -> None:
        """
        Constructor

        This function constructs the program-global configuration object, used
        to hold all of the invokation-specific configuration options and
        settings.

        LogWriter:
            The Logger instance to use to write log messages out to. If not
            provided, a default is used which simply ignores all write
            operations.

        Return (None):
            None, on completion the Configuration object is initialized and
            ready for use.
        """

        self._LogWriter: Logger.Logger = LogWriter
        self._StartTime: datetime.datetime = datetime.datetime.now()
        self._OutputDirectory: str = ""

        self.EnableDryRun: bool = False
        self.ValidateArguments: bool = False
        self.HeadlessMode: bool = False

        self.ImageStackFile: str = ""
        self.ClearingAlgorithm: str = ""
        self.LIFSeriesName: str = ""
        self.MIPGamma: float = 1.0
        self.EnableZTruncation: bool = False
        self.ZTruncationThreshold: float = 0.0

        self.MIProjectionFilename: str = ""
        self.ConfigurationDumpFilename: str = ""
        #   ...

        return
    ### ----- End Magic Methods -----

    ### +++++ Begin Public Methods +++++
    def ValidateSettings(self: Configuration) -> bool:
        """
        ValidateSettings

        This function performs the argument validation for all of the
        command-line options and analysis-specific settings available to this
        program. This asserts that all such options are set if necessary, and
        the required external resources exist and are accessible.

        Return (Bool):
            A boolean value indicating whether or not the program should
            continue after argument validation. False may be due to either a
            validation error, or by specifying the option to only perfrom
            argument validation.
        """

        Valid: bool = True
        self._LogWriter.Println(f"Beginning command-line argument validation...")

        if ( self.EnableDryRun ):
            self._LogWriter.Println(f"Enabling dry-run mode. Filesystem alterations and expensive computations will be skipped...")

        if ( self.ValidateArguments ):
            self._LogWriter.Println(f"Performing argument validation only, no further operations will be performed...")

        if ( self.HeadlessMode ):
            self._LogWriter.Println(f"Running in headless mode. All screen-based operations will be suppressed.")

        #   Check and validate each of the command-line options available to the
        #   configuration. Any optional settings must have valid defaults
        #   provided by the constructor.
        if ( self.ClearingAlgorithm is None ) or ( self.ClearingAlgorithm == "" ):
            self.ClearingAlgorithm = "LVCC"
            self._LogWriter.Warnln(f"Image-Stack clearing algorithm was not set! Defaulting to [ {self.ClearingAlgorithm} ].")
        else:
            self._LogWriter.Println(f"Working with image-clearing algorithm [ {self.ClearingAlgorithm} ].")

        if ( Config.EnableZTruncation ):
            self._LogWriter.Println(f"Enabling saturated feature truncation of the Z-Stack prior to computation of the Maximum Intensity Projection.")
            if ( not 0 < self.ZTruncationThreshold < 1.0 ):
                self._LogWriter.Errorln(f"Saturated feature truncation threshold must be a real number on the range (0, 1) - Got [ {self.ZTruncationThreshold} ]!")
                self._LogWriter.Warnln(f"Disabling Z-Stack saturated feature truncation due to invalid threshold...")
                self.EnableZTruncation = False
            else:
                self._LogWriter.Println(f"Saturated feature truncation will allow features up to a brightness curvature value of [ {self.ZTruncationThreshold*100}% ] the logarithmic pixel depth.")
        else:
            self._LogWriter.Println(f"Disabling Z-Stack saturated feature truncation...")

        #   Ensure the image stack file provided is suitable.
        Valid |= self._OpenImageStack(self.ImageStackFile, self.ClearingAlgorithm)

        #   Validate the Gamma value used for contrast adjustment
        if ( self.MIPGamma <= 0.0 ):
            self._LogWriter.Errorln(f"Gamma value for non-linear constrast stretch of maximum intensity projection of the Z-Stack must be a positive real number - Got [ {self.MIPGamma} ].")
            Valid |= False
        else:
            self._LogWriter.Println(f"Working with gamma value of [ {self.MIPGamma} ] for Z-Stack constrast stretch.")

        #   ...

        if Valid:
            self._LogWriter.Println(f"Command-line arguments validated successfully!")
        else:
            self._LogWriter.Errorln(f"Failed to validate provided command-line arguments!")

        return Valid

    def OutputDirectory(self: Configuration) -> str:
        """
        OutputDirectory

        This function will return the absolute path to the output directory for
        the current invokation. This directory contains all of the generated
        artefacts from running this program, and is unique per execution of the
        program.

        Return (str):
            The absolute path to the output directory to use for the current
            invokation of the program.
        """

        if ( self.ImageStackFile is None ) or ( self.ImageStackFile == "" ):
            raise RuntimeError(f"Output directory cannot be determined prior to knowing the image file to operate on!")

        if ( self._OutputDirectory is None ) or ( self._OutputDirectory == "" ):
            Filename: str = os.path.splitext(os.path.basename(self.ImageStackFile))[0]
            self._OutputDirectory = os.path.join(os.path.abspath(os.path.dirname(self.ImageStackFile)), f"{Filename} Analysis Results - {datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')}")

        if ( not os.path.exists(self._OutputDirectory) ) and ( not self.EnableDryRun ):
            self._LogWriter.Println(f"Creating artefact output directory [ {self._OutputDirectory} ].")
            os.makedirs(self._OutputDirectory, mode=0o755, exist_ok=True)

        return self._OutputDirectory

    def GetZStack(self: Configuration) -> np.ndarray:
        """
        GetZStack

        This function will return the Z-Stack image to be operated on, opening
        and preparing it if this has not yet been performed. Once opened, the
        Z-Stack will be cached to save having to prepare it multiple times.

        Return (ndarray):
            A, minimally 3-D, array of pixel values corresponding to the image
            planes within the Z-Stack image to be processed. The zeroth index
            corresponds to the Z-axis, while the 1st and 2nd indices correspond
            to x, and y respectively. Further indices, if they exist, correspond
            to the different colour channels of the image.
        """

        if ( self._Z_Stack is None ):
            if ( not self._OpenImageStack(self.ImageStackFile, self.ClearingAlgorithm) ):
                raise ValueError(f"Image stack not yet opened, and failed to open image stack file!")

        return self._Z_Stack

    def DumpConfiguration(self: Configuration) -> str:
        """
        DumpConfiguration

        This function is a standard documentation function, used to dump out the
        full configuration state of this software on completion of the analysis.
        This is used to ensure the full set of tunable settings or analysis
        parameters are clearly captured to simplify repeatable execution of the
        analysis, as well as ensuring no analysis parameters are accidentally
        forgotten.

        Return (str):
            A formatted string containing the full set of analysis parameters,
            tunable settings, platform versioning, and environment capture to
            allow for repeatable execution of the analysis.
        """

        TimeFormat: str = "%Y-%m-%d, %H:%M:%S"
        AnalysisFinished: datetime.datetime = datetime.datetime.now()
        AnalysisDuration: datetime.timedelta = math.ceil((AnalysisFinished - self._StartTime).total_seconds())
        Hours, rem = divmod(AnalysisDuration, 60*60)
        Minutes, Seconds = divmod(rem, 60)

        return '\n'.join([
            f"",
            f"---------- Platform and Software Versioning ----------",
            f"Python Version:                   {platform.python_version()}",
            f"Platform Architecture:            {platform.platform()}",
            f"NumPy Version:                    {np.version.full_version}",
            f"OpenCV Version:                   {cv2.getVersionString()}",
            f"MatPlotLib Version:               {matplotlib.__version__}",
            f"ReadLIF Version:                  {readlif.__version__}",
            f"...",
            f"---------- Analysis Timing Information ----------",
            f"Analysis Started :                {self._StartTime.strftime(TimeFormat)}",
            f"Analysis Finished:                {datetime.datetime.now().strftime(TimeFormat)}",
            f"Analysis Duration:                {Hours:02d}:{Minutes:02d}:{Seconds:02d} (HH:MM:SS)",
            f"---------- Command-Line Arguments ----------",
            f"Image Stack File:                 {os.path.basename(self.ImageStackFile)}",
            f"Image Clearing Algorithm:         {self.ClearingAlgorithm.upper() if self.ClearingAlgorithm is not None else 'N/A'}",
            f"Z-Stack Series Name:              {self.LIFSeriesName if self.LIFSeriesName is not None else 'N/A'}",
            f"...",
            f"---------- Tuning Parameters & Variables ----------",
            f"MIP Gamma Value:                  {self.MIPGamma:.4f}",
            f"Z-Stack Truncation Enabled:       {self.EnableZTruncation}",
            f"Z-Stack Truncation Threshold:     {self.ZTruncationThreshold if self.EnableZTruncation else 'N/A'}",
            f"...",
            f"---------- Results & Metrics ----------",
            f"Image Bit Depth:                  {np.iinfo(self._Z_Stack.dtype).bits if self._Z_Stack is not None else 'unknown'} bit",
            f"...",
            f"---------- Enable/Disable Flags ----------",
            f"Dry-Run Mode:                     {self.EnableDryRun}",
            f"Validate Arguments Only:          {self.ValidateArguments}",
            f"Headless Mode:                    {self.HeadlessMode}",
            f"...",
            f"---------- Output Artefacts ----------",
            f"Artefact Directory:               {self.OutputDirectory() if not self.EnableDryRun else 'N/A'}",
            f"Output Log File:                  {self._LogWriter.GetOutputFilename() if self._LogWriter.WritesToFile() else 'None'}",
            f"Configuration State File:         {os.path.basename(self.ConfigurationDumpFilename)}",
            f"Maximum Intensity Projection:     {os.path.basename(self.MIProjectionFilename)}",
            f"...",
            f"",
        ])

    ### ----- End Public Methods -----

    ### +++++ Begin Private Methods +++++
    def _OpenImageStack(self: Configuration, ImageStackFilename: str, ClearingAlgorithm: str = None) -> bool:
        """
        _OpenImageStack

        This function attempts to open the provided image stack file, attempting
        to open it as either of a *.LIF or a *.TIFF formatted file. The *.LIF
        format is the standard format as exported by the Leica Thunder
        microscope, whereas a multi-page *.TIFF is a more generic format for
        including multiple 2D images within a single file.

        ImageStackFilename:
            The full path to the file containing the image stack to open and
            read. This must either be a *.LIF or multi-page *.TIFF file.
        ClearingAlgorithm:
            If working with a *.LIF file, the clearing algorithm is used to
            determine which of the 'Series' included in the file to extract as
            the Z-Stack. This is the short-form name of the algorithm used to
            clean up the images from the well-known artefacts of the particular
            imaging modality of the Thunder microscope.

        Return (bool):
            A boolean indicating the image stack was opened, read, and converted
            to a generic internal representation successfully. If this returns
            True, then the Config.GetZStack() function allows access to the
            Z-Stack as read from the provided file.
        """
        self._LogWriter.Println(f"Attempting to open image-stack file [ {ImageStackFilename} ]...")

        Success, self._Z_Stack = OpenLIFFile(ImageStackFilename, ClearingAlgorithm)
        if ( Success ):
            self._LogWriter.Println(f"Successfully opened and extraced Z-Stack image from *.LIF file [ {ImageStackFilename} ].")
            return True

        Success, self._Z_Stack = OpenTIFFFile(ImageStackFilename)
        if ( Success ):
            self._LogWriter.Println(f"Successfully opened and extraced Z-Stack image from *.TIFF file [ {ImageStackFilename} ].")
            return True

        self._LogWriter.Errorln(f"Failed to open file [ {ImageStackFilename} ] as either *.LIF or *.TIFF file to read Z-Stack image data!")
        return False

    ### ----- End Private Methods -----

#   Standalone Helper Functions
def OpenLIFFile(Filename: str = "", SeriesSubstring: str = "") -> typing.Tuple[bool, np.ndarray]:
    """
    OpenLIFFile

    This function attempts to open the provided filename as a *.LIF formatted
    image file.  If this succeeds, it then attempts to find a "series" within
    the file with the provided substring as a search token, to use as the
    Z-Stack image. The Z-Stack pixels are linearly rescaled twice, first to take
    full advantage of the bit-depth of the NumPy data type of the array, and
    second to fully cover this range.  This typically implies a scaling from
    12-bit to 16-bits by multiplying pixels by 16, followed by linearly
    stretching the pixel values to cover the full range [0,65536).

    Filename:
        The full path to the file to open and attempt to read as a *.LIF image
        file.
    SeriesSubstring:
        A search token to use to identify which specific "Series" within the
        *.LIF file corresponds to the Z-stack of interest.

    Return (Tuple[bool, ndarray]):
        [0] - bool:
            A boolean indicating whether the image file was opened successfully
            and the Z-stack extracted.
        [1] - ndarray:
            The Z-stack as extracted and initially normalized from the image
            file, or None on an error.
    """

    if ( Filename is None ) or ( Filename == "" ):
        LogWriter.Errorln(f"Missing [ Filename ] for opening *.LIF image file!")
        return (False, None)

    if ( SeriesSubstring is None ) or ( SeriesSubstring == "" ):
        LogWriter.Errorln(f"Missing [ SeriesSubstring ] for opening *.LIF image file!")
        return (False, None)

    LogWriter.Println(f"Attempting to open file [ {Filename} ] as *.LIF image file...")

    try:
        LifStack: LifFile = LifFile(Filename)
        LogWriter.Println(f"Successfully opened file [ {Filename} ] as *.LIF file.")

        LogWriter.Println(f"Found a total of [ {LifStack.num_images} ] images and/or series within the file...")
        LogWriter.Println(f"Searching for a series containing the clearing algorithm [ {SeriesSubstring} ]...")

        #   Prepare an index for the z-stack corresponding to the post-filtered images.
        ClearedSeriesIndex: int = -1
        for Index, Image in enumerate(LifStack.image_list, start=1):
            LogWriter.Println(f"Image/Series [ {Index}/{LifStack.num_images} ] - Name: {Image['name']}")
            if ( str(Image['name']).lower().find(SeriesSubstring.lower()) >= 0 ):
                LogWriter.Println(f"Found series using clearing algorithm [ {SeriesSubstring} ] at index [ {Index} ].")
                Config.LIFSeriesName = Image['name']
                ClearedSeriesIndex = Index-1

        if ( Config.HeadlessMode ):
            LogWriter.Errorln(f"Failed to identify expected series name for z-stack. Cannot determine the correct series while in headless mode!")
            raise ValueError(f"Failed to identify expected series name for z-stack!")
        else:
            if ( ClearedSeriesIndex < 0 ):
                #   Failed to identify the likely series based off the name.
                #   Prompt the user to provide the index of the series.
                LogWriter.Warnln(f"Failed to identify expected series name for z-stack...")
                ClearedSeriesIndex = int(input("Please enter the index of the image or series to open as the z-stack: "))

        LogWriter.Println(f"Working with series index [ {ClearedSeriesIndex+1} ] as the DRG Z-Stack...")
        ImageStack: LifImage = LifStack.get_image(ClearedSeriesIndex)
        LogWriter.Println(f"Image Series has dimensions: x={ImageStack.dims.x}, y={ImageStack.dims.y}, z={ImageStack.dims.z}, t={ImageStack.dims.t}, c={ImageStack.channels}")

        #   Specifically extract out the z-stack...
        #   Scale up to the full 16-bits, if not already.
        LogWriter.Println(f"Image Series is [ {ImageStack.bit_depth[0]}bit ] colour depth...")
        ScaleFactor: int = 1
        if ( ImageStack.bit_depth[0] != 16 ):
            LogWriter.Println(f"Converting image stack to 16 bit depth...")
            ScaleFactor = (16 - ImageStack.bit_depth[0]) ** 2

        LogWriter.Println(f"Reading Z-slices from *.LIF file...")
        Z_Stack: np.ndarray = np.array([np.uint16(np.array(i) * ScaleFactor) for i in ImageStack.get_iter_z()])
        LogWriter.Println(f"Finished Reading z-slices from *.LIF file into numpy array!")

        return (True, Z_Stack)
    except:
        LogWriter.Warnln(f"File [ {Filename} ] failed to be opened and read as a *.LIF file!")

    return (False, None)

def OpenTIFFFile(Filename: str = None) -> typing.Tuple[bool, np.ndarray]:
    """
    OpenTIFFFile

    This function provides the same functionality as OpenLIFFile() above, but
    for multi-page *.TIFF files.

    Filename:
        The full path to the *.TIFF file to attempt to open, containing the
        Z-Stack image.

    Return (Tuple[bool, ndarray]):
        [0] - bool:
            A boolean indicating whether the image file was opened successfully
            and the Z-stack extracted.
        [1] - ndarray:
            The Z-stack as extracted and initially normalized from the image
            file, or None on an error.
    """

    if ( Filename is None ) or ( Filename == "" ):
        LogWriter.Errorln(f"Missing [ Filename ] for opening *.TIFF image file!")
        return (False, None)

    LogWriter.Println(f"Attempting to open file [ {Filename} ] as a multi-page *.TIFF image file...")
    Config.ClearingAlgorithm = None

    try:
        Valid, Z_Stack = cv2.imreadmulti(Filename, [], cv2.IMREAD_ANYDEPTH)
        if ( not Valid ):
            raise ValueError(f"Image file [ {Filename} ] cannot be read with cv2.imreadmulti()!")

        #   Convert from a list of NumPy arrays to a single large array.
        Z_Stack = np.array(Z_Stack)
        LogWriter.Println(f"Successfully read Z-Stack from *.TIFF file [ {Filename} ]...")
        LogWriter.Println(f"Image Stack has dimensions: x={Z_Stack.shape[1]}, y={Z_Stack.shape[2]}, z={Z_Stack.shape[0]}")

        return (True, Z_Stack)
    except:
        LogWriter.Warnln(f"File [ {Filename} ] failed to be opened and read as a multi-page *.TIFF file!")

    return (False, None)

#   Set the logger for this application to be a default-initialized logger
LogWriter: Logger.Logger = Logger.Logger(Prefix=os.path.basename(sys.argv[0]))

#   Initialize the top-level configuration object for this program
Config: Configuration = Configuration(LogWriter=LogWriter)

#   Main
#       This is the main entry point of the script.
def main() -> int:

    LogWriter.Println(f"Beginning the Dorsal Root Ganglion Neurite Outgrowth Analysis...")

    #   Clean up the Z-Stacks to remove the staining saturation "noise" signals.
    if ( Config.EnableZTruncation ):
        LogWriter.Println(f"Cleaning up Z-Stack to remove regions where the staining oversaturated the fluorescent signal...")
        RemoveStainSaturation(Config.GetZStack())
        LogWriter.Println(f"Finished cleaning up Z-Stack!")

    #   Segment the Z-Stack into the following four classes:
    #   1)  DRG Body
    #   2)  Growth Chip Edges
    #   3)  Neurites
    #   4)  Background
    #
    #   So that each of these can be operated on independently for the remainder of the analysis.
    #   ...


    #   Compute the maximum intensity projection of the Z-Stack.
    LogWriter.Println(f"Computing the Maximum Intensity Projections (MIPs) of the Z-Stack...")
    X_MIProjection: np.ndarray = MaximumIntensityProjection(Config.GetZStack(), axis='x')
    Y_MIProjection: np.ndarray = MaximumIntensityProjection(Config.GetZStack(), axis='y')
    Z_MIProjection: np.ndarray = MaximumIntensityProjection(Config.GetZStack(), axis='z')
    LogWriter.Println(f"Successfully computed MIPs of the provided Z-Stack!")

    #   ...

    LogWriter.Println(f"Finished the Dorsal Root Ganglion Neurite Outgrowth Analysis!")

    WriteConfigurationState()

    return 0

def RemoveStainSaturation(Z_Stack: np.ndarray) -> None:
    """
    RemoveStainSaturation

    This function...

    Z_Stack:
        ...

    Return (None):
        ...
    """

    SaturationPercentileStart: float = 0.95
    SaturationPercentileEnd: float = 1.0
    StepSize: float = 0.0001
    CurvatureThreshold: int = int(np.exp(Config.ZTruncationThreshold * np.log(np.iinfo(Z_Stack.dtype).max)))

    #   Iterate through each slice of the Z-Stack...
    for SliceIndex, Slice in enumerate(Z_Stack):

        #   We want to look for "outliers" in terms of pixel brightness values.
        #
        #   We know that the most problematic outliers are those where the
        #   brightness saturates and therefore overwhelms anything and
        #   everything in any other layer of the Z-Stack when taking the Maximum
        #   Intensity Projection.
        #
        #   One way we can remove these outliers is to consider the relationship
        #   between the n-th percentile, and the corresponding pixel brightness
        #   value it corresponds to. In the bulk of the image, we expect a
        #   relatively slow change, or low slope to the curve. For the saturated
        #   outliers, on the other hand, we expect to see this change to very
        #   large changes in pixel brightness for a very small increment in
        #   percentile. Therefore, we want to look for the "knee" of this curve
        #   and declare everything past the knee to be outliers to be masked
        #   out.
        Percentiles: np.ndarray = np.arange(SaturationPercentileStart, SaturationPercentileEnd, StepSize)
        PixelBrightnesses: np.ndarray = np.array(
            [np.quantile(Slice, x) for x in Percentiles]
        )
        Slopes: np.ndarray = np.diff(PixelBrightnesses, n=1) / (StepSize * 100)
        Curvatures: np.ndarray = np.diff(Slopes, n=1) / (StepSize * 100)

        SaturationPercentile: float = Percentiles[np.argmax(Curvatures > CurvatureThreshold) + 2]

        #   Compute the saturation threshold (in terms of pixel brightness) for the current slice.
        SaturationThreshold: int = int(np.quantile(Slice, SaturationPercentile))
        LogWriter.Println(f"Applying saturation threshold at the [ {SaturationPercentile*100:.4f}-th ] percentile ({SaturationThreshold}) for slice [ {SliceIndex}/{Z_Stack.shape[0]} ]...")

        #   Set all oversaturated pixels to 0, to allow features in other layers to take priority.
        if ( not Config.HeadlessMode ):
            Original: np.ndarray = Slice.copy()
            Truncated: np.ndarray = Original.copy()
            Truncated[Truncated > SaturationThreshold] = 0.0
            Original = Utils.GammaCorrection(Original, Gamma=Config.MIPGamma)
            Truncated = Utils.GammaCorrection(Truncated, Gamma=Config.MIPGamma)
            Utils.DisplayImage(f"Original and Post Saturation Slice {SliceIndex}/{Z_Stack.shape[0]}", cv2.hconcat([Utils.ConvertTo8Bit(Original), Utils.ConvertTo8Bit(Truncated)]), HoldTime=1, Topmost=True)

        #   Update the actual Z-Stack pixel values with the newly filtered slice.
        Slice[Slice > SaturationThreshold] = 0
        Z_Stack[SliceIndex,...] = Slice

    return


def MaximumIntensityProjection(Z_Stack: np.ndarray, Axis: str='z') -> np.ndarray:
    """
    MaximumIntensityProjection

    This function computes and prepares the Maximum Intensity Projection from
    the Z-Stack, returning a single 2D image consiting of the collection of the
    brightest pixel values from any slice through the stack. This is a commonly
    used projection method for operating with Z-Stack images as a 'smaller' 2D
    image, ideally without losing too much information.

    In addition to simply computing this projection image, the contrast is
    non-linearly stretched by the --mip-gamma command-line parameter, the
    projection is displayed, and the image is saved to the output artefact
    directory for later review or further processing.

    Z_Stack:
        The current open Z-stack of the DRG to compute the projection of.
    Axis:
        Which axis of the Z-Stack should be collapsed in the MIP.

    Return (np.ndarray):
        The resulting 2D NumPy array of the MIP image. The pixel values are
        scaled to the full range of the bit depth of the image, and the contrast
        has been stretched based off the --mip-gamma command-line argument
        (default = 1).
    """

    axis: int = 0
    if ( Axis.lower() == 'z' ):
        axis = 0
    elif ( Axis.lower() == 'y' ):
        axis = 1
    elif ( Axis.lower() == 'x' ):
        axis = 2
    else:
        raise ValueError(f"Maximum Intensity Projection Axis must be one of [ 'x', 'y', 'z' ]. Got [ '{Axis}' ]")

    #   Given that the Z_Stack has the 0th axis corresponding to each Z-Slice through the stack,
    #   the maximum intensity projection (MIP) is computed as the maximum pixel value over the
    #   0th axis of the 3D array.
    Projection: np.ndarray = np.max(Z_Stack, axis=axis)

    #   Apply Gamma correction to this projection image...
    Projection = Utils.GammaCorrection(Projection, Gamma=Config.MIPGamma)

    #   Display the MIP image to the user...
    if ( not Config.HeadlessMode ):
        LogWriter.Println(f"Displaying {Axis.upper()}-Axis Maximum Intensity Projection of the Z-Stack now...")
        Utils.DisplayImage(f"{Axis.upper()}-Axis Maximum Intensity Projection", Utils.ConvertTo8Bit(Projection), 0, True)

    #   If dry run mode is not enabled, write out this projection as an output artefact.
    if ( not Config.EnableDryRun ):
        SaveFilename: str = os.path.join(Config.OutputDirectory(), f"Maximum Intensity Projection - {Axis.upper()} Axis - Gamma {Config.MIPGamma:.4f}.png")
        Config.MIProjectionFilename = SaveFilename
        Success: bool = cv2.imwrite(SaveFilename, Projection)
        if ( Success ):
            LogWriter.Println(f"Successfully saved {Axis.upper()}-Axis Maximum Intensity Projection as file [ {SaveFilename} ].")
        else:
            LogWriter.Println(f"Failed to save {Axis.upper()}-Axis Maximum Intensity Projection as file [ {SaveFilename} ]!")

    return Projection

def WriteConfigurationState() -> None:
    """
    WriteConfigurationState

    This function simply dumps out the configuration state of the program to
    both the Logger, and to a file in the output artefact directory for possible
    future review.

    Return (None):
        This function returns nothing. The formatted configuration state is
        written to the global Logger, and is written to a text file in the
        output directory.
    """

    Config.ConfigurationDumpFilename  = os.path.join(Config.OutputDirectory(), f"{os.path.splitext(os.path.basename(sys.argv[0]))[0]} - Configuration State.txt")
    ConfigurationState: str = Config.DumpConfiguration()

    LogWriter.Println(f"Writing application configuration state to standard output...")
    LogWriter.Write(ConfigurationState)

    if ( not Config.EnableDryRun ):
        LogWriter.Println(f"Writing configuration state to the file [ {os.path.basename(Config.ConfigurationDumpFilename)} ].")
        with open(Config.ConfigurationDumpFilename, "+w") as OutFile:
            OutFile.write(ConfigurationState)

    LogWriter.Println(f"Finished writing application configuration state.")
    return

def ProcessArguments() -> bool:
    """
    ProcessArguments

    This function handles the full initialization, processing, and validation of
    the command-line arguments of the program.

    Return (bool):
        A boolean value indicating whether or not execution should continue
        after the arguments have been processed.
    """

    LogWriter.Println(f"Parsing command-line arguments...")

    #   Create the argument parser to accept and process the command-line
    #   arguments.
    Flags: argparse.ArgumentParser = argparse.ArgumentParser(description="", add_help=True)

    #   Add in the flags the program should accept.
    Flags.add_argument("--image-stack", dest="ImageStack", metavar="file-path", type=str, required=True, default="", help="The file path to either the *.lif or multi-page *.tiff image file to analyze.")
    Flags.add_argument("--clearing-algorithm", dest="ClearingAlgo", metavar="algorithm", type=str, required=False, default="lvcc", help="The short-form of the name of the clearing algorithm used by the microscope during the Z-stack image capture.")

    #   Add in flags for tuning parameters or analysis inputs.
    Flags.add_argument("--mip-gamma", dest="MIPGamma", metavar="gamma", type=float, required=False, default=1.0, help="The brightness gamma value to use to non-linearly stretch the constrast of the generated MIP image from the Z-Stack.")
    Flags.add_argument("--z-truncation-threshold", dest="ZTruncationThreshold", metavar="threshold", type=float, required=False, default=0.875, help="If Z Truncation is enabled, what logarithmic fraction of the bit depth of the image should be used as the threshold for determining saturated features?")

    #   Add in flags to enable or disable certain functionalities within the script
    Flags.add_argument("--disable-z-truncation", dest="EnableZTruncation", action="store_false", required=False, default=True, help="Disable the pixel saturation filtering of the Z-Stack used to remove saturated artefacts from the staining process.")

    #   Add in flags for manipulating the logging functionality of the script.
    Flags.add_argument("--log-file", dest="LogFile", metavar="file-path", type=str, required=False, default="-", help="File path to the file to write all log messages of this program to.")
    Flags.add_argument("--quiet",    dest="Quiet",   action="store_true",  required=False, default=False, help="Enable quiet mode, disabling logging of eveything except fatal errors.")

    #   Finally, add in scripts for modifying the basic environment or end-state of the script.
    Flags.add_argument("--dry-run",  dest="DryRun",   action="store_true", required=False, default=False, help="Enable dry-run mode, where no file-system alterations or heavy computations are performed.")
    Flags.add_argument("--validate", dest="Validate", action="store_true", required=False, default=False, help="Only validate the command-line arguments, do not proceed with the remainder of the program.")
    Flags.add_argument("--headless", dest="Headless", action="store_true", required=False, default=False, help="Run in 'headless' mode, where nothing is displayed to the screen.")

    #   ...

    #   Actually parse the command-line arguments
    Arguments: argparse.Namespace = Flags.parse_args()

    #   Pull out the settings from the arguments name-space, setting the
    #   program-local variables and settings within the global Configuration
    #   object as required.
    #   ...

    Config.ImageStackFile        = Arguments.ImageStack

    if ( Arguments.Quiet ):
        LogWriter.SetOutputFilename("/dev/null")
    elif ( Arguments.LogFile == "-" ):
        LogWriter.SetOutputStream(sys.stdout)
    else:
        LogWriter.SetOutputFilename(os.path.join(Config.OutputDirectory(), Arguments.LogFile))

    Config.EnableDryRun         = Arguments.DryRun
    Config.ValidateArguments    = Arguments.Validate
    Config.HeadlessMode         = Arguments.Headless

    Config.ClearingAlgorithm    = Arguments.ClearingAlgo
    Config.MIPGamma             = Arguments.MIPGamma
    Config.EnableZTruncation    = Arguments.EnableZTruncation
    Config.ZTruncationThreshold = Arguments.ZTruncationThreshold

    LogWriter.Println(f"Finished parsing command-line arguments!")

    #   Validate all of the arguments as given, asserting they are sensible and
    #   the program should continue.
    return (( Config.ValidateSettings() ) and ( not Config.ValidateArguments ))

#   Allow this script to be called from the command-line and execute the main
#   function. If anything needs to happen before executing main, add it here
#   before the call.
if __name__ == "__main__":
    if ( ProcessArguments() ):
        try:
            sys.exit(main())
        except Exception as e:
            LogWriter.Fatalln(f"Exception raised in main(): [ {e} ]\n\n{''.join(traceback.format_exception(e, value=e, tb=e.__traceback__))}")
    else:
        sys.exit(1)
