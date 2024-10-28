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
from scipy.signal import correlate
#   ...

#   Import the desired locally written modules
from MTG_Common import Logger
from MTG_Common import Utils
from MTG_Common import ZStack
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

        self._Z_Stack: ZStack.ZStack = None
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

    def GetZStack(self: Configuration) -> ZStack.ZStack:
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
            f"ReadLIF Version:                  {ZStack.readlif.__version__}",
            f"czifile Version:                  {ZStack.czifile.__version__}",
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
            f"Image Bit Depth:                  {np.iinfo(self._Z_Stack.Pixels.dtype).bits if self._Z_Stack is not None else 'unknown'} bit",
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

        self._Z_Stack: ZStack.ZStack = ZStack.ZStack(LogWriter=LogWriter)
        Success: bool = False
        match os.path.splitext(ImageStackFilename)[1].lower():
            case ".lif":
                Success = self._Z_Stack.OpenLIFFile(ImageStackFilename, ClearingAlgorithm)
            case ".tif", ".tiff":
                Success = self._Z_Stack.OpenTIFFile(ImageStackFilename)
            case ".czi":
                Success = self._Z_Stack.OpenCZIFile(ImageStackFilename)
            case _:
                self._LogWriter.Errorln(f"Unknown Z-Stack file extension for file [ {ImageStackFilename} ]...")
                return False

        if ( Success ):
            self._LogWriter.Println(f"Successfully opened and extraced Z-Stack image from file [ {ImageStackFilename} ].")
            return True

        self._LogWriter.Errorln(f"Unknown error while attempting to open Z-Stack image from file [ {ImageStackFilename} ]...")
        return False


    ### ----- End Private Methods -----

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
        RemoveStainSaturation(Config.GetZStack().Pixels)
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
    MIP_Z: np.ndarray = Utils.GammaCorrection(Config.GetZStack().MaximumIntensityProjection(Axis='z'), Gamma=0.5)
    for Axis in ['x', 'y', 'z']:
        Projection: np.ndarray = Utils.GammaCorrection(Config.GetZStack().MaximumIntensityProjection(Axis=Axis), Gamma=0.5)
        Utils.DisplayImage(
            f"Maximum Intensity Projection ({Axis} Axis)",
            Projection,
            10,
            True,
            (not Config.HeadlessMode)
        )

        _, Thresholded = cv2.threshold(Utils.ConvertTo8Bit(Projection), 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        Utils.DisplayImage("Thresholded Z-MIP", Thresholded, 0, True)

        MedianFiltered: np.ndarray = cv2.medianBlur(Thresholded, ksize=201)
        Utils.DisplayImage("Median Filtered, Thresholded Z-MIP", MedianFiltered, 0, True)

        CentroidLocation: typing.Tuple[int, int] = EstimateCentroid(MedianFiltered)
        LogWriter.Println(f"Centroid detected at: {CentroidLocation}...")
        Utils.DisplayImage("Annotated Centroid", cv2.circle(Utils.GreyscaleToBGR(MedianFiltered), CentroidLocation, 10, (0, 255, 0), -1), 0, True)

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

def EstimateCentroid(ThresholdedImage: np.ndarray, CorrelationThreshold: float = 0.95, KernelStepSize: int = 10, InitialKernelSize: int = 11, CentroidJitterThreshold: int = 2) -> typing.Tuple[int, int]:
    """
    EstimateCentroid

    This function estimates the location of the centroid of the DRG within the provided pre-thresholded and binarized image,
    using an iterative cross-correlation search method to identify the location of the largest and most consistently circular-ish figure
    within the image.

    ThresholdedImage:
        ...
    CorrelationThreshold:
        ...
    KernelStepSize:
        ...
    InitialKernelSize:
        ...
    CentroidJitterThreshold:
        ...

    Return (Tuple):
        [0] - int:
            ...
        [1] - int:
            ...
    """

    if ( ThresholdedImage is None ):
        raise ValueError(f"ThresholdedImage must be provided!")

    if ( CorrelationThreshold <= 0.0 ) or ( CorrelationThreshold >= 1.0 ):
        raise ValueError(f"CorrelationThreshold must be within the range (0, 1.0)!")

    if ( KernelStepSize < 1 ):
        raise ValueError(f"KernelStepSize must be at least 1!")

    if ( InitialKernelSize <= 2 ):
        raise ValueError(f"InitialKernelSize must be at least 3!")

    if ( InitialKernelSize <= 0 ):
        raise ValueError(f"InitialKernelSize must be at least 1!")

    #   The previously-determined list of centroid locations within the image.
    Centroid_Xs: typing.List[int] = []
    Centroid_Ys: typing.List[int] = []

    #   Convert the input image to a float64 image on the range [0, 1.0]
    ThresholdedImage = Utils.GammaCorrection(ThresholdedImage.copy().astype(np.float64), Minimum=0.0, Maximum=1.0)

    #   Iterate over the possible kernel sizes...
    for KernelSize in range(InitialKernelSize, int(min(ThresholdedImage.shape)), KernelStepSize):

        #   Create the circular kernel to correlate against the DRG image
        SearchKernel: np.ndarray = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(KernelSize, KernelSize)).astype(np.float64)

        #   Compute the correlation map between the two images, asserting that it is the same size and
        #   uses the same coordinates as the original image.
        CorrelationMap: np.ndarray = correlate(ThresholdedImage, SearchKernel, mode='same', method='fft')

        #   Filter the correlation map to only find the regions of sufficiently strong correlation
        CorrelationMap[CorrelationMap < CorrelationThreshold*np.max(CorrelationMap)] = 0.0

        #   Compute the image moments, in order to determine the centroid of the highly-correlated region(s).
        ImageMoments: cv2.Moments = cv2.moments(CorrelationMap, binaryImage=False)

        #   Actually compute the centroid location
        Centroid_X = int(ImageMoments["m10"] / ImageMoments["m00"])
        Centroid_Y = int(ImageMoments["m01"] / ImageMoments["m00"])

        #   If the centroid does not exist, then the kernel is improperly sized so we exit.
        if ( Centroid_X == 0 ) and ( Centroid_Y == 0 ):
            continue

        #   Check where the current averaged centroid location is...
        Current_X, Current_Y = 0, 0
        if ( len(Centroid_Xs) > 0 ) and ( len(Centroid_Ys) > 0 ):
            Current_X, Current_Y = tuple([int(np.mean(np.array(x))) for x in [Centroid_Xs, Centroid_Ys]])

        Centroid_Xs.append(Centroid_X)
        Centroid_Ys.append(Centroid_Y)
        Next_X, Next_Y = tuple([int(np.mean(np.array(x))) for x in [Centroid_Xs, Centroid_Ys]])

        if (abs(Next_X - Current_X) < CentroidJitterThreshold ) and ((Next_Y - Current_Y) < CentroidJitterThreshold ):
            return (Next_X, Next_Y)

    return tuple([int(np.mean(np.array(x))) for x in [Centroid_Xs, Centroid_Ys]])

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
    Flags.add_argument("--enable-z-truncation", dest="EnableZTruncation", action="store_true", required=False, default=False, help="Enable the pixel saturation filtering of the Z-Stack used to remove saturated artefacts from the staining process.")

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
