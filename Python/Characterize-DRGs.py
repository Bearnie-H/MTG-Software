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
import os
import platform
import sys
import traceback
import typing
#   ...

#   Import the necessary third-part modules
import cv2
import matplotlib
import numpy as np
import readlif
from readlif.reader import LifFile, LifImage
#   ...

#   Import the desired locally written modules
from MTG import Logger
from MTG import Utils
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
        self._OutputDirectory: str = None

        self.EnableDryRun: bool = False
        self.ValidateArguments: bool = False
        self.HeadlessMode: bool = False

        self.ImageStackFile: str = ""
        self.ClearingAlgorithm: str = None

        self.MIProjectionFilename: str = ""
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
        #   configuration.  Any optional settings must have valid defaults
        #   provided by the constructor.
        if ( self.ClearingAlgorithm is None ) or ( self.ClearingAlgorithm == "" ):
            self._LogWriter.Warnln(f"Image-Stack clearing algorithm was not set! Defaulting to [ LVCC ].")
            self.ClearingAlgorithm = "LVCC"
        else:
            self._LogWriter.Println(f"Working with image-clearing algorithm [ {self.ClearingAlgorithm} ].")

        Valid |= self._OpenImageStack(self.ImageStackFile, self.ClearingAlgorithm)
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

        if ( self._OutputDirectory is None ) or ( self._OutputDirectory == "" ):
            self._OutputDirectory = os.path.join(os.path.abspath("./"), f"Analysis Results - {datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')}")    #   FIXME

        if ( not os.path.exists(self._OutputDirectory) ) and ( not self.EnableDryRun ):
            self._LogWriter.Println(f"Creating artefact output directory [ {self._OutputDirectory} ].")
            os.makedirs(self._OutputDirectory, mode=0o755, exist_ok=True)

        return self._OutputDirectory

    def GetZStack(self: Configuration) -> np.ndarray:
        """
        GetZStack

        This function...

        Return (ndarray):
            ...
        """

        if ( self._Z_Stack is None ):
            if ( not self._OpenImageStack(self.ImageStackFile, self.ClearingAlgorithm) ):
                raise ValueError(f"Image stack not yet opened, and failed to open image stack file!")

        return self._Z_Stack

    def DumpConfiguration(self: Configuration, JSON: bool = False) -> str:
        """
        DumpConfiguration

        This function...

        Return (str):
            ...
        """

        if ( JSON ):
            return self._DumpConfiguration_JSON()

        return self._DumpConfiguration_String()

    ### ----- End Public Methods -----

    ### +++++ Begin Private Methods +++++
    def _DumpConfiguration_String(self: Configuration) -> str:
        """
        _DumpConfiguration_String

        This function...

        Return (str):
            ...
        """

        TimeFormat: str = "%Y-%m-%d, %H:%M:%S"

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
            f"Analysis Started At:              {self._StartTime.strftime(TimeFormat)}",
            f"Analysis Finished At:             {datetime.datetime.now().strftime(TimeFormat)}",
            f"---------- Command-Line Arguments ----------",
            f"Image Stack File:                 {os.path.basename(self.ImageStackFile)}",
            f"Image Clearing Algorithm:         {self.ClearingAlgorithm.upper() if self.ClearingAlgorithm is not None else 'N/A'}",
            f"...",
            f"---------- Tuning Parameters & Variables ----------",
            f"...",
            f"---------- Results & Metrics ----------",
            f"...",
            f"---------- Enable/Disable Flags ----------",
            f"Dry-Run Mode:                     {self.EnableDryRun}",
            f"Validate Arguments Only:          {self.ValidateArguments}",
            f"Headless Mode:                    {self.HeadlessMode}",
            f"...",
            f"---------- Output Artefacts ----------",
            f"Artefact Directory:               {self.OutputDirectory()}",
            f"Maximum Intensity Projection:     {os.path.basename(self.MIProjectionFilename)}",
            f"...",
            f"",
        ])

    def _DumpConfiguration_JSON(self: Configuration) -> str:
        """
        _DumpConfiguration_JSON

        This function...

        Return (str):
            ...
        """
        raise NotImplementedError(f"_DumpConfiguration_JSON is not implemented.")

    def _OpenImageStack(self: Configuration, ImageStackFilename: str, ClearingAlgorithm: str) -> bool:
        """
        _OpenImageStack

        This function...

        ImageStackFilename:
            ...
        ClearingAlgorithm:
            ...

        Return (bool):
            ...
        """
        self._LogWriter.Println(f"Attempting to open image-stack file [ {ImageStackFilename} ]...")

        if ( self.EnableDryRun ):
            self._LogWriter.Println(f"Dry-Run Mode: Skipping memory-intensive opening of z-stack...")
            self._LogWriter.Println(f"Generating random z-stack array for testing...")
            self._Z_Stack = np.ones((2, 2048, 2048), dtype=np.uint16)
            return True

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

    This function...

    Filename:
        ...

    Return (Tuple[bool, ndarray]):
        [0] - bool:
            ...
        [1] - ndarray:
            ...
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

        ClearedSeriesIndex: int = -1    #   Prepare an index for the z-stack corresponding to the post-filtered images.
        for Index, Image in enumerate(LifStack.image_list, start=1):
            LogWriter.Println(f"Image/Series [ {Index}/{LifStack.num_images} ] - Name: {Image['name']}")
            if ( str(Image['name']).lower().find(SeriesSubstring.lower()) >= 0 ):
                LogWriter.Println(f"Found series using clearing algorithm [ {SeriesSubstring} ] at index [ {Index} ].")
                ClearedSeriesIndex = Index-1

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
        LogWriter.Println(f"Image Series is [ {ImageStack.bit_depth[0]}bits ]...")
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

    This function...

    Filename:
        ...

    Return (Tuple[bool, ndarray]):
        [0] - bool:
            ...
        [1] - ndarray:
            ...
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

        LogWriter.Println(f"Successfully read Z-Stack from *.TIFF file [ {Filename} ]...")
        LogWriter.Println(f"Image Stack has dimensions: x={Z_Stack.shape[1]}, y={Z_Stack.shape[2]}, z={Z_Stack.shape[0]}")

        LogWriter.Println(f"Converting Z-Stack to 16-bit depth...")
        Z_Stack = Z_Stack.astype(np.uint16)
        for i in range(Z_Stack.shape[0]):
            Z_Stack[i,:,:] = Utils.GammaCorrection(Z_Stack[i,:,:])

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

    LogWriter.Println(f"Beginning Analysis...")

    LogWriter.Println(f"Computing the Maximum Intensity Projection (MIP) of the Z-Stack...")
    MIProjection: np.ndarray = MaximumIntensityProjection(Config.GetZStack())
    LogWriter.Println(f"Successfully computed MIP of the provided Z-Stack...")

    #   ...

    LogWriter.Println(f"Finished Analysis!")

    LogWriter.Println(f"Writing analysis configuration settings...")
    LogWriter.Write(Config.DumpConfiguration())
    return 0

def MaximumIntensityProjection(Z_Stack: np.ndarray) -> np.ndarray:
    """
    MaximumIntensityProjection

    This function...

    Z_Stack:
        ...

    Return (np.ndarray):
        ...
    """

    #   Given that the Z_Stack has the 0th axis corresponding to each Z-Slice through the stack,
    #   the maximum intensity projection (MIP) is computed as the maximum pixel value over the
    #   0th axis of the 3D array.
    Projection: np.ndarray = np.max(Z_Stack, axis=0)

    #   Display the MIP image to the user...
    LogWriter.Println(f"Displaying MIP of Z-Stack now...")
    Utils.DisplayImage("Maximum Intensity Projection", Utils.ConvertTo8Bit(Projection), 0, True, ShowOverride=(not Config.HeadlessMode))

    #   If dry run mode is not enabled, write out this projection as an output artefact.
    SaveFilename: str = os.path.join(Config.OutputDirectory(), "Maximum Intensity Projection.png")
    Config.MIProjectionFilename: str = SaveFilename
    if ( not Config.EnableDryRun ):
        Success: bool = cv2.imwrite(SaveFilename, Projection)
        if ( Success ):
            LogWriter.Println(f"Successfully saved Maximum Intensity Projection as file [ {SaveFilename} ].")
        else:
            LogWriter.Println(f"Failed to save Maximum Intensity Projection as file [ {SaveFilename} ]!")


    return Projection

def ProcessArguments() -> bool:
    """
    ProcessArguments

    This function...

    Return (bool):
        ...
    """

    LogWriter.Println(f"Parsing command-line arguments...")

    #   Create the argument parser to accept and process the command-line
    #   arguments.
    Flags: argparse.ArgumentParser = argparse.ArgumentParser(description="", add_help=True)

    #   Add in the flags the program should accept.
    Flags.add_argument("--image-stack", dest="ImageStack", metavar="file-path", type=str, required=True, default="", help="The file path to either the *.lif or multi-page *.tiff image file to analyze.")
    Flags.add_argument("--clearing-algorithm", dest="ClearingAlgo", metavar="algorithm", type=str, required=False, default="lvcc", help="The short-form of the name of the clearing algorithm used by the microscope during the Z-stack image capture.")

    Flags.add_argument("--log-file", dest="LogFile", metavar="file-path", type=str, required=False, default="-", help="File path to the file to write all log messages of this program to.")
    Flags.add_argument("--quiet",    dest="Quiet",   action="store_true",  required=False, default=False, help="Enable quiet mode, disabling logging of eveything except fatal errors.")

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
    if ( Arguments.Quiet ):
        LogWriter.SetOutputFilename("/dev/null")
    else:
        LogWriter.SetOutputFilename(Arguments.LogFile)

    Config.EnableDryRun: bool         = Arguments.DryRun
    Config.ValidateArguments: bool    = Arguments.Validate
    Config.HeadlessMode: bool         = Arguments.Headless

    Config.ImageStackFile: str        = Arguments.ImageStack
    Config.ClearingAlgorithm: str     = Arguments.ClearingAlgo


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
