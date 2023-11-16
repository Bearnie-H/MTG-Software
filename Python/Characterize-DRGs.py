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
import numpy as np
import cv2
import matplotlib
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

        self._LogWriter = LogWriter
        self._StartTime = datetime.datetime.now()
        self._OutputDirectory = None

        self.EnableDryRun = False
        self.ValidateArguments = False

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

        #   Check and validate each of the command-line options available to the
        #   configuration.  Any optional settings must have valid defaults
        #   provided by the constructor.
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
            f"---------- Platform and Software Versioning ----------",
            f"Python Version:                   {platform.python_version()}",
            f"Platform Architecture:            {platform.platform()}",
            f"NumPy Version:                    {np.version.full_version}",
            f"OpenCV Version:                   {cv2.getVersionString()}",
            f"MatPlotLib Version:               {matplotlib.__version__}",
            f"...",
            f"---------- Analysis Timing Information ----------",
            f"Analysis Started At:              {self._StartTime.strftime(TimeFormat)}",
            f"Analysis Finished At:             {datetime.datetime.now().strftime(TimeFormat)}",
            f"...",
            f"---------- Command-Line Arguments ----------",
            f"...",
            f"---------- Tuning Parameters & Variables ----------",
            f"...",
            f"---------- Results & Metrics ----------",
            f"...",
            f"---------- Enable/Disable Flags ----------",
            f"Dry-Run Mode:                     {self.EnableDryRun}",
            f"Validate Arguments Only:          {self.ValidateArguments}",
            f"...",
            f"---------- Output Artefacts ----------",
            f"Artefact Directory:               {self.OutputDirectory()}",
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

        #   ...

        return f""

    ### ----- End Private Methods -----

#   Set the logger for this application to be a default-initialized logger
LogWriter: Logger.Logger = Logger.Logger(Prefix=os.path.basename(sys.argv[0]))

#   Initialize the top-level configuration object for this program
Config: Configuration = Configuration(LogWriter=LogWriter)

#   Main
#       This is the main entry point of the script.
def main() -> int:

    LogWriter.Println(f"Beginning Analysis...")

    #   ...

    LogWriter.Println(f"Finished Analysis!")

    LogWriter.Println(f"Writing analysis configuration settings...")
    LogWriter.Write(Config.DumpConfiguration())
    return 0

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
    Flags.add_argument("--log-file", dest="LogFile", metavar="file-path", type=str, required=False, default="-", help="File path to the file to write all log messages of this program to.")
    Flags.add_argument("--quiet",    dest="Quiet",   action="store_true",  required=False, default=False, help="Enable quiet mode, disabling logging of eveything except fatal errors.")

    Flags.add_argument("--dry-run",  dest="DryRun",   action="store_true", required=False, default=False, help="Enable dry-run mode, where no file-system alterations or heavy computations are performed.")
    Flags.add_argument("--validate", dest="Validate", action="store_true", required=False, default=False, help="Only validate the command-line arguments, do not proceed with the remainder of the program.")
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
