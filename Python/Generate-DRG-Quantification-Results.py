#!/usr/bin/env python3

#   Author: Joseph Sadden
#   Date:   29th April, 2025

#   Script Purpose: This script implements the process of converting the individual
#                       JSON results files generated from analyzing DRG images
#                       into the full set of figures to compare the effects of
#                       the experimental additives and gels to identify the optimal
#                       conditions for DRG growth.

#   Import the necessary standard library modules
from __future__ import annotations
import typing

import argparse
import os
#   ...

#   Import the necessary third-part modules
#   ...

#   Import the desired locally written modules
from MTG_Common import Logger
from MTG_Common.DRG_Quantification import *
#   ...

#   Define the globals to set by the command-line arguments
#   ...

#   Main
#       This is the main entry point of the script.
def main() -> None:

    #   Prepare the available command-line flags this script will accept.
    #   These are the source and destination directories, where JSON files will
    #   be read from and where the output figures will be written to.
    Flags: argparse.ArgumentParser = argparse.ArgumentParser()
    Flags.add_argument("--source", dest="Source", metavar="file-path", type=str, required=False, default=None, help="The full path to the directory to read for source JSON files to process.")
    Flags.add_argument("--destination", dest="Destination", metavar="file-path", type=str, required=False, default=None, help="The full path to the directory to write the processed results out to.")

    #   Parse the provided command-line flags and store the values for use with the remainder of this script.
    Arguments: argparse.Namespace = Flags.parse_args()
    SourceDirectory: str = Arguments.Source
    DestinationDirectory: str = Arguments.Destination

    #   If no destination directory is given, create a suitably named folder in the
    #   current working directory
    if ( DestinationDirectory is None ) or ( DestinationDirectory == "" ):
        DestinationDirectory = os.path.join(os.getcwd(), "DRG Quantification Summary")

    #   Prepare the set of results to work with, including a logger to write out helpful messages during processing.
    Results: DRGQuantificationResultsSet = DRGQuantificationResultsSet(LogWriter=Logger.Logger())

    #   If a source directory has been provided, attempt to read all of the JSON files within
    #   the directory, parsing out an instance of a DRGQuantificationResults for each file.
    if ( SourceDirectory is not None ):
        Results = Results.ReadDirectory(SourceDirectory)
        for Result in Results:
            Result.MedianNeuriteDistance = float(list(Result.MedianNeuriteDistance.values())[0][0])
    else:
        #   Otherwise, if no source directory is given, generate a set of random values
        #   to allow testing and validating of the figure generation and internal logic.
        RandomResultCount: int = 10000
        [Results.Add(DRGQuantificationResults.GenerateRandom()) for n in range(RandomResultCount)]

    #   Call the top-level function which will summarize all of the results into the set of desired figures.
    Results.Summarize(OutputDirectory=DestinationDirectory)

    return

#   Allow this script to be called from the command-line and execute the main function.
#   If anything needs to happen before executing main, add it here before the call.
if __name__ == "__main__":
    main()
