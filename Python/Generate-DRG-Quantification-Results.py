#!/usr/bin/env python3

#   Author: Joseph Sadden
#   Date:   29th April, 2025

#   Script Purpose: This script...

#   Import the necessary standard library modules
from __future__ import annotations
import typing

import argparse
import os
#   ...

#   Import the necessary third-part modules
#   ...

#   Import the desired locally written modules
from MTG_Common.DRG_Quantification import *
#   ...

#   Define the globals to set by the command-line arguments
#   ...

#   Main
#       This is the main entry point of the script.
def main() -> None:

    Flags: argparse.ArgumentParser = argparse.ArgumentParser()

    Flags.add_argument("--source", dest="Source", metavar="file-path", type=str, required=False, default=None, help="The full path to the directory to read for source JSON files to process.")
    Flags.add_argument("--destination", dest="Destination", metavar="file-path", type=str, required=False, default=None, help="The full path to the directory to write the processed results out to.")

    Arguments: argparse.Namespace = Flags.parse_args()

    SourceDirectory: str = Arguments.Source
    DestinationDirectory: str = Arguments.Destination

    if ( DestinationDirectory is None ) or ( DestinationDirectory == "" ):
        DestinationDirectory = os.path.join(os.getcwd(), "DRG Quantification Summary")

    Results: DRGQuantificationResultsSet = None
    if ( SourceDirectory is not None ):
        Results = DRGQuantificationResultsSet.FromDirectory(SourceDirectory)
    else:
        RandomResultCount: int = 10000
        Results = DRGQuantificationResultsSet([DRGQuantificationResults.GenerateRandom() for n in range(RandomResultCount)])

    Results.SetLogger(Logger.Logger())
    Results.Summarize(OutputDirectory=DestinationDirectory)

    return

#   Allow this script to be called from the command-line and execute the main function.
#   If anything needs to happen before executing main, add it here before the call.
if __name__ == "__main__":
    main()
