#!/usr/bin/env python3

#   Author: Joseph Sadden
#   Date:   21st March, 2025

#   Script Purpose: ...
#                       ...

#   Import the necessary standard library modules
from __future__ import annotations
import typing

import argparse
import os
import traceback
#   ...

#   Import the necessary third-part modules
#   ...

#   Import the desired locally written modules
from MTG_Common.Logger import Logger
from MTG_Common.DRG_Quantification import DRGExperimentalCondition
import DRG_Neurite_Quantification
#   ...

#   Define the globals to set by the command-line arguments
#   ...

LogWriter: Logger = Logger(Prefix="DRG Neurite Quantification Batch Analysis")

#   Main
#       This is the main entry point of the script.
def main() -> None:

    Flags: argparse.ArgumentParser = argparse.ArgumentParser()

    Flags.add_argument("--spreadsheet", dest="Spreadsheet", metavar="file-path", type=str, required=True, help="The file path to the *.CSV file containing all of the experimental conditions to process.")
    Flags.add_argument("--folder-base", dest="FolderBase", metavar="file-path", type=str, required=True, help="The path to the base folder from which the \"FilePath\" column of the spreadsheet is referenced.")

    Arguments: argparse.Namespace = Flags.parse_args()

    InputFile: str = Arguments.Spreadsheet
    FolderBase: str = Arguments.FolderBase

    ExperimentalConditions: typing.Sequence[DRGExperimentalCondition] = ParseSpreadsheet(InputFile, FolderBase)

    AnalyzeConditions(ExperimentalConditions)

    return

def ParseSpreadsheet(FilePath: str, FolderBase: str) -> typing.Sequence[DRGExperimentalCondition]:
    """
    ParseSpreadsheet

    This function...

    """

    ExperimentalConditions: typing.Sequence[DRGExperimentalCondition] = []

    with open(FilePath, "r") as Spreadsheet:
        for RowIndex, Row in enumerate(Spreadsheet.readlines()[1:]):

            LogWriter.Println(f"Parsing row [ {RowIndex+1} ] for experimental details...")
            Condition: DRGExperimentalCondition = DRGExperimentalCondition().ExtractFields(Row.strip().split(",")).SetFolderBase(FolderBase)
            if ( Condition.Validate() ):
                ExperimentalConditions.append(Condition)
                # Condition.Describe()
                LogWriter.Println(f"Successufully parsed row [ {RowIndex+1} ].")
            else:
                LogWriter.Errorln(f"Failed to parse row [ {RowIndex+1} ]!")

    return ExperimentalConditions

def AnalyzeConditions(ExperimentalConditions: typing.Sequence[DRGExperimentalCondition]) -> None:
    """
    AnalyzeConditions

    This function...

    """

    ConditionCount: int = len(ExperimentalConditions)

    for ConditionIndex, Condition in enumerate(ExperimentalConditions):
        LogWriter.Println(f"Starting analysis of experimental condition [ {ConditionIndex+1}/{ConditionCount} ] - [ {os.path.basename(Condition.LIFFilePath)} ]...")
        try:
            DRG_Neurite_Quantification.LogWriter = Logger(OutputStream=LogWriter.RawStream(), Prefix=f"DRG Batch Analysis - {os.path.basename(Condition.LIFFilePath)} ({ConditionIndex+1}/{ConditionCount})")
            DRG_Neurite_Quantification.Config = DRG_Neurite_Quantification.Configuration(LogWriter=DRG_Neurite_Quantification.LogWriter).ExtractFromCondition(Condition)
            DRG_Neurite_Quantification.Results = DRG_Neurite_Quantification.QuantificationResults(LogWriter=DRG_Neurite_Quantification.LogWriter)
            if ( DRG_Neurite_Quantification.main() == 0 ):
                LogWriter.Println(f"Finished analysis of experimental condition [ {ConditionIndex+1}/{ConditionCount} ] - [ {os.path.basename(Condition.LIFFilePath)} ].")
            else:
                LogWriter.Errorln(f"Analysis failed for experimental condition [ {ConditionIndex+1}/{ConditionCount} ] - [ {os.path.basename(Condition.LIFFilePath)} ].")
        except Exception as e:
            LogWriter.Errorln(f"Exception raised in row ({ConditionIndex+1}/{ConditionCount}): [ {e} ]\n\n{''.join(traceback.format_exception(e, value=e, tb=e.__traceback__))}")

    return


#   Allow this script to be called from the command-line and execute the main function.
#   If anything needs to happen before executing main, add it here before the call.
if __name__ == "__main__":
    main()
