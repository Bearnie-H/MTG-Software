#!/usr/bin/env python3

#   Author: Joseph Sadden
#   Date:   21st March, 2025

#   Script Purpose: This script provides the interface to run
#                       the DRG_Neurite_Quantification.py script
#                       over a large number of LIF files, using
#                       details and conditions as described in a
#                       dedicated spreadsheet.

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
from MTG_Common import Logger
from MTG_Common.DRG_Quantification import *
from MTG_Common import Utils
import DRG_Neurite_Quantification
import MTG_Common.DRG_Quantification
#   ...

#   Define the globals to set by the command-line arguments
JSONDirectory: str = ""
#   ...

LogWriter: Logger.Logger = Logger.Logger(Prefix="DRG Neurite Quantification Batch Analysis")

#   Main
#       This is the main entry point of the script.
def main() -> None:

    global JSONDirectory

    #   Prepare the two command-line flags (so far) this tool will accept.
    Flags: argparse.ArgumentParser = argparse.ArgumentParser()

    Flags.add_argument("--spreadsheet", dest="Spreadsheet", metavar="file-path", type=str, required=True, help="The file path to the *.CSV file containing all of the experimental conditions to process.")
    Flags.add_argument("--folder-base", dest="FolderBase", metavar="file-path", type=str, required=True, help="The path to the base folder from which the \"FilePath\" column of the spreadsheet is referenced.")
    Flags.add_argument("--json-directory", dest="JSONDirectory", metavar="file-path", type=str, required=True, help="The path to the folder in which all of the compiled JSON results will be written.")
    Flags.add_argument("--pre-check", dest="ManualPreCheck", action="store_true", required=False, default=False, help="Manually preview the image results to check for whether or not the images should even be processed.")

    Arguments: argparse.Namespace = Flags.parse_args()

    InputFile: str = Arguments.Spreadsheet
    FolderBase: str = Arguments.FolderBase
    ManualPreview: bool = Arguments.ManualPreCheck
    JSONDirectory = Arguments.JSONDirectory

    #   Parse the spreadsheet, identifying and validating all of the experimental conditions
    #   described within.
    ExperimentalConditions: typing.Sequence[DRGExperimentalCondition] = ParseSpreadsheet(InputFile, FolderBase)

    StatusReportFilename: str = f'DRG Batch Analysis Status Reporting - {datetime.now().strftime("%Y-%m-%d")}.csv'
    LogWriter.Println(f"Creating status report file [ {StatusReportFilename} ] to track execution status of experimental conditions...")
    with open(os.path.join(FolderBase, StatusReportFilename), "+w") as StatusReport:
        StatusReport.write(f"Analysis File,Execution Status,Status Code\n")

        if ( ManualPreview ):
            ExperimentalConditions = ManuallyPreviewConditions(ExperimentalConditions, StatusReport)

        #   For each of the valid conditions, actually process the LIF file.
        AnalyzeConditions(ExperimentalConditions, StatusReport)

    #   If the JSON folder has been provided, run the summarization logic to generate the output plots and figures
    DRGQuantificationResultsSet.FromDirectory(JSONDirectory).Summarize(os.path.join(JSONDirectory, "Summarized Results"))

    return

def ParseSpreadsheet(FilePath: str, FolderBase: str) -> typing.Sequence[DRGExperimentalCondition]:
    """
    ParseSpreadsheet

    This function reads the provided spreadsheet line by line, splitting out the
    fields of each row in commas and attempting to parse out each row into an
    instance of the DRGExperimentalCondition class.

    FilePath:
        The full file path to the spreadsheet file to read and parse data from.
    FolderBase:
        The folder from which all of the LIF files in the spreadsheet should be referenced from.

    Return (Sequence[DRGExperimentalCondition]):
        A sequence (list) of DRGExperimentalCondition instances which have been
        parsed and validated from the spreadsheet, ready to be further processed.
    """

    ExperimentalConditions: typing.Sequence[DRGExperimentalCondition] = []

    with open(FilePath, "r") as Spreadsheet:
        for RowIndex, Row in enumerate(Spreadsheet.readlines()[1:], start=1):

            LogWriter.Println(f"Parsing row [ {RowIndex} ] for experimental details...")
            Condition: DRGExperimentalCondition = DRGExperimentalCondition().ExtractFields(Row.strip().split(",")).SetFolderBase(FolderBase)
            if ( Condition.Validate() ):
                LogWriter.Println(f"Successufully validated row [ {RowIndex} ].")
                Condition.AnalysisStatus = DRGAnalysis_StatusCode(DRGAnalysis_StatusCode.StatusNotYetProcessed)
            else:
                LogWriter.Errorln(f"Failed to validate row [ {RowIndex} ]!")
                Condition.AnalysisStatus = DRGAnalysis_StatusCode(DRGAnalysis_StatusCode.StatusValidationFailed)

            ExperimentalConditions.append(Condition)

    return ExperimentalConditions

def ManuallyPreviewConditions(ExperimentalConditions: typing.Sequence[DRGExperimentalCondition], StatusReport: typing.TextIO) -> typing.Sequence[DRGExperimentalCondition]:
    """
    ManuallyPreviewConditions

    This function...

    ExperimentalCondition:
        ...

    Return (Sequence[DRGExperimentalCondition]):
        ...
    """

    global JSONDirectory

    ConditionCount: int = len(ExperimentalConditions)

    for ConditionIndex, Condition in enumerate(ExperimentalConditions, start=1):

        if ( Condition.AnalysisStatus & DRGAnalysis_StatusCode.StatusValidationFailed == 0 ) and ( Condition.SkipProcessing == False ):

            LogWriter.Println(f"Starting manual preview of experimental condition [ {ConditionIndex}/{ConditionCount} ] - [ {os.path.basename(Condition.LIFFilePath)} ]...")
            try:

                DRG_Neurite_Quantification.LogWriter = Logger.Logger(OutputStream=LogWriter.RawStream(), Prefix=f"DRG Batch Analysis (Manual Preview) - {os.path.basename(Condition.LIFFilePath)} ({ConditionIndex}/{ConditionCount})", AlwaysFlush=True)
                DRG_Neurite_Quantification.Config = DRG_Neurite_Quantification.Configuration(LogWriter=DRG_Neurite_Quantification.LogWriter).ExtractFromCondition(Condition)
                DRG_Neurite_Quantification.Config.ManualPreview = True
                DRG_Neurite_Quantification.Config.OutputDirectory = os.path.splitext(Condition.LIFFilePath)[0] + f" - Analyzed {datetime.now().strftime('%Y-%m-%d %H-%M-%S')}"
                DRG_Neurite_Quantification.Config.JSONDirectory = JSONDirectory
                DRG_Neurite_Quantification.QuantificationStacks = DRG_Neurite_Quantification.QuantificationIntermediates(LogWriter=DRG_Neurite_Quantification.LogWriter)
                DRG_Neurite_Quantification.Results = MTG_Common.DRG_Quantification.DRGQuantificationResults()

                if ( DRG_Neurite_Quantification.main() == DRGAnalysis_StatusCode.StatusSuccess ):
                    LogWriter.Println(f"Preview accepted for experimental condition [ {ConditionIndex}/{ConditionCount} ] - [ {os.path.basename(Condition.LIFFilePath)} ].")
                else:
                    LogWriter.Errorln(f"Preview rejected for experimental condition [ {ConditionIndex}/{ConditionCount} ] - [ {os.path.basename(Condition.LIFFilePath)} ] - [ {str(Condition.AnalysisStatus)} ({int(Condition.AnalysisStatus)})].")
                    Condition.AnalysisStatus = DRGAnalysis_StatusCode(DRGAnalysis_StatusCode.StatusPreviewRejected)
            except Exception as e:
                LogWriter.Errorln(f"Exception raised in row ({ConditionIndex}/{ConditionCount}): [ {e} ]\n\n{''.join(traceback.format_exception(e, value=e, tb=e.__traceback__))}")
                Condition.AnalysisStatus |= DRGAnalysis_StatusCode(DRGAnalysis_StatusCode.StatusUnknownException)
                Condition.AnalysisStatus &= ~DRGAnalysis_StatusCode(DRGAnalysis_StatusCode.StatusNotYetProcessed)

        if ( Condition.SkipProcessing ):
            LogWriter.Println(f"Skipping analysis of experimental condition [ {ConditionIndex}/{ConditionCount} ] - [ {os.path.basename(Condition.LIFFilePath)} ]...")
            Condition.AnalysisStatus = DRGAnalysis_StatusCode(DRGAnalysis_StatusCode.StatusSkipped)
        else:
            LogWriter.Println(f"Analysis of experimental condition [ {ConditionIndex}/{ConditionCount} ] - [ {os.path.basename(Condition.LIFFilePath)} ] already failed validation...")

        StatusReport.write(f"{Condition.LIFFilePath},{str(Condition.AnalysisStatus)},{int(Condition.AnalysisStatus)}\n")
        StatusReport.flush()

    return ExperimentalConditions

def AnalyzeConditions(ExperimentalConditions: typing.Sequence[DRGExperimentalCondition], StatusReport: typing.TextIO) -> None:
    """
    AnalyzeConditions

    This function passes the details from the DRGExperimentalCondition to the
    DRG_Neurite_Quantification.py script to run the analysis on the provided
    LIF file.

    ExperimentalConditions:
        The set of DRGExperimentalCondition instances parsed and validated,
        and ready to be processed.

    Return (None):
        None, the analysis generates it's own outputs and stores them
        based off the path to the LIF file being processed.
    """

    global JSONDirectory

    ConditionCount: int = len(ExperimentalConditions)
    for ConditionIndex, Condition in enumerate(ExperimentalConditions, start=1):

        if (( Condition.AnalysisStatus & DRGAnalysis_StatusCode.StatusValidationFailed ) == 0 ) and (( Condition.AnalysisStatus & DRGAnalysis_StatusCode.StatusPreviewRejected ) == 0 )  and ( Condition.SkipProcessing == False ):

            LogWriter.Println(f"Starting analysis of experimental condition [ {ConditionIndex}/{ConditionCount} ] - [ {os.path.basename(Condition.LIFFilePath)} ]...")
            try:

                DRG_Neurite_Quantification.LogWriter = Logger.Logger(OutputStream=LogWriter.RawStream(), Prefix=f"DRG Batch Analysis - {os.path.basename(Condition.LIFFilePath)} ({ConditionIndex}/{ConditionCount})", AlwaysFlush=True)
                DRG_Neurite_Quantification.Config = DRG_Neurite_Quantification.Configuration(LogWriter=DRG_Neurite_Quantification.LogWriter).ExtractFromCondition(Condition)
                DRG_Neurite_Quantification.Config.ManualPreview = False
                DRG_Neurite_Quantification.Config.OutputDirectory = os.path.splitext(Condition.LIFFilePath)[0] + f" - Analyzed {datetime.now().strftime('%Y-%m-%d %H-%M-%S')}"
                DRG_Neurite_Quantification.Config.JSONDirectory = JSONDirectory
                DRG_Neurite_Quantification.QuantificationStacks = DRG_Neurite_Quantification.QuantificationIntermediates(LogWriter=DRG_Neurite_Quantification.LogWriter)
                DRG_Neurite_Quantification.Results = MTG_Common.DRG_Quantification.DRGQuantificationResults()
                DRG_Neurite_Quantification.Results.ExtractExperimentalDetails(Condition)
                DRG_Neurite_Quantification.Results.SourceHash = Utils.Sha256Sum(Condition.LIFFilePath)

                Condition.AnalysisStatus = DRGAnalysis_StatusCode(DRG_Neurite_Quantification.main())
                if ( Condition.AnalysisStatus == DRGAnalysis_StatusCode.StatusSuccess ):
                    LogWriter.Println(f"Successfully finished analysis of experimental condition [ {ConditionIndex}/{ConditionCount} ] - [ {os.path.basename(Condition.LIFFilePath)} ].")
                else:
                    LogWriter.Errorln(f"Analysis failed for experimental condition [ {ConditionIndex}/{ConditionCount} ] - [ {os.path.basename(Condition.LIFFilePath)} ] - [ {str(Condition.AnalysisStatus)} ({int(Condition.AnalysisStatus)})].")
            except Exception as e:
                LogWriter.Errorln(f"Exception raised in row ({ConditionIndex}/{ConditionCount}): [ {e} ]\n\n{''.join(traceback.format_exception(e, value=e, tb=e.__traceback__))}")
                Condition.AnalysisStatus |= DRGAnalysis_StatusCode(DRGAnalysis_StatusCode.StatusUnknownException)
                Condition.AnalysisStatus &= ~DRGAnalysis_StatusCode(DRGAnalysis_StatusCode.StatusNotYetProcessed)

        if ( Condition.SkipProcessing ):
            Condition.AnalysisStatus = DRGAnalysis_StatusCode(DRGAnalysis_StatusCode.StatusSkipped)
            LogWriter.Println(f"Skipping analysis of experimental condition [ {ConditionIndex}/{ConditionCount} ] - [ {os.path.basename(Condition.LIFFilePath)} ].")
            MTG_Common.DRG_Quantification.DRGQuantificationResults().ExtractExperimentalDetails(Condition).Save(JSONDirectory)
        else:
            LogWriter.Println(f"Analysis of experimental condition [ {ConditionIndex}/{ConditionCount} ] - [ {os.path.basename(Condition.LIFFilePath)} ] already failed validation or manual preview...")

        StatusReport.write(f"{Condition.LIFFilePath},{str(Condition.AnalysisStatus)},{int(Condition.AnalysisStatus)}\n")
        StatusReport.flush()

    return ExperimentalConditions


#   Allow this script to be called from the command-line and execute the main function.
#   If anything needs to happen before executing main, add it here before the call.
if __name__ == "__main__":
    main()
