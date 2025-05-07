#!/usr/bin/env python3

#   Author: Joseph Sadden
#   Date:   20th March, 2025

#   Script Purpose: This file intends to collect all of the helper functions,
#                       and classes used for working with the DRG Neurite Quantitication
#                       experiments for Mend the Gap

#   Import the necessary standard library modules
from __future__ import annotations
import typing

from datetime import datetime
import glob
import itertools
import jsonpickle
import math
import os
import random
import str2bool
#   ...

#   Import the necessary third-part modules
import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.cm
#   ...

#   Import the desired locally written modules
from . import Utils
from . import Logger
#   ...

#   Helper parsing functions to wrap and error handling in a consistent manner
def NormalizePathSeparators(Path: str, Separator: str = "/") -> str:
    """
    NormalizePathSeparators

    This function normalizes the path separators within the file path for the LIF
    file to process, so that differences due to platform or file system type
    do not affect the ability to find the files to process.

    Path:
        The current, unsantized and normalized path to normalize
    Separator:
        The path separator to use in the final normalized path. If empty,
        the default for the currently running system will be used.

    Return (str):
        The resulting normalized and sanitized file path with consistent
        path separators.
    """

    if ( Separator is None ) or ( Separator == "" ):
        Separator = os.sep

    TemporaryPath: typing.Sequence[str] = '\\'.join([k for k in Path.split('/') if len(k) > 0])
    FinalPath: str = Separator.join([k for k in TemporaryPath.split('\\') if len(k) > 0])

    return FinalPath

def TryParseDatetime(Input: str, Format: str, ErrorMessage: str = "") -> datetime | None:
    """
    TryParseDatetime

    This function attempts to parse a datetime object from a given string,
    according to a provided format string. See documentation for strptime().

    Input:
        The raw datestamp string to parse.
    Format:
        The format string describing how to parse the Input string.
        See strptime().
    ErrorMessage:
        An optional error message to print to stdout if the parsing fails.

    Return (datetime | None):
        If the Input string is successfully parsed according to Format,
        a valid datetime object is returned, otherwise None.
    """

    try:
        return datetime.strptime(Input, Format)
    except Exception as e:
        if ( ErrorMessage is not None ) and ( ErrorMessage != "" ):
            print(f"Datetime Parse Error: {ErrorMessage} - {e}")
        return None

def TryParseInteger(Input: str, ErrorMessage: str = "") -> int | None:
    """
    TryParseInteger

    This function attempts to parse Input as an integer value.

    Input:
        The raw string to attempt to parse as an integer.
    ErrorMessage:
        An optional error message to print to stdout if the parsing fails.

    Return (int | None):
        The resulting integer value, if successfully parsed, otherwise None.
        An empty or missing Input string defaults to 0.
    """

    if ( Input is None ) or ( Input == "" ):
        print(f"Note: Integer Value Empty, Defaulting to [ 0 ]: {ErrorMessage}.")
        Input = "0"

    try:
        return int(Input)
    except Exception as e:
        if ( ErrorMessage is not None ) and ( ErrorMessage != "" ):
            print(f"Integer Parse Error: {ErrorMessage} - {e}")
        return None

def TryParseFloat(Input: str, ErrorMessage: str = "") -> float | None:
    """
    TryParseFloat

    This function attempts to parse the Input string as a floating point
    value.

    Input:
        The raw string to attempt to parse.
    ErrorMessage:
        An optional error message to print to stdout if the parsing fails.

    Return (float | None):
        The resulting floating point value, if successfully parsed, otherwise None.
        An empty or missing Input string defaults to 0.0.
    """

    if ( Input is None ) or ( Input == "" ):
        print(f"Note: Float Value Empty, Defaulting to [ 0.0 ]: {ErrorMessage}.")
        Input = "0.0"

    try:
        return float(Input)
    except Exception as e:
        if ( ErrorMessage is not None ) and ( ErrorMessage != "" ):
            print(f"Float Parse Error: {ErrorMessage} - {e}")
        return None

def TryParseBool(Input: str, ErrorMessage: str = "") -> bool | None:
    """
    TryParseBool

    This function attempts to parse the given Input string as a boolean value.

    Input:
        The raw string to attempt to parse, accoring to str2bool().
    ErrorMessage:
        An optional error message to print to stdout if the parsing fails.

    Return (bool | None):
        The resulting boolean value, if successfully parsed, otherwise None.
        An empty or missing Input string defaults to False.
    """

    if ( Input is None ) or ( Input == "" ):
        print(f"Note: Boolean Value Empty, Defaulting to [ False ]: {ErrorMessage}.")
        Input = "False"

    try:
        return str2bool.str2bool(Input.lower(), True)
    except Exception as e:
        if ( ErrorMessage is not None ) and ( ErrorMessage != "" ):
            print(f"Boolean Parse Error: {ErrorMessage} - {e}")
        return None

def TryParseString(Input: str) -> str | None:
    """
    TryParseString

    This function maintains the pattern of the TryParse_() functions, but for strings.
    As we don't care about the content during parsing, this simply guarantees that empty strings
    are replaced with None.

    Input:
        The raw string to check.

    Return (str | None):
        Either the original string, or None if empty.
    """

    if ( Input is None ) or ( Input == "" ):
        return None

    return Input

class DRGAnalysis_StatusCode():
    StatusSuccess:          int = 1 << 0
    StatusNotYetProcessed:  int = 1 << 1
    StatusValidationFailed: int = 1 << 2
    StatusNoLIFFile:        int = 1 << 3
    NoBrightFieldImage:     int = 1 << 4
    NoFluorescentImage:     int = 1 << 5
    StatusPreviewAccepted:  int = 1 << 6
    StatusPreviewRejected:  int = 1 << 7
    StatusBodyMaskFailed:   int = 1 << 8
    StatusWellMaskFailed:   int = 1 << 9
    StatusNoNeurites:       int = 1 << 10
    StatusUnknownException: int = 1 << 11
    StatusIntentionalAbort: int = 1 << 12
    StatusSkipped:          int = 1 << 13

    def __str__(self: DRGAnalysis_StatusCode) -> str:

        StatusCodeMapping: typing.Dict[int, str] = {
            DRGAnalysis_StatusCode.StatusSuccess:              "Success.",
            DRGAnalysis_StatusCode.StatusNotYetProcessed:      "Not Yet Processed.",
            DRGAnalysis_StatusCode.StatusValidationFailed:     "Parameter Validation Failure.",
            DRGAnalysis_StatusCode.StatusNoLIFFile:            "LIF File Could Not Be Found.",
            DRGAnalysis_StatusCode.NoBrightFieldImage:         "No Bright Field Image In LIF File.",
            DRGAnalysis_StatusCode.NoFluorescentImage:         "No Fluorescent Image In LIF File.",
            DRGAnalysis_StatusCode.StatusPreviewAccepted:      "Manual Preview Accepted.",
            DRGAnalysis_StatusCode.StatusPreviewRejected:      "Rejected During Manual Preview.",
            DRGAnalysis_StatusCode.StatusBodyMaskFailed:       "DRG Body Mask Generation Failure.",
            DRGAnalysis_StatusCode.StatusWellMaskFailed:       "Well Interior Mask Generation Failure.",
            DRGAnalysis_StatusCode.StatusNoNeurites:           "No Neurites Identified.",
            DRGAnalysis_StatusCode.StatusUnknownException:     "Unknown Exception Occurred.",
            DRGAnalysis_StatusCode.StatusIntentionalAbort:     "Intentionally Ended Early.",
            DRGAnalysis_StatusCode.StatusSkipped:              "Analysis Skipped.",
        }

        Output: str = ""
        for Code in sorted(StatusCodeMapping.keys()):
            if (( Code & int(self) ) != 0 ):
                if ( Output == "" ):
                    Output = StatusCodeMapping[Code]
                else:
                    Output += f" {StatusCodeMapping[Code]}"

        if ( Output == "" ):
            Output = "Unknown Status."

        return Output

class BaseGels():
    BaseGel_Ultimatrix: str = "Ultimatrix"

    BaseGel_GelMA: str = "GelMA"

    BaseGel_H6: str = "?H6?"
    BaseGel_H7: str = "?H7?"
    BaseGel_H8: str = "?H8?"

    BaseGel_PH15: str = "?PH15?"
    BaseGel_PH16: str = "?PH16?"
    BaseGel_PH18: str = "?PH18?"
    BaseGel_PH19: str = "?PH19?"

class GelMACrosslinkers():
    GelMA_RuSPS: str = "RuSPS"
    GelMA_Riboflavin: str = "Riboflavin"

class NasrinCrosslinker():
    NasrinGel_A: str = "??"

class DRGExperimentalCondition():
    """
    DRGExperimentalCondition

    This class represents the full set of experimental variables and conditions
    for the DRG neurite growth body of work.
    """

    LIFFilePath: str

    ExperimentDate: datetime

    CultureDuration: int

    SampleIndex: int

    # ImagedBy: str
    # ImagedFor: str

    ImageResolution: float  #   µm/px

    BrightFieldSeriesIndex: int
    BrightFieldChannelIndex: int

    NeuriteSeriesIndex: int
    NeuriteChannelIndex: int

    BaseGel: str

    #   Only applicable for GelMA samples
    GelMAPercentage: int
    DegreeOfFunctionalization: int

    Polymer: str    #   For Nasrin, to do with how her gels crosslink?

    Crosslinker: str    #   RuSPS or Riboflavin

    Peptide: str
    PeptideIn: str
    PeptideConcentration: float

    DilutionMedia: str
    IncludesPhenolRed: bool
    B27Inclusion: bool
    FBSInclusion: bool

    GelIlluminationTime: float  #   Seconds

    RutheniumConcentration: float
    SodiumPersulfateConcentration: bool
    RiboflavinConcentration: float

    IKVAV: bool
    IKVAVConcentration: float

    Gelatin: bool
    GelatinConcentration: float

    Glutathione: bool
    GlutathioneConcentration: float

    GDNF: bool
    GDNFConcentration: float

    BDNF: bool
    BDNFConcentration: float

    Laminin: bool
    LamininConcentration: float

    ###
    SkipProcessing: bool
    AnalysisStatus: DRGAnalysis_StatusCode


    ### Magic Methods
    def __init__(self: DRGExperimentalCondition) -> None:
        """
        Constructor

        This function...

        Return (None):
            ...
        """

        #   ...

        self.SkipProcessing = False

        return

    ### Public Methods
    def ExtractFields(self: DRGExperimentalCondition, Fields: typing.Sequence[str]) -> DRGExperimentalCondition:
        """
        ExtractFields

        This function extracts the values for this class from the top-level spreadsheet
        defining the set of conditions and variables used for this experiments. This is
        a simple in-order process where the sequence of columns MUST match the order in
        which they are pulled rom the provided fields sequence. These values are only
        validated in terms of data type, with no logic to check for value. Missing or empty
        fields will be default initialized to a sensible and consistent "zero-value".

        Fields:
            The sequence of fields to use to fill out the member variables of this
            particular class instance. These must be in the order as described below,
            and the body of this function MUST be modified to account for any added or removed
            columns, and ensure that the ordering matches.

        Return (self):
            The same experimental condition instance is returned, allowing chaining of operations.
        """

        ColumnIndex: int = 0

        self.LIFFilePath, ColumnIndex                   = (NormalizePathSeparators(Fields[ColumnIndex])),                                              (ColumnIndex + 1)
        self.SkipProcessing, ColumnIndex                = (TryParseBool(           Fields[ColumnIndex], "Column: [ SkipProcessing ]")),                (ColumnIndex + 1)
        self.ExperimentDate, ColumnIndex                = (TryParseDatetime(       Fields[ColumnIndex], "%Y%m%d")),                                    (ColumnIndex + 1)
        self.CultureDuration, ColumnIndex               = (TryParseInteger(        Fields[ColumnIndex], "Column: [ CultureDuration ]")),               (ColumnIndex + 1)
        self.SampleIndex, ColumnIndex                   = (TryParseInteger(        Fields[ColumnIndex], "Column: [ SampleIndex ]")),                   (ColumnIndex + 1)
        self.ImageResolution, ColumnIndex               = (TryParseFloat(          Fields[ColumnIndex], "Column: [ ImageResolution ]")),               (ColumnIndex + 1)
        self.BrightFieldSeriesIndex, ColumnIndex        = (TryParseInteger(        Fields[ColumnIndex], "Column: [ BrightFieldSeriesIndex ]")),        (ColumnIndex + 1)
        self.BrightFieldChannelIndex, ColumnIndex       = (TryParseInteger(        Fields[ColumnIndex], "Column: [ BrightFieldChannelIndex ]")),       (ColumnIndex + 1)
        self.NeuriteSeriesIndex, ColumnIndex            = (TryParseInteger(        Fields[ColumnIndex], "Column: [ NeuriteSeriesIndex ]")),            (ColumnIndex + 1)
        self.NeuriteChannelIndex, ColumnIndex           = (TryParseInteger(        Fields[ColumnIndex], "Column: [ NeuriteChannelIndex ]")),           (ColumnIndex + 1)
        self.BaseGel, ColumnIndex                       = (TryParseString(         Fields[ColumnIndex])),                                              (ColumnIndex + 1)
        self.GelMAPercentage, ColumnIndex               = (TryParseInteger(        Fields[ColumnIndex], "Column: [ GelMAPercentage ]")),               (ColumnIndex + 1)
        self.DegreeOfFunctionalization, ColumnIndex     = (TryParseInteger(        Fields[ColumnIndex], "Column: [ DegreeOfFunctionalization ]")),     (ColumnIndex + 1)
        self.Polymer, ColumnIndex                       = (TryParseString(         Fields[ColumnIndex])),                                              (ColumnIndex + 1)
        self.Crosslinker, ColumnIndex                   = (TryParseString(         Fields[ColumnIndex])),                                              (ColumnIndex + 1)
        self.Peptide, ColumnIndex                       = (TryParseString(         Fields[ColumnIndex])),                                              (ColumnIndex + 1)
        self.PeptideIn, ColumnIndex                     = (TryParseString(         Fields[ColumnIndex])),                                              (ColumnIndex + 1)
        self.PeptideConcentration, ColumnIndex          = (TryParseFloat(          Fields[ColumnIndex], "Column: [ PeptideConcentration ]")),          (ColumnIndex + 1)
        self.DilutionMedia, ColumnIndex                 = (TryParseString(         Fields[ColumnIndex])),                                              (ColumnIndex + 1)
        self.IncludesPhenolRed, ColumnIndex             = (TryParseBool(           Fields[ColumnIndex], "Column: [ IncludesPhenolRed ]")),             (ColumnIndex + 1)
        self.B27Inclusion, ColumnIndex                  = (TryParseBool(           Fields[ColumnIndex], "Column: [ B27Inclusion ]")),                  (ColumnIndex + 1)
        self.FBSInclusion, ColumnIndex                  = (TryParseBool(           Fields[ColumnIndex], "Column: [ FBSInclusion ]")),                  (ColumnIndex + 1)
        self.GelIlluminationTime, ColumnIndex           = (TryParseFloat(          Fields[ColumnIndex], "Column: [ GelIlluminationTime ]")),           (ColumnIndex + 1)
        self.RutheniumConcentration, ColumnIndex        = (TryParseFloat(          Fields[ColumnIndex], "Column: [ RutheniumConcentration ]")),        (ColumnIndex + 1)
        self.SodiumPersulfateConcentration, ColumnIndex = (TryParseFloat(          Fields[ColumnIndex], "Column: [ SodiumPersulfateConcentration ]")), (ColumnIndex + 1)
        self.RiboflavinConcentration, ColumnIndex       = (TryParseFloat(          Fields[ColumnIndex], "Column: [ RiboflavinConcentration ]")),       (ColumnIndex + 1)
        self.IKVAV, ColumnIndex                         = (TryParseBool(           Fields[ColumnIndex], "Column: [ IKVAV ]")),                         (ColumnIndex + 1)
        self.IKVAVConcentration, ColumnIndex            = (TryParseFloat(          Fields[ColumnIndex], "Column: [ IKVAVConcentration ]")),            (ColumnIndex + 1)
        self.Gelatin, ColumnIndex                       = (TryParseBool(           Fields[ColumnIndex], "Column: [ Gelatin ]")),                       (ColumnIndex + 1)
        self.GelatinConcentration, ColumnIndex          = (TryParseFloat(          Fields[ColumnIndex], "Column: [ GelatinConcentration ]")),          (ColumnIndex + 1)
        self.Glutathione, ColumnIndex                   = (TryParseBool(           Fields[ColumnIndex], "Column: [ Glutathione ]")),                   (ColumnIndex + 1)
        self.GlutathioneConcentration, ColumnIndex      = (TryParseFloat(          Fields[ColumnIndex], "Column: [ GlutathioneConcentration ]")),      (ColumnIndex + 1)
        self.GDNF, ColumnIndex                          = (TryParseBool(           Fields[ColumnIndex], "Column: [ GDNF ]")),                          (ColumnIndex + 1)
        self.GDNFConcentration, ColumnIndex             = (TryParseFloat(          Fields[ColumnIndex], "Column: [ GDNFConcentration ]")),             (ColumnIndex + 1)
        self.BDNF, ColumnIndex                          = (TryParseBool(           Fields[ColumnIndex], "Column: [ BDNF ]")),                          (ColumnIndex + 1)
        self.BDNFConcentration, ColumnIndex             = (TryParseFloat(          Fields[ColumnIndex], "Column: [ BDNFConcentration ]")),             (ColumnIndex + 1)
        self.Laminin, ColumnIndex                       = (TryParseBool(           Fields[ColumnIndex], "Column: [ LamininConcentration ]")),          (ColumnIndex + 1)
        self.LamininConcentration, ColumnIndex          = (TryParseFloat(          Fields[ColumnIndex], "Column: [ Laminin ]")),                       (ColumnIndex + 1)

        return self

    def FormatBaseGel(self: DRGExperimentalCondition) -> str:
        """
        FormatBaseGel

        This function...

        Return (str):
            ...
        """

        match self.BaseGel:
            case BaseGels.BaseGel_Ultimatrix:
                return BaseGels.BaseGel_Ultimatrix
            case BaseGels.BaseGel_GelMA:
                return BaseGels.BaseGel_GelMA
            case BaseGels.BaseGel_H6:
                return BaseGels.BaseGel_H6
            case BaseGels.BaseGel_H7:
                return BaseGels.BaseGel_H7
            case BaseGels.BaseGel_H8:
                return BaseGels.BaseGel_H8
            case BaseGels.BaseGel_PH15:
                return BaseGels.BaseGel_PH15
            case BaseGels.BaseGel_PH16:
                return BaseGels.BaseGel_PH16
            case BaseGels.BaseGel_PH18:
                return BaseGels.BaseGel_PH18
            case BaseGels.BaseGel_PH19:
                return BaseGels.BaseGel_PH19
            case _:
                return f"Unknown - {self.BaseGel}"

    def FormatGelMACrosslinker(self: DRGExperimentalCondition) -> str:
        """
        FormatGelMACrosslinker

        This function...

        Return (str):
            ...
        """

        match self.Crosslinker:
            case GelMACrosslinkers.GelMA_RuSPS:
                return GelMACrosslinkers.GelMA_RuSPS
            case GelMACrosslinkers.GelMA_Riboflavin:
                return GelMACrosslinkers.GelMA_Riboflavin
            case _:
                return f"Unknown - {self.Crosslinker}"

    def FormatNasrinsCrosslinkers(self: DRGExperimentalCondition) -> str:
        """
        FormatNasrinsCrosslinkers

        This function...

        Return (str):
            ...
        """

        match self.Polymer:
            case NasrinCrosslinker.NasrinGel_A:
                return NasrinCrosslinker.NasrinGel_A
            case _:
                return f"Unknown - {self.Polymer}"

    def SetFolderBase(self: DRGExperimentalCondition, Folder: str) -> DRGExperimentalCondition:
        """
        SetFolderBase

        This function sets the folder prefix for the LIF File member variable, to allow
        the spreadsheet to reference a directory structure without a root. This
        prefix is added to the LIF file path to provide an absolute path to the
        file when it needs to be processed or opened.

        Folder:
            The base folder, from the filesystem root, to where the directory tree of the
            LIF File column of the spreadsheet is referenced from.

        Return (self):
            The same instance of this class, with the LIF File path member variable updated
            to account for the provided base folder.
        """

        self.LIFFilePath = os.path.join(Folder, self.LIFFilePath)

        return self

    def Validate(self: DRGExperimentalCondition) -> bool:
        """
        Validate

        This function performs the necessary logical validation of the content
        and dependency tree of the values within this instance. This guarantees that the
        data as extracted from the spreadsheet to describe a single experimental trial
        is sensible and sufficiently complete to allow the necessary grouping and
        comparisons of the results.

        Return (bool):
            A boolean indicating whether or not the data within this instance is
            valid and should be analyzed.
        """

        IsValid: bool = True

        #   Check if the requested file exists
        if ( not os.path.exists(self.LIFFilePath) ):
            print(f"LIF file [ {self.LIFFilePath} ] does not exist or is not accessible!")
            IsValid = False

        #   ...

        return IsValid

    def Describe(self: DRGExperimentalCondition) -> str:
        """
        Describe

        This function provides a human-readable description of the entirety of
        the experimental data within this instance. This is solely for human
        consumption for verification and checking, and not to be used as
        authoritative records of the contents of an experimental trial.

        Return (str):
            A string containing a human-readable description of the experimental
            condition used to initialize this instance.
        """

        return "\n".join([
        ])


class DRGQuantificationResults():
    """
    DRGQuantificationResults

    This class...
    """

    ### Public Members
    SourceHash: str         #   A hash of the source file(s) used in generating these results

    ExperimentDate: str     #   YYYY-MM-DD date
    CultureDuration: int    #   How many days was the sample cultured for?
    SampleIndex: int        #   Which sample number within the chip is this?
    BaseGel: str            #   What gel was the sample cultured in?
    DilutionMedia: str      #   What media is used to dilute the gels created?
    IncludesPhenolRed: bool #   Does the media include Phenol Red?
    IncludesB27: bool       #   Does the media include B27?
    IncludesFetalBovineSerum: bool       #   Does the media include Fetal Bovine Serum?

    ##  ONLY APPLICABLE FOR GelMA
    GelMAPercentage: float                  #   Value on the range [0, 100]
    DegreeOfFunctionalization: float        #   Value on the range [0, 100]
    GelMACrosslinker: str                   #   Either RuSPS or Riboflavin
    RutheniumConcentration: float           #   Value in units of mM
    SodiumPersulfateConcentration: float    #   Value in units of mM
    RiboflavinConcentration: float          #   Value in units of ??
    GelIlluminationDuration: float          #   Seconds

    ##  ONLY APPLICABLE FOR NASRIN'S GELS
    CrosslinkingPolymer: str    #   How do these gels crosslink?
    Peptide: str                #   ???
    PeptideIn: str              #   ???
    PeptideConcentration: float #   ???

    ##  Additives
    IKVAV: bool                     #   Is the additive included?
    Gelatin: bool                   #   Is the additive included?
    Glutathione: bool               #   Is the additive included?
    GDNF: bool                      #   Is the additive included?
    BDNF: bool                      #   Is the additive included?
    Laminin: bool                   #   Is the additive included?
    IKVAVConcentration: float       #   Value in units of mM
    GelatinConcentration: float     #   Value in units of wt/%
    GlutathioneConcentration: float #   Value in units of mM
    GDNFConcentration: float        #   Value in units of ng/mL
    BDNFConcentration: float        #   Value in units of ng/mL
    LamininConcentration: float     #   Value in units of µg/mL

    ##  Results and Quantification Metrics
    DRGCentroidLocation: typing.List[int, int]              #   Where in the image is the centroid of the DRG Body? (X, Y) Pixel coordinates
    InclusionMaskFraction:  float                           #   What fraction of the image is included in the final inclusion mask, i.e. what fraction of the image can neurites grow within?
    NeuriteDistancesByLayer: typing.Dict[int, typing.List[int]]    #   Keys = Layer Index, Values = Count of Neurite Pixels at each integer distance from the centroid
    MedianNeuriteDistancesByLayer: typing.Dict[int, int]            #   Keys = LayerIndex, Values = Median Distance Neurites Grew To
    MedianNeuriteDistance: int  #   Median distance of all neurite pixels from the DRG centroid
    NeuriteDensityByLayer: typing.Dict[int, float]                 #   Keys = Layer Index, Values = Fraction of well interior occupied by neurite pixels
    NeuriteDensity: float   #   Ratio of neurite pixels to well interior pixels
    OrientationAngularResolution: float                     #   How many degrees are between each tested angle for the orientation results
    NeuriteOrientationsByLayer: typing.Dict[int, typing.List[int]] #   Keys = Layer Index, Values = Count of neurite pixels of each tested orientation
    OrientationMetricsByLayer: typing.Dict[int, typing.Tuple[int, float, float, float]] #   Keys = Layer Index, Values = (Count, AlignmentFraction, Mean, Stdev)
    OrientationMetrics: typing.Tuple[int, float, float, float]  #   (Count, AlignmentFraction, Mean, Stdev)
    #   ...

    ### Magic Methods
    def __init__(self: DRGQuantificationResults) -> None:
        """
        Constructor

        This function...

        Return (None):
            ...
        """

        self.SourceHash                     = ""

        self.ExperimentDate                 = "Unknown"
        self.CultureDuration                = -1
        self.SampleIndex                    = -1
        self.BaseGel                        = "Unknown"
        self.DilutionMedia                  = "Unknown"
        self.IncludesPhenolRed              = False
        self.IncludesB27                    = False
        self.IncludesFetalBovineSerum       = False

        self.GelMAPercentage                = -1.0
        self.DegreeOfFunctionalization      = -1.0
        self.GelMACrosslinker               = "N/A"
        self.RutheniumConcentration         = -1.0
        self.SodiumPersulfateConcentration  = -1.0
        self.RiboflavinConcentration        = -1.0
        self.GelIlluminationDuration        = -1.0

        self.CrosslinkingPolymer            = "N/A"
        self.Peptide                        = "N/A"
        self.PeptideIn                      = "N/A"
        self.PeptideConcentration           = "N/A"

        self.IKVAV                          = False
        self.Gelatin                        = False
        self.Glutathione                    = False
        self.GDNF                           = False
        self.BDNF                           = False
        self.Laminin                        = False

        self.IKVAVConcentration             = -1.0
        self.GelatinConcentration           = -1.0
        self.GlutathioneConcentration       = -1.0
        self.GDNFConcentration              = -1.0
        self.BDNFConcentration              = -1.0
        self.LamininConcentration           = -1.0

        self.DRGCentroidLocation            = [-1, -1]
        self.InclusionMaskFraction          = -1.0
        self.NeuriteDistancesByLayer        = {}
        self.MedianNeuriteDistancesByLayer  = {}
        self.MedianNeuriteDistance          = -1.0
        self.NeuriteDensityByLayer          = {}
        self.NeuriteDensity                 = -1.0
        self.OrientationAngularResolution   = -1.0
        self.NeuriteOrientationsByLayer     = {}
        self.OrientationMetricsByLayer      = {}
        self.OrientationMetrics             = ()

        return

    def __iter__(self: DRGQuantificationResults) -> typing.Iterable[typing.Any]:
        yield self.ExperimentDate
        yield self.CultureDuration
        yield self.BaseGel
        yield self.DilutionMedia
        yield self.IncludesPhenolRed
        yield self.IncludesB27
        yield self.IncludesFetalBovineSerum
        yield self.GelMAPercentage
        yield self.DegreeOfFunctionalization
        yield self.GelMACrosslinker
        yield self.RutheniumConcentration
        yield self.SodiumPersulfateConcentration
        yield self.RiboflavinConcentration
        yield self.GelIlluminationDuration
        yield self.CrosslinkingPolymer
        yield self.Peptide
        yield self.PeptideIn
        yield self.PeptideConcentration
        yield self.IKVAV
        yield self.Gelatin
        yield self.Glutathione
        yield self.GDNF
        yield self.BDNF
        yield self.Laminin
        yield self.IKVAVConcentration
        yield self.GelatinConcentration
        yield self.GlutathioneConcentration
        yield self.GDNFConcentration
        yield self.BDNFConcentration
        yield self.LamininConcentration

    def __eq__(self: DRGQuantificationResults, Other: DRGQuantificationResults) -> bool:

        return self.ExperimentDate              == Other.ExperimentDate                 and \
            self.CultureDuration                == Other.CultureDuration                and \
            self.BaseGel                        == Other.BaseGel                        and \
            self.DilutionMedia                  == Other.DilutionMedia                  and \
            self.IncludesPhenolRed              == Other.IncludesPhenolRed              and \
            self.IncludesB27                    == Other.IncludesB27                    and \
            self.IncludesFetalBovineSerum       == Other.IncludesFetalBovineSerum       and \
            self.GelMAPercentage                == Other.GelMAPercentage                and \
            self.DegreeOfFunctionalization      == Other.DegreeOfFunctionalization      and \
            self.GelMACrosslinker               == Other.GelMACrosslinker               and \
            self.RutheniumConcentration         == Other.RutheniumConcentration         and \
            self.SodiumPersulfateConcentration  == Other.SodiumPersulfateConcentration  and \
            self.RiboflavinConcentration        == Other.RiboflavinConcentration        and \
            self.GelIlluminationDuration        == Other.GelIlluminationDuration        and \
            self.CrosslinkingPolymer            == Other.CrosslinkingPolymer            and \
            self.Peptide                        == Other.Peptide                        and \
            self.PeptideIn                      == Other.PeptideIn                      and \
            self.PeptideConcentration           == Other.PeptideConcentration           and \
            self.IKVAV                          == Other.IKVAV                          and \
            self.Gelatin                        == Other.Gelatin                        and \
            self.Glutathione                    == Other.Glutathione                    and \
            self.GDNF                           == Other.GDNF                           and \
            self.BDNF                           == Other.BDNF                           and \
            self.Laminin                        == Other.Laminin                        and \
            self.IKVAVConcentration             == Other.IKVAVConcentration             and \
            self.GelatinConcentration           == Other.GelatinConcentration           and \
            self.GlutathioneConcentration       == Other.GlutathioneConcentration       and \
            self.GDNFConcentration              == Other.GDNFConcentration              and \
            self.BDNFConcentration              == Other.BDNFConcentration              and \
            self.LamininConcentration           == Other.LamininConcentration

    def __hash__(self: DRGQuantificationResults) -> int:
        return hash(tuple(self))

    ### Public Methods
    @staticmethod
    def GenerateRandom() -> DRGQuantificationResults:
        """
        GenerateRandom

        This function...

        return (DRGQuantificationResults):
            ...
        """

        Result: DRGQuantificationResults = DRGQuantificationResults()

        Result.SourceHash                     = random.randbytes(32).hex()

        Result.ExperimentDate                 = f"{random.choice([2024, 2025])}-{random.choice([1, 4, 7, 10])}-{random.choice([7, 14, 21, 28])}"
        Result.CultureDuration                = 7
        Result.SampleIndex                    = random.randint(1, 4)
        Result.BaseGel                        = random.choice([BaseGels.BaseGel_Ultimatrix, BaseGels.BaseGel_GelMA]) #, BaseGels.BaseGel_H6, BaseGels.BaseGel_H7, BaseGels.BaseGel_H8, BaseGels.BaseGel_PH15, BaseGels.BaseGel_PH16, BaseGels.BaseGel_PH18, BaseGels.BaseGel_PH19])
        Result.DilutionMedia                  = random.choice(["BaseMedia", "PBS"])
        Result.IncludesPhenolRed              = random.choice([True, False])
        Result.IncludesB27                    = random.choice([True, False])
        Result.IncludesFetalBovineSerum       = random.choice([True, False])

        if ( Result.BaseGel == BaseGels.BaseGel_GelMA ):
            Result.GelMAPercentage                = random.choice([3, 6])
            Result.DegreeOfFunctionalization      = random.choice([50, 80])
            Result.GelMACrosslinker               = random.choice([GelMACrosslinkers.GelMA_RuSPS, GelMACrosslinkers.GelMA_Riboflavin])
            if ( Result.GelMACrosslinker == GelMACrosslinkers.GelMA_RuSPS ):
                Result.RutheniumConcentration         = 0.2
                Result.SodiumPersulfateConcentration  = 0.02
            elif ( Result.GelMACrosslinker == GelMACrosslinkers.GelMA_Riboflavin ):
                Result.RiboflavinConcentration        = 1
            else:
                pass
            Result.GelIlluminationDuration        = 50

        Result.CrosslinkingPolymer            = "N/A"
        Result.Peptide                        = "N/A"
        Result.PeptideIn                      = "N/A"
        Result.PeptideConcentration           = "N/A"

        Result.IKVAV                          = random.choice([True, False])
        Result.GelatinIKVAV                   = random.choice([True, False])
        Result.GlutathioneIKVAV               = random.choice([True, False])
        Result.GDNF                           = random.choice([True, False])
        Result.BDNF                           = random.choice([True, False])
        Result.LamininIKVAV                   = random.choice([True, False])

        Result.IKVAVConcentration             = random.choice([1, 3]) if Result.IKVAV else 0
        Result.GelatinConcentration           = random.choice([1, 3]) if Result.Gelatin else 0
        Result.GlutathioneConcentration       = random.choice([1, 3]) if Result.Glutathione else 0
        Result.GDNFConcentration              = random.choice([1, 3]) if Result.GDNF else 0
        Result.BDNFConcentration              = random.choice([1, 3]) if Result.BDNF else 0
        Result.LamininConcentration           = random.choice([1, 3]) if Result.Laminin else 0

        Result.DRGCentroidLocation            = [-1, -1]
        Result.InclusionMaskFraction          = -1.0
        Result.NeuriteDistancesByLayer        = {}
        Result.MedianNeuriteDistancesByLayer  = {}
        Result.MedianNeuriteDistance          = random.random() * 1000
        Result.NeuriteDensityByLayer          = {}
        Result.NeuriteDensity                 = random.random()
        Result.OrientationAngularResolution   = -1.0
        Result.NeuriteOrientationsByLayer     = {}
        Result.OrientationMetricsByLayer      = {}
        Result.OrientationMetrics             = ()

        return Result

    def Copy(self: DRGQuantificationResults) -> DRGQuantificationResults:
        """
        Copy

        This function...

        Return (DRGQuantificationResults):
            ...
        """

        New: DRGQuantificationResults = DRGQuantificationResults()

        New.SourceHash = self.SourceHash

        New.ExperimentDate = self.ExperimentDate
        New.CultureDuration = self.CultureDuration
        New.SampleIndex = self.SampleIndex
        New.BaseGel = self.BaseGel
        New.DilutionMedia = self.DilutionMedia
        New.IncludesPhenolRed = self.IncludesPhenolRed
        New.IncludesB27 = self.IncludesB27
        New.IncludesFetalBovineSerum = self.IncludesFetalBovineSerum
        New.GelMAPercentage = self.GelMAPercentage
        New.DegreeOfFunctionalization = self.DegreeOfFunctionalization
        New.GelMACrosslinker = self.GelMACrosslinker
        New.RutheniumConcentration = self.RutheniumConcentration
        New.SodiumPersulfateConcentration = self.SodiumPersulfateConcentration
        New.RiboflavinConcentration = self.RiboflavinConcentration
        New.GelIlluminationDuration = self.GelIlluminationDuration
        New.CrosslinkingPolymer = self.CrosslinkingPolymer
        New.Peptide = self.Peptide
        New.PeptideIn = self.PeptideIn
        New.PeptideConcentration = self.PeptideConcentration
        New.IKVAV = self.IKVAV
        New.Gelatin = self.Gelatin
        New.Glutathione = self.Glutathione
        New.GDNF = self.GDNF
        New.BDNF = self.BDNF
        New.Laminin = self.Laminin
        New.IKVAVConcentration = self.IKVAVConcentration
        New.GelatinConcentration = self.GelatinConcentration
        New.GlutathioneConcentration = self.GlutathioneConcentration
        New.GDNFConcentration = self.GDNFConcentration
        New.BDNFConcentration = self.BDNFConcentration
        New.LamininConcentration = self.LamininConcentration
        New.DRGCentroidLocation = self.DRGCentroidLocation
        New.InclusionMaskFraction = self.InclusionMaskFraction
        New.NeuriteDistancesByLayer = self.NeuriteDistancesByLayer
        New.MedianNeuriteDistancesByLayer = self.MedianNeuriteDistancesByLayer
        New.MedianNeuriteDistance = self.MedianNeuriteDistance
        New.NeuriteDensityByLayer = self.NeuriteDensityByLayer
        New.NeuriteDensity = self.NeuriteDensity
        New.OrientationAngularResolution = self.OrientationAngularResolution
        New.NeuriteOrientationsByLayer = self.NeuriteOrientationsByLayer
        New.OrientationMetricsByLayer = self.OrientationMetricsByLayer
        New.OrientationMetrics = self.OrientationMetrics

        return New

    def Describe(self: DRGQuantificationResults, Verbose: bool = False) -> str:
        """
        Describe

        This function...

        Verbose:
            ...

        Return (str):
            ...
        """

        if ( not Verbose ):
            return "_".join(tuple(str(x) for x in tuple(self))).strip("-")

        return ", ".join([
            f"{self.ExperimentDate=}",
            f"{self.CultureDuration=}",
            f"{self.BaseGel=}",
            f"{self.DilutionMedia=}",
            f"{self.IncludesPhenolRed=}",
            f"{self.IncludesB27=}",
            f"{self.IncludesFetalBovineSerum=}",
            f"{self.GelMAPercentage=}",
            f"{self.DegreeOfFunctionalization=}",
            f"{self.GelMACrosslinker=}",
            f"{self.RutheniumConcentration=}",
            f"{self.SodiumPersulfateConcentration=}",
            f"{self.RiboflavinConcentration=}",
            f"{self.GelIlluminationDuration=}",
            f"{self.CrosslinkingPolymer=}",
            f"{self.Peptide=}",
            f"{self.PeptideIn=}",
            f"{self.PeptideConcentration=}",
            f"{self.IKVAV=}",
            f"{self.Gelatin=}",
            f"{self.Glutathione=}",
            f"{self.GDNF=}",
            f"{self.BDNF=}",
            f"{self.Laminin=}",
            f"{self.IKVAVConcentration=}",
            f"{self.GelatinConcentration=}",
            f"{self.GlutathioneConcentration=}",
            f"{self.GDNFConcentration=}",
            f"{self.BDNFConcentration=}",
            f"{self.LamininConcentration=}",
        ]).replace("self.", "")

    def Equivalent(self: DRGQuantificationResults, Other: DRGQuantificationResults, Template: DRGQuantificationResults) -> bool:
        """
        Equivalent

        This function...

        Other:
            ...
        Template:
            ...

        Return (bool):
            ...
        """

        for A, B, T in zip(tuple(self), tuple(Other), tuple(Template)):
            if ( T is not None ) and ( A != B ):
                return False

        return True

    def CountVariations(self: DRGQuantificationResults, Other: DRGQuantificationResults) -> int:
        """
        CountVariations

        This function...

        Other:
            ...

        Return (int):
            ...
        """

        Count: int = 0

        for This, That in zip(tuple(self), tuple(Other)):
            if ( This != That ):
                Count += 1

        return Count

    def DescribeVariations(self: DRGQuantificationResults, Other: DRGQuantificationResults) -> str:
        """
        DescribeVariations

        This function...

        Other:
            ...

        Return (str):
            ...
        """

        This, That = self.Describe(Verbose=True), Other.Describe(Verbose=True)

        Description: str = ", ".join([
            f"{x} vs. {y}" for x, y in zip(This.split(", "), That.split(", ")) if x != y
        ])

        return Description.strip(", ")

    def ExtractExperimentalDetails(self: DRGQuantificationResults, ExperimentDetails: DRGExperimentalCondition) -> DRGQuantificationResults:
        """
        ExtractExperimentalDetails

        This function...

        ExperimentDetails:
            ...

        Return (self):
            ...
        """

        self.ExperimentDate = ExperimentDetails.ExperimentDate.strftime(f"%Y-%m-%d")
        self.CultureDuration = ExperimentDetails.CultureDuration
        self.SampleIndex = ExperimentDetails.SampleIndex
        self.BaseGel = ExperimentDetails.FormatBaseGel()
        self.DilutionMedia = ExperimentDetails.DilutionMedia
        self.IncludesPhenolRed = ExperimentDetails.IncludesPhenolRed
        self.IncludesB27 = ExperimentDetails.B27Inclusion
        self.IncludesFetalBovineSerum = ExperimentDetails.FBSInclusion

        if ( self.BaseGel == BaseGels.BaseGel_GelMA ):
            self.GelMAPercentage = ExperimentDetails.GelMAPercentage
            self.DegreeOfFunctionalization = ExperimentDetails.DegreeOfFunctionalization
            self.GelMACrosslinker = ExperimentDetails.FormatGelMACrosslinker()
            self.RutheniumConcentration = ExperimentDetails.RutheniumConcentration
            self.SodiumPersulfateConcentration = ExperimentDetails.SodiumPersulfateConcentration
            self.RiboflavinConcentration = ExperimentDetails.RiboflavinConcentration
            self.GelIlluminationDuration = ExperimentDetails.GelIlluminationTime
        elif ( self.BaseGel == BaseGels.BaseGel_Ultimatrix ):
            #   TODO: ...
            pass
        else:
            self.CrosslinkingPolymer = ExperimentDetails.FormatNasrinsCrosslinkers()
            self.Peptide = ExperimentDetails.Peptide
            self.PeptideIn = ExperimentDetails.PeptideIn
            self.PeptideConcentration = ExperimentDetails.PeptideConcentration

        self.IKVAV = ExperimentDetails.IKVAV
        self.Gelatin = ExperimentDetails.Gelatin
        self.Glutathione = ExperimentDetails.Glutathione
        self.GDNF = ExperimentDetails.GDNF
        self.BDNF = ExperimentDetails.BDNF
        self.Laminin = ExperimentDetails.Laminin

        self.IKVAVConcentration = ExperimentDetails.IKVAVConcentration
        self.GelatinConcentration = ExperimentDetails.GelatinConcentration
        self.GlutathioneConcentration = ExperimentDetails.GlutathioneConcentration
        self.GDNFConcentration = ExperimentDetails.GDNFConcentration
        self.BDNFConcentration = ExperimentDetails.BDNFConcentration
        self.LamininConcentration = ExperimentDetails.LamininConcentration

        return self

    def Save(self: DRGQuantificationResults, Folder: str, DryRun: bool) -> bool:
        """
        Save

        This function...

        Folder:
            ...
        DryRun:
            ...

        Return (bool):
            ...
        """

        Success: bool = True

        Stringified: str = jsonpickle.encode(
            self,
            unpicklable=False,
            make_refs=False,
            keys=True,
            indent=4,
        )

        if ( not DryRun ):
            if ( not os.path.exists(Folder) ):
                os.makedirs(Folder, mode=0o755, exist_ok=True)

            with open(os.path.join(Folder, f"{self.SourceHash} - Results.json"), mode="+w") as OutFile:
                OutFile.write(Stringified)
        else:
            print(Stringified)

        return Success

    @staticmethod
    def FromJSON(JSONData: str) -> DRGQuantificationResults:
        """
        FromJSON

        This function...

        JSONData:
            ...

        Return (self):
            ...
        """

        return jsonpickle.decode(JSONData, keys=True, classes=DRGQuantificationResults, on_missing='ignore')

    @staticmethod
    def FromJSONFile(Filename: str) -> DRGQuantificationResults:
        """
        FromJSONFile

        This function...

        Filename:
            ...

        Return (self):
            ...
        """

        Contents: str = ""
        with open(Filename, "r") as InFile:
            Contents = InFile.read()

        return DRGQuantificationResults.FromJSON(Contents)


class DRGQuantificationResultsSet():
    """
    DRGQuantificationResultsSet

    This class...
    """

    _Results: typing.List[DRGQuantificationResults]
    _LogWriter: Logger.Logger

    def __init__(self: DRGQuantificationResultsSet, Results: typing.Sequence[DRGQuantificationResults] = list(), LogWriter: Logger.Logger = Logger.Discarder) -> None:
        """
        Constructor...

        This function...

        Results:
            ...

        Return (None):
            ...
        """

        self._Results = list(Results)
        self._LogWriter = LogWriter

        return

    def __iter__(self: DRGQuantificationResultsSet) -> typing.Iterator[DRGQuantificationResults]:
        return iter(self._Results)

    def __next__(self: DRGQuantificationResultsSet) -> typing.Generator[None, DRGQuantificationResults, None]:

        for Result in self._Results:
            yield Result
        else:
            raise StopIteration

    def __len__(self: DRGQuantificationResultsSet) -> int:
        return len(self._Results)

    @staticmethod
    def FromDirectory(Directory: str) -> DRGQuantificationResultsSet:
        """
        FromDirectory

        This function...

        Directory:
            ...

        Return (DRGQuantificationResultsSet):
            ...
        """

        Results: typing.List[DRGQuantificationResults] = []
        for File in glob.glob(f"{Directory}/*.json"):
            Results.append(DRGQuantificationResults.FromJSONFile(File))

        return DRGQuantificationResultsSet(Results)

    def SetLogger(self: DRGQuantificationResultsSet, LogWriter: Logger.Logger) -> DRGQuantificationResultsSet:
        """
        SetLogger

        This function...

        LogWriter:
            ...

        return (self):
            ...
        """

        self._LogWriter = LogWriter

        return self

    def Add(self: DRGQuantificationResultsSet, ToAdd: DRGQuantificationResults) -> DRGQuantificationResultsSet:
        """
        Add

        This function...

        ToAdd:
            ...

        Return (self):
            ...
        """

        self._Results.append(ToAdd)

        return self

    def Split(self: DRGQuantificationResultsSet) -> typing.Sequence[DRGQuantificationResultsSet]:
        """
        Split

        This function...

        Return (Sequence[DRGQuantificationResultsSet]):
            ...
        """

        Groups: typing.Dict[int, DRGQuantificationResultsSet] = {}

        for Index, Result in enumerate(self):
            if ( hash(Result) not in Groups.keys() ):
                Groups[hash(Result)] = DRGQuantificationResultsSet([Result], self._LogWriter)
                self._LogWriter.Println(f"Created group [ {len(Groups)} ] ({Index}/{len(self._Results)})")
            else:
                Groups[hash(Result)].Add(Result)

        self._LogWriter.Println(f"Split results into {len(Groups)} unique group(s).")
        return list(Groups.values())

    def GroupBy(self: DRGQuantificationResultsSet, Template: DRGQuantificationResults) -> typing.Sequence[DRGQuantificationResultsSet]:
        """
        GroupBy

        This function...

        Template:
            ...

        Return (Sequence[DRGQuantificationResultsSet]):
            ...
        """

        Groups: typing.Sequence[DRGQuantificationResultsSet] = []

        for Index, Result in enumerate(self):
            for Group in Groups:
                if ( Group._Results[0].Equivalent(Result, Template) ):
                    Group.Add(Result)
                    break
            else:
                Groups.append(DRGQuantificationResultsSet([Result], self._LogWriter))
                self._LogWriter.Println(f"Created group [ {len(Groups)} ] ({Index}/{len(self._Results)})")

        self._LogWriter.Println(f"Split into a total of {len(Groups)} group(s).")
        return Groups

    def Filter(self: DRGQuantificationResultsSet, FilterFunc: typing.Callable[[DRGQuantificationResults], bool]) -> DRGQuantificationResultsSet:
        """
        Filter

        This function...

        FilterFunc:
            ...

        Return (DRGQuantificationResultsSet):
            ...
        """

        Filtered: DRGQuantificationResultsSet = DRGQuantificationResultsSet([], self._LogWriter)
        for Result in self:
            if ( FilterFunc(Result) ):
                Filtered.Add(Result)

        self._LogWriter.Println(f"Provided filter function accepted a total of [ {len(Filtered)} ] results.")
        return Filtered

    def Unique(self: DRGQuantificationResultsSet, GetParameter: typing.Callable[[DRGQuantificationResults], typing.Any]) -> typing.Sequence[typing.Any]:
        """
        Unique

        This function...

        Return (Sequence[Any]):
            ...
        """

        Values: typing.Set[typing.Any] = set()

        for Result in self:
            Values.add(GetParameter(Result))

        self._LogWriter.Println(f"Provided uniqueness function identified a total of [ {len(Values)} ].")
        return list(sorted(Values))

    def Summarize(self: DRGQuantificationResultsSet, OutputDirectory: str) -> None:
        """
        Summarize

        This function provides a single entry-point for summarizing an entire
        set of results from the DRG Neurite Quantification analysis. This is
        called on an instance of the DRGQuantificationResultsSet class which has
        been initialized on the full set of results to be included.

        This function must be implemented by calling dedicated private methods
        to perform the actual grouping, filtering, and plotting of the results
        contained. This function must only orchestrate this lower-level
        plotting, and not perform any data manipulations here to ensure that
        specific summarization steps can be extracted and run independently if
        necessary.

        OutputDirectory:
            The full path to the folder into which the summarized results should
            be written out to.

        Return (None):
            This function returns nothing to the caller, the summarized results are generated
            and written out to the specified folder.
        """

        if ( OutputDirectory is None ) or ( OutputDirectory == "" ):
            raise ValueError(f"OutputDirectory must be provided and non-empty.")

        #   One figure we want to generate is a scatter-plot showing how the median lengths vary
        #   across all of the experimental conditions
        self._MedianNeuriteLengthOverConditions(os.path.join(OutputDirectory, "Median Neurite Length versus Neurite Density"))

        #   We also want to explore the differences between the 3% and 6% GelMA,
        #   for each of the 50 and 80 DOF formulations.
        self._GelMAPercentageAndDOF(os.path.join(OutputDirectory, f"Neurite Length by GelMA Percentage and DOF"), CollapseDates=True)
        self._GelMAPercentageAndDOF(os.path.join(OutputDirectory, f"Neurite Length by GelMA Percentage and DOF By Date"))

        #   ...
        self._GelMAPercentageAndDOFByDilutionMedia(os.path.join(OutputDirectory, f"Neurite Length by GelMA Percentage, DOF, and Dilution Medium"), CollapseDates=True)
        self._GelMAPercentageAndDOFByDilutionMedia(os.path.join(OutputDirectory, f"Neurite Length by GelMA Percentage, DOF, and Dilution Medium By Date"))

        #   ...

        return

    ### Private Methods
    def _MedianNeuriteLengthOverConditions(self: DRGQuantificationResultsSet, OutputDirectory: str) -> None:
        """
        _MedianNeuriteLengthOverConditions

        This function...

        OutputDirectory:
            ...

        Return (None):
            ...
        """

        self._LogWriter.Println(f"Preparing scatterplot of neurite lengths versus neurite density across the experimental conditions...")

        #   Split up the results on every permutation of the experimental conditions.
        Groups: typing.Sequence[DRGQuantificationResultsSet] = self.Split()

        #   Prepare a file for helping to map the markers and colours back to specific experimental conditions
        if ( not os.path.exists(OutputDirectory) ):
            self._LogWriter.Println(f"Creating output directory [ {OutputDirectory} ]...")
            os.makedirs(OutputDirectory, mode=0o755, exist_ok=True)

        with open(os.path.join(OutputDirectory, "Scatterplot Marker Mapping.csv"), "w+") as MapFile:
            self._LogWriter.Println(f"Preparing file [ {MapFile.name} ] to describe plot markers")

            F: Figure = Utils.PrepareFigure()
            Ax: Axes = F.add_subplot(111)

            F.suptitle(f"Median Neurite Outgrowth Length by Experimental Condition")
            Ax.set_title(f"Median Neurite Length versus Neurite Density")
            Ax.set_xlabel(f"Neurite Density (n.d.)")
            Ax.set_ylabel(f"Median Neurite Length (µm)")

            Shapes: typing.Sequence[str] = ["o", "v", "^", "<", ">", "1", "2", "3", "4", "s", "*", "x"]
            Colours: typing.Sequence[typing.Any] = matplotlib.cm.rainbow(np.linspace(0, 1, math.ceil(len(Groups) / len(Shapes))))

            for GroupIndex, (Group, (Shape, Colour)) in enumerate(zip(Groups, itertools.product(Shapes, Colours)), start=1):
                Ax.scatter([x.NeuriteDensity for x in Group._Results], [x.MedianNeuriteDistance for x in Group._Results], color=Colour, marker=Shape)
                MapFile.write(f"Group={GroupIndex}, Count={len(Group._Results)}, Marker={Shape}, Colour={Colour}, {Group._Results[0].Describe(Verbose=True)}\n")

        PlotFilename: str = "Median Neurite Length versus Neurite Density.png"
        Utils.WriteImage(Utils.FigureToImage(F), os.path.join(OutputDirectory, PlotFilename))
        self._LogWriter.Println(f"Writing figure to file [ {PlotFilename} ]...")

        self._LogWriter.Println(f"Finished scatterplot of neurite lengths versus neurite density across the experimental conditions.")

        return

    def _GelMAPercentageAndDOF(self: DRGQuantificationResultsSet, OutputDirectory: str, CollapseDates: bool = False) -> None:
        """
        _GelMAPercentageAndDOF

        This function...

        OutputDirectory:
            ...

        Return (None):
            ...
        """

        self._LogWriter.Println(f"Preparing boxplots of neurite length as a function of GelMA percentage and degree of functionalization...")

        if ( not os.path.exists(OutputDirectory) ):
            os.makedirs(OutputDirectory, mode=0o755, exist_ok=True)
            self._LogWriter.Println(f"Creating output directory [ {OutputDirectory } ]...")

        GelMAResults: DRGQuantificationResultsSet = DRGQuantificationResultsSet([x for x in self._Results if x.BaseGel == BaseGels.BaseGel_GelMA], LogWriter=self._LogWriter)
        if ( len(GelMAResults) == 0 ):
            self._LogWriter.Println(f"No results were found where BaseGel=GelMA...")
            return

        #   Identify the possible values for the GelMA percentage and the Degree of Functionalization of the gel.
        GelMAPercentages: typing.Sequence[float] = GelMAResults.Unique(lambda x: x.GelMAPercentage)
        DegreeOfFunctionalizations: typing.Sequence[float] = GelMAResults.Unique(lambda x: x.DegreeOfFunctionalization)

        self._LogWriter.Println(f"Found results for GelMA percentages: [ {GelMAPercentages} ]...")
        self._LogWriter.Println(f"Found results for degree of functionalizaton: [ {DegreeOfFunctionalizations} ]...")

        #   We need to generate groups which are unique in all parameters *Except* the GelMA percentage and Degree of Functionalization.
        #   Then, we can split on these last two parameters and get meaningful comparisons across these two experimental variables for
        #   every other larger set of experimental variables.
        Template: DRGQuantificationResults = GelMAResults._Results[0].Copy()
        if ( CollapseDates ):
            Template.ExperimentDate = None
        Template.GelMAPercentage = None
        Template.DegreeOfFunctionalization = None
        Groups: typing.Sequence[DRGQuantificationResultsSet] = GelMAResults.GroupBy(Template)

        #   For each set of experimental conditions, identify the cases we care about for these figures:
        for GroupIndex, Group in enumerate(Groups, start=1):
            self._LogWriter.Println(f"Preparing boxplot for condition [ {GroupIndex}/{len(Groups)} ]...")
            F: Figure = Utils.PrepareFigure()
            Ax: Axes = F.add_subplot(111)
            Example: DRGQuantificationResults = Group._Results[0]
            AxisTitle: str = "".join([
                f"{Example.ExperimentDate if not CollapseDates else ''}",
                f", {Example.DilutionMedia}",
                f', Phenol Red' if Example.IncludesPhenolRed else '',
                f', B27' if Example.IncludesB27 else '',
                f', FBS' if Example.IncludesFetalBovineSerum else '',
                f', Ru-SPS {Example.RutheniumConcentration}, {Example.SodiumPersulfateConcentration}' if Example.GelMACrosslinker == GelMACrosslinkers.GelMA_RuSPS else f', Riboflavin {Example.RiboflavinConcentration}',
                f', IKVAV {Example.IKVAVConcentration}' if Example.IKVAV else '',
                f', Gelatin {Example.GelatinConcentration}' if Example.Gelatin else '',
                f', Glutathione {Example.GlutathioneConcentration}' if Example.GlutathioneConcentration else '',
                f', GDNF {Example.GDNFConcentration}' if Example.GDNF else '',
                f', BDNF {Example.BDNFConcentration}' if Example.BDNF else '',
                f', Laminin {Example.LamininConcentration}' if Example.Laminin else '',
            ]).strip(", ")

            with open(os.path.join(OutputDirectory, f"{AxisTitle}.csv"), "+w") as DataFile:
                for Index, (GelMAPercentage, DegreeOfFunctionalization) in enumerate(itertools.product(GelMAPercentages, DegreeOfFunctionalizations)):

                    Condition: DRGQuantificationResultsSet = DRGQuantificationResultsSet([
                        x for x in Group if \
                            x.GelMAPercentage == GelMAPercentage and \
                            x.DegreeOfFunctionalization == DegreeOfFunctionalization
                        ])
                    Distances: typing.List[float] = [x.MedianNeuriteDistance for x in Condition]
                    Ax.boxplot(Distances, sym='', positions=[Index], labels=[f"{GelMAPercentage}% GelMA\n{DegreeOfFunctionalization} DOF\nn={len(Condition)}"])

                    Ax.scatter(np.random.normal(Index, 0.04, len(Distances)), Distances, c='k', alpha=0.5)

                    DataFile.write(f"{GelMAPercentage}% GelMA - {DegreeOfFunctionalization} DOF")
                    DataFile.write(''.join([f",{x}" for x in Distances]))
                    DataFile.write("\n")

            F.suptitle(f"Median DRG Neurite Length versus GelMA Concentration and Degree of Functionalization")
            Ax.set_title(AxisTitle)
            Ax.minorticks_on()
            Ax.set_ylim(bottom=0.0)
            Ax.set_ylabel(f"Median Neurite Length (µm)")
            Ax.set_xlabel(f"GelMA Percentage & Degree of Functionalization")
            self._LogWriter.Println(f"Created boxplot for condition [ {GroupIndex}/{len(Groups)} ].")

            # Utils.DisplayFigure(AxisTitle, F, 1, True, True)

            Utils.WriteImage(Utils.FigureToImage(F), os.path.join(OutputDirectory, f"{AxisTitle}.png"))
            self._LogWriter.Println(f"Saved figure to file [ {AxisTitle}.png ]...")
            F.clear()

        self._LogWriter.Println(f"Finished creating boxplots of neurite length as a function of GelMA percentage and degree of functionalization.")

        return

    def _GelMAPercentageAndDOFByDilutionMedia(self: DRGQuantificationResultsSet, OutputDirectory: str, CollapseDates: bool = False) -> None:
        """
        _GelMAPercentageAndDOFByDilutionMedia

        This function...

        OutputDirectory:
            ...
        CollapseDates:
            ...

        Return (None):
            ...
        """

        self._LogWriter.Println(f"Preparing boxplots of neurite length as a function of GelMA percentage, degree of functionalization, and dilution media...")

        if ( not os.path.exists(OutputDirectory) ):
            os.makedirs(OutputDirectory, mode=0o755, exist_ok=True)
            self._LogWriter.Println(f"Creating output directory [ {OutputDirectory } ]...")

        GelMAResults: DRGQuantificationResultsSet = DRGQuantificationResultsSet([x for x in self._Results if x.BaseGel == BaseGels.BaseGel_GelMA], LogWriter=self._LogWriter)
        if ( len(GelMAResults) == 0 ):
            self._LogWriter.Println(f"No results were found where BaseGel=GelMA...")
            return

        #   Identify the possible values for the GelMA percentage and the Degree of Functionalization of the gel.
        GelMAPercentages: typing.Sequence[float] = GelMAResults.Unique(lambda x: x.GelMAPercentage)
        DegreeOfFunctionalizations: typing.Sequence[float] = GelMAResults.Unique(lambda x: x.DegreeOfFunctionalization)
        DilutionMedia: typing.Sequence[str] = GelMAResults.Unique(lambda x: x.DilutionMedia)

        self._LogWriter.Println(f"Found results for GelMA percentages: [ {GelMAPercentages} ]...")
        self._LogWriter.Println(f"Found results for degree of functionalizaton: [ {DegreeOfFunctionalizations} ]...")
        self._LogWriter.Println(f"Found results for dilution media: [ {DilutionMedia} ]...")

        #   We need to generate groups which are unique in all parameters *Except* the GelMA percentage and Degree of Functionalization.
        #   Then, we can split on these last two parameters and get meaningful comparisons across these two experimental variables for
        #   every other larger set of experimental variables.
        Template: DRGQuantificationResults = GelMAResults._Results[0].Copy()
        if ( CollapseDates ):
            Template.ExperimentDate = None
        Template.GelMAPercentage = None
        Template.DegreeOfFunctionalization = None
        Template.DilutionMedia = None
        Groups: typing.Sequence[DRGQuantificationResultsSet] = GelMAResults.GroupBy(Template)

        #   For each set of experimental conditions, identify the 4 cases we care about for these figures:
        for GroupIndex, Group in enumerate(Groups, start=1):
            self._LogWriter.Println(f"Preparing boxplot for condition [ {GroupIndex}/{len(Groups)} ]...")
            F: Figure = Utils.PrepareFigure()
            Ax: Axes = F.add_subplot(111)
            Example: DRGQuantificationResults = Group._Results[0]
            AxisTitle: str = "".join([
                f"{Example.ExperimentDate if not CollapseDates else ''}",
                f', Phenol Red' if Example.IncludesPhenolRed else '',
                f', B27' if Example.IncludesB27 else '',
                f', FBS' if Example.IncludesFetalBovineSerum else '',
                f', Ru-SPS {Example.RutheniumConcentration}, {Example.SodiumPersulfateConcentration}' if Example.GelMACrosslinker == GelMACrosslinkers.GelMA_RuSPS else f', Riboflavin {Example.RiboflavinConcentration}',
                f', IKVAV {Example.IKVAVConcentration}' if Example.IKVAV else '',
                f', Gelatin {Example.GelatinConcentration}' if Example.Gelatin else '',
                f', Glutathione {Example.GlutathioneConcentration}' if Example.GlutathioneConcentration else '',
                f', GDNF {Example.GDNFConcentration}' if Example.GDNF else '',
                f', BDNF {Example.BDNFConcentration}' if Example.BDNF else '',
                f', Laminin {Example.LamininConcentration}' if Example.Laminin else '',
            ]).strip(", ")

            with open(os.path.join(OutputDirectory, f"{AxisTitle}.csv"), "+w") as DataFile:
                for Index, (DilutionMedium, GelMAPercentage, DegreeOfFunctionalization) in enumerate(itertools.product(DilutionMedia, GelMAPercentages, DegreeOfFunctionalizations)):

                    Condition: DRGQuantificationResultsSet = DRGQuantificationResultsSet([
                        x for x in Group if \
                            x.GelMAPercentage == GelMAPercentage and \
                            x.DegreeOfFunctionalization == DegreeOfFunctionalization and \
                            x.DilutionMedia == DilutionMedium
                        ])
                    Distances: typing.List[float] = [x.MedianNeuriteDistance for x in Condition]
                    Ax.boxplot(Distances, sym='', positions=[Index], labels=[f"{DilutionMedium}\n{GelMAPercentage}% GelMA\n{DegreeOfFunctionalization} DOF\nn={len(Condition)}"])

                    Ax.scatter(np.random.normal(Index, 0.04, len(Distances)), Distances, c='k', alpha=0.5)

                    DataFile.write(f"{DilutionMedium} - {GelMAPercentage}% GelMA - {DegreeOfFunctionalization} DOF")
                    DataFile.write(''.join([f",{x}" for x in Distances]))
                    DataFile.write("\n")

            F.suptitle(f"Median DRG Neurite Length versus GelMA Concentration, Degree of Functionalization, and Dilution Medium")
            Ax.set_title(AxisTitle)
            Ax.minorticks_on()
            Ax.set_ylim(bottom=0.0)
            Ax.set_ylabel(f"Median Neurite Length (µm)")
            Ax.set_xlabel(f"Dilution Medium, GelMA Percentage, Degree of Functionalization")
            self._LogWriter.Println(f"Created boxplot for condition [ {GroupIndex}/{len(Groups)} ].")

            # Utils.DisplayFigure(AxisTitle, F, 1, True, True)

            Utils.WriteImage(Utils.FigureToImage(F), os.path.join(OutputDirectory, f"{AxisTitle}.png"))
            self._LogWriter.Println(f"Saved figure to file [ {AxisTitle}.png ]...")
            F.clear()

        self._LogWriter.Println(f"Finished creating boxplots of neurite length as a function of GelMA percentage, degree of functionalization, and dilution medium.")

        return
