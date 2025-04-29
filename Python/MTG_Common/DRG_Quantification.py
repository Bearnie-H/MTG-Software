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
import jsonpickle
import os
import str2bool

from MTG_Common import Utils
#   ...

#   Import the necessary third-part modules
#   ...

#   Import the desired locally written modules
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

class DRGAnalysis_StatusCode(int):
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

class BaseGels(str):
    BaseGel_Ultimatrix: str = "Ultimatrix"

    BaseGel_GelMA: str = "GelMA"

    BaseGel_H6: str = "?H6?"
    BaseGel_H7: str = "?H7?"
    BaseGel_H8: str = "?H8?"

    BaseGel_PH15: str = "?PH15?"
    BaseGel_PH16: str = "?PH16?"
    BaseGel_PH18: str = "?PH18?"
    BaseGel_PH19: str = "?PH19?"

class GelMACrosslinkers(str):
    GelMA_RuSPS: str = "RuSPS"
    GelMA_Riboflavin: str = "Riboflavin"

class NasrinCrosslinker(str):
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
    IncludesFBS: bool       #   Does the media include Fetal Bovine Serum?

    ##  ONLY APPLICABLE FOR GelMA
    GelMAPercentage: float                  #   Value on the range [0, 1]
    DegreeOfFunctionalization: float        #   Value on the range [0, 1]
    GelMACrosslinker: str                   #   Either RuSPS or Riboflavin
    RutheniumConcentration: float           #   Value in units of mM
    SodiumPersulfateConcentration: float    #   Value in units of mM
    RiboflavinConcentration: float          #   Value in units of mM
    GelIlluminationDuration: float          #   Seconds

    ##  ONLY APPLICABLE FOR NASRIN'S GELS
    CrosslinkingPolymer: str    #   How do these gels crosslink?
    Peptide: str                #   ???
    PeptideIn: str              #   ???
    PeptideConcentration: float #   ???

    ##  Additives
    IKVAVConcentration: float       #   Value in units of mM
    GelatinConcentration: float     #   Value in units of mM
    GlutathioneConcentration: float #   Value in units of mM
    GDNFConcentration: float        #   Value in units of mM
    BDNFConcentration: float        #   Value in units of mM
    LamininConcentration: float     #   Value in units of mM

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
        self.IncludesFBS                    = False

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

    ### Public Methods
    def ExtractExperimentalDetails(self: DRGQuantificationResults, ExperimentDetails: DRGExperimentalCondition) -> DRGQuantificationResults:
        """
        ExtractExperimentalDetails

        This function...

        ExperimentDetails:
            ...

        Return (self):
            ...
        """

        self.SourceHash = Utils.Sha256Sum(ExperimentDetails.LIFFilePath)

        self.ExperimentDate = ExperimentDetails.ExperimentDate.strftime(f"%Y-%m-%d")
        self.CultureDuration = ExperimentDetails.CultureDuration
        self.SampleIndex = ExperimentDetails.SampleIndex
        self.BaseGel = ExperimentDetails.FormatBaseGel()
        self.DilutionMedia = ExperimentDetails.DilutionMedia
        self.IncludesPhenolRed = ExperimentDetails.IncludesPhenolRed
        self.IncludesB27 = ExperimentDetails.B27Inclusion
        self.IncludesFBS = ExperimentDetails.FBSInclusion

        if ( self.BaseGel == BaseGels.BaseGel_GelMA ):
            self.GelMAPercentage = ExperimentDetails.GelMAPercentage / 100.0
            self.DegreeOfFunctionalization = ExperimentDetails.DegreeOfFunctionalization / 100.0
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

    def __init__(self: DRGQuantificationResultsSet, Results: typing.Sequence[DRGQuantificationResults] = []) -> None:
        """
        Constructor...

        This function...

        Results:
            ...

        Return (None):
            ...
        """

        self._Results = list(Results)

        return

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

    def Filter(self: DRGQuantificationResultsSet, FilterFunc: typing.Callable[[DRGQuantificationResults, typing.List[typing.Any]], bool], FilterArgs: typing.List[typing.Any]) -> typing.Tuple[DRGQuantificationResultsSet, DRGQuantificationResultsSet]:
        """
        Filter

        This function...

        FilterFunc:
            ...
        FilterArgs:
            ...

        Return (Tuple):
            [0] - DRGQuantificationResultsSet:
                ...
            [1] - DRGQuantificationResultsSet:
                ...
        """

        Included: typing.List[DRGQuantificationResults] = []
        Excluded: typing.List[DRGQuantificationResults] = []

        for Result in self._Results:
            if ( FilterFunc(Result, FilterArgs) ):
                Included.append(Result)
            else:
                Excluded.append(Result)

        return (DRGQuantificationResultsSet(Included), DRGQuantificationResultsSet(Excluded))

    def UniqueByField(self: DRGQuantificationResultsSet, FieldName: str) -> typing.Tuple[int, typing.Sequence[DRGQuantificationResultsSet]]:
        """
        UniqueByField

        This function...

        FieldName:
            ...

        Return (Tuple):
            [0] - int:
                ...
            [1] - Sequence[DRGQuantificationResultsSet]:
                ...
        """

        if ( len(self._Results) == 0 ):
            return (0, [])

        if ( FieldName not in self._Results[0].__dict__.keys() ):
            raise ValueError(f"Field [ {FieldName} ] is not a valid field name!")

        UniqueValues: typing.Set[typing.Any] = {}
        for Result in self._Results:
            UniqueValues.add(Result.__getattribute__(FieldName))

        Split: typing.Dict[str, DRGQuantificationResultsSet] = {str(x): DRGQuantificationResultsSet() for x in UniqueValues}

        for Result in self._Results:
            Split[str(Result.__getattribute__(FieldName))].Add(Result)

        return (len(UniqueValues), [Split[x] for x in sorted(Split.keys())])
