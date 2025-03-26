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
import str2bool
import os
#   ...

#   Import the necessary third-part modules
#   ...

#   Import the desired locally written modules
#   ...

#   Helper parsing functions to wrap and error handling in a consistent manner
def NormalizePathSeparators(Path: str, Separator: str = "/") -> str:
    """
    NormalizePathSeparators

    This function...

    Path:
        ...
    Separator:
        ...

    Return (str):
        ...
    """

    if ( Separator is None ) or ( Separator == "" ):
        Separator = os.sep

    TemporaryPath: typing.Sequence[str] = '\\'.join([k for k in Path.split('/') if len(k) > 0])
    FinalPath: str = Separator.join([k for k in TemporaryPath.split('\\') if len(k) > 0])

    return FinalPath

def TryParseDatetime(Input: str, Format: str, ErrorMessage: str = "") -> datetime | None:
    """
    TryParseDatetime

    This function...

    Input:
        ...
    Format:
        ...

    Return (datetime | None):
        ...
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

    This function...

    Input:
        ...

    Return (int | None):
        ...
    """

    if ( Input is None ) or ( Input == "" ):
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

    This function...

    Input:
        ...

    Return (float | None):
        ...
    """

    if ( Input is None ) or ( Input == "" ):
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

    This function...

    Input:
        ...

    Return (bool | None):
        ...
    """

    if ( Input is None ) or ( Input == "" ):
        Input = "False"

    try:
        return str2bool.str2bool(Input.lower(), True)
    except Exception as e:
        if ( ErrorMessage is not None ) and ( ErrorMessage != "" ):
            print(f"Boolean Parse Error: {ErrorMessage} - {e}")
        return None

class DRGExperimentalCondition():
    """
    DRGExperimentalCondition

    This class...
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

    Polymer: str    #   ???

    Crosslinker: str    #   RuSPS or Riboflavin

    Peptide: str
    PeptideIn: str
    PeptideConcentration: float

    DilutionMedia: str

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


    ### Magic Methods
    def __init__(self: DRGExperimentalCondition) -> None:
        """
        Constructor

        This function...

        Return (None):
            ...
        """

        #   ...

        return

    ### Public Methods
    def ExtractFields(self: DRGExperimentalCondition, Fields: typing.Sequence[str]) -> DRGExperimentalCondition:
        """
        ExtractFields

        This function...

        Fields:
            ...

        Return (self):
            ...
        """

        ColumnIndex: int = 0

        self.LIFFilePath, ColumnIndex                   = (NormalizePathSeparators(Fields[ColumnIndex])), (ColumnIndex + 1)
        self.ExperimentDate, ColumnIndex                = (TryParseDatetime(Fields[ColumnIndex], "%Y%m%d")), (ColumnIndex + 1)
        self.CultureDuration, ColumnIndex               = (TryParseInteger(Fields[ColumnIndex], "Column: [ CultureDuration ]")), (ColumnIndex + 1)
        self.SampleIndex, ColumnIndex                   = (TryParseInteger(Fields[ColumnIndex], "Column: [ SampleIndex ]")), (ColumnIndex + 1)
        self.ImageResolution, ColumnIndex               = (TryParseFloat(Fields[ColumnIndex], "Column: [ ImageResolution ]")), (ColumnIndex + 1)
        self.BrightFieldSeriesIndex, ColumnIndex        = (TryParseInteger(Fields[ColumnIndex], "Column: [ BrightFieldSeriesIndex ]")), (ColumnIndex + 1)
        self.BrightFieldChannelIndex, ColumnIndex       = (TryParseInteger(Fields[ColumnIndex], "Column: [ BrightFieldChannelIndex ]")), (ColumnIndex + 1)
        self.NeuriteSeriesIndex, ColumnIndex            = (TryParseInteger(Fields[ColumnIndex], "Column: [ NeuriteSeriesIndex ]")), (ColumnIndex + 1)
        self.NeuriteChannelIndex, ColumnIndex           = (TryParseInteger(Fields[ColumnIndex], "Column: [ NeuriteChannelIndex ]")), (ColumnIndex + 1)
        self.BaseGel, ColumnIndex                       = (Fields[ColumnIndex]), (ColumnIndex + 1)
        self.GelMAPercentage, ColumnIndex               = (TryParseInteger(Fields[ColumnIndex], "Column: [ GelMAPercentage ]")), (ColumnIndex + 1)
        self.DegreeOfFunctionalization, ColumnIndex     = (TryParseInteger(Fields[ColumnIndex], "Column: [ DegreeOfFunctionalization ]")), (ColumnIndex + 1)
        self.Polymer, ColumnIndex                       = (Fields[ColumnIndex]), (ColumnIndex + 1)
        self.Crosslinker, ColumnIndex                   = (Fields[ColumnIndex]), (ColumnIndex + 1)
        self.Peptide, ColumnIndex                       = (Fields[ColumnIndex]), (ColumnIndex + 1)
        self.PeptideIn, ColumnIndex                     = (Fields[ColumnIndex]), (ColumnIndex + 1)
        self.PeptideConcentration, ColumnIndex          = (TryParseFloat(Fields[ColumnIndex], "Column: [ PeptideConcentration ]")), (ColumnIndex + 1)
        self.DilutionMedia, ColumnIndex                 = (Fields[ColumnIndex]), (ColumnIndex + 1)
        self.FBSInclusion, ColumnIndex                  = (TryParseBool(Fields[ColumnIndex], "Column: [ FBSInclusion ]")), (ColumnIndex + 1)
        self.GelIlluminationTime, ColumnIndex           = (TryParseFloat(Fields[ColumnIndex], "Column: [ GelIlluminationTime ]")), (ColumnIndex + 1)
        self.RutheniumConcentration, ColumnIndex        = (TryParseFloat(Fields[ColumnIndex], "Column: [ RutheniumConcentration ]")), (ColumnIndex + 1)
        self.SodiumPersulfateConcentration, ColumnIndex = (TryParseFloat(Fields[ColumnIndex], "Column: [ SodiumPersulfateConcentration ]")), (ColumnIndex + 1)
        self.RiboflavinConcentration, ColumnIndex       = (TryParseFloat(Fields[ColumnIndex], "Column: [ RiboflavinConcentration ]")), (ColumnIndex + 1)
        self.IKVAV, ColumnIndex                         = (TryParseBool(Fields[ColumnIndex], "Column: [ IKVAV ]")), (ColumnIndex + 1)
        self.IKVAVConcentration, ColumnIndex            = (TryParseFloat(Fields[ColumnIndex], "Column: [ IKVAVConcentration ]")), (ColumnIndex + 1)
        self.Gelatin, ColumnIndex                       = (TryParseBool(Fields[ColumnIndex], "Column: [ Gelatin ]")), (ColumnIndex + 1)
        self.GelatinConcentration, ColumnIndex          = (TryParseFloat(Fields[ColumnIndex], "Column: [ GelatinConcentration ]")), (ColumnIndex + 1)
        self.Glutathione, ColumnIndex                   = (TryParseBool(Fields[ColumnIndex], "Column: [ Glutathione ]")), (ColumnIndex + 1)
        self.GlutathioneConcentration, ColumnIndex      = (TryParseFloat(Fields[ColumnIndex], "Column: [ GlutathioneConcentration ]")), (ColumnIndex + 1)
        self.GDNF, ColumnIndex                          = (TryParseBool(Fields[ColumnIndex], "Column: [ GDNF ]")), (ColumnIndex + 1)
        self.GDNFConcentration, ColumnIndex             = (TryParseFloat(Fields[ColumnIndex], "Column: [ GDNFConcentration ]")), (ColumnIndex + 1)
        self.BDNF, ColumnIndex                          = (TryParseBool(Fields[ColumnIndex], "Column: [ BDNF ]")), (ColumnIndex + 1)
        self.BDNFConcentration, ColumnIndex             = (TryParseFloat(Fields[ColumnIndex], "Column: [ BDNFConcentration ]")), (ColumnIndex + 1)
        self.Laminin, ColumnIndex                       = (TryParseBool(Fields[ColumnIndex], "Column: [ LamininConcentration ]")), (ColumnIndex + 1)
        self.LamininConcentration, ColumnIndex          = (TryParseFloat(Fields[ColumnIndex], "Column: [ Laminin ]")), (ColumnIndex + 1)

        return self

    def SetFolderBase(self: DRGExperimentalCondition, Folder: str) -> DRGExperimentalCondition:
        """
        SetFolderBase

        This function...

        Folder:
            ...

        Return (self):
            ...
        """

        self.LIFFilePath = os.path.join(Folder, self.LIFFilePath)

        #   ...

        return self

    def Validate(self: DRGExperimentalCondition) -> bool:
        """
        Validate

        This function...

        Return (bool):
            ...
        """

        IsValid: bool = True

        #   ...

        return IsValid

    def Describe(self: DRGExperimentalCondition) -> str:
        """
        Describe

        This function...

        Return (str):
            ...
        """

        return "\n".join([
        ])
