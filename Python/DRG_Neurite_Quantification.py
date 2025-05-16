#!/usr/bin/env python3

#   Author: Joseph Sadden
#   Date:   24th January, 2025

#   Script Purpose: This script provides an automated, objective analysis pipeline
#                       for analyzing and quantifying the growth length and directionality
#                       of neurites originating from Dorsal Root Ganglia.

#   Import the necessary standard library modules
from __future__ import annotations
import argparse
from datetime import datetime
import itertools
import os
import platform
import sys
import traceback
import typing

#   Import the necessary third-part modules
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.widgets import PolygonSelector
import numpy as np
import cv2
from scipy.signal import correlate
import scipy.stats

#   Import the desired locally written modules
from MTG_Common import Logger
from MTG_Common import Utils
from MTG_Common import ZStack
from MTG_Common.DRG_Quantification import *
from Alignment_Analysis import PrepareEllipticalKernel, ApplyEllipticalConvolution, CreateOrientationVisualization, ComputeAlignmentMetric, AngleTracker
from MTG_Common.DRG_Quantification import DRGQuantificationResults

DEBUG_DISPLAY_ENABLED: bool = False
DEBUG_DISPLAY_TIMEOUT: float = 0.25
# DEBUG_DISPLAY_TIMEOUT: float = 0

#   Add a sequence number to the images as generated and exported from this script.
ImageSequenceNumber: int = 1

class Configuration():
    """
    Configuration

    This class represents the full configuration state of the script, and all of the
    behaviours and options available to tune or modify the performance of this script.
    """

    #   Public Class Members

    BrightFieldImageFile: str
    BrightFieldImage: ZStack.ZStack
    MIPImageFile: str
    FluorescentImage: ZStack.ZStack

    ManualPreview: bool
    DRGBodyMaskFilename: str
    DRGBodyMask: np.ndarray
    WellInteriorMaskFilename: str
    WellInteriorMask: np.ndarray

    ApplyManualROISelection: bool

    ExperimentalDetails: DRGExperimentalCondition

    OutputDirectory: str
    JSONDirectory: str

    DistinctOrientations: float

    LogFile: str
    QuietMode: bool
    DryRun: bool
    SaveIntermediates: bool
    ValidateOnly: bool
    HeadlessMode: bool

    #   Private Class Members
    _LogWriter: Logger.Logger

    ### Magic Methods
    def __init__(self: Configuration, LogWriter: Logger.Logger = Logger.Discarder) -> None:
        """
        Construtor

        LogWriter:
            The Logger to use to write any and all log messages to during execution.
        """
        self.BrightFieldImageFile = ""
        self.BrightFieldImage = None
        self.FluorescentImageFile = ""
        self.FluorescentImage = None

        self.OutputDirectory = ""
        self.JSONDirectory = ""
        self.ManualPreview = False
        self.DRGBodyMaskFilename = ""
        self.DRGBodyMask = None
        self.WellInteriorMaskFilename = ""
        self.WellInteriorMask = None

        ### +++
        ### Add in all of the settings and values used for identifying a particular experimental condition
        self.ExperimentalDetails = None


        ### ---

        self.ApplyManualROISelection = False

        self.LogFile = ""
        self.QuietMode = False
        self.DryRun = False
        self.SaveIntermediates = False
        self.ValidateOnly = False
        self.HeadlessMode = False

        self._LogWriter = LogWriter

        return

    def __str__(self: Configuration) -> str:
        """
        Stringify

        This function return the information within this configuration object in the form of a human-readable string.
        """

        return "\n".join([
            f"---------- Platform and Software Versioning ----------",
            f"Python Version:                       {platform.python_version()}",
            f"Analysis Platform:                    {platform.platform()}",
            f"Numpy Version:                        {np.version.full_version}",
            f"Matplotlib Version:                   {matplotlib.__version__}",
            f"OpenCV Version:                       {cv2.getVersionString()}",
            f"Scipy Version:                        {scipy.__version__}",
            f"",
            f"---------- Experimental Setup Parameters ----------",
            f"Bright-Field Image Filename:          {os.path.basename(self.BrightFieldImageFile)}",
            f"Fluorescent Image Filename:           {os.path.basename(self.FluorescentImageFile)}",
            f"",
            f"---------- Analysis Tuning and Variable Parameters ----------",
            f"",
            f"---------- Estimated Result Values ----------",
            f"",
            f"---------- Behaviour Enable Parameters ----------",
            f"Headless Mode:                        {self.HeadlessMode}",
            f"Dry-Run Mode:                         {self.DryRun}",
            f"Intermediate Images Saved:            {self.SaveIntermediates}",
            f"Validate Arguments Only:              {self.ValidateOnly}",
            f"",
            f"---------- Output Artefacts ----------",
            f"Artefact Directory:                   {self.OutputDirectory}",
            f"JSON Directory:                       {self.JSONDirectory}",
            f"",
        ])

    ### Public Methods
    def ExtractFromCondition(self: Configuration, ExperimentalCondition: DRGExperimentalCondition) -> Configuration:
        """
        ExtractFromCondition

        This function allows for the setup of the global Configuration instance
        in a batch-processing context. This extracts the necessary details from
        the DRGExperimentalCondition instance into this Configuration.

        ExperimentalCondition:
            An instance of the DRGExperimentalCondition which contains the
            necessary information for describing a particular experimental
            trial.

        Return (self):
            A reference to the filled-out Configuration instance.
        """

        self.ExperimentalDetails = ExperimentalCondition

        ValidCondition: bool = True

        if ( not os.path.exists(ExperimentalCondition.LIFFilePath) ):
            ValidCondition = False
            self._LogWriter.Errorln(f"LIF File does not exist or could not be found!")
            ExperimentalCondition.AnalysisStatus |= DRGAnalysis_StatusCode.StatusNoLIFFile
        else:
            self.BrightFieldImageFile = ExperimentalCondition.LIFFilePath
            self.BrightFieldImage = ZStack.ZStack.FromLIF(ExperimentalCondition.LIFFilePath, SeriesIndex=ExperimentalCondition.BrightFieldSeriesIndex, ChannelIndex=ExperimentalCondition.BrightFieldChannelIndex)
            if ( self.BrightFieldImage is None ):
                self._LogWriter.Errorln(f"Failed to open Bright Field Image with Series and Channel Indices [ {ExperimentalCondition.BrightFieldSeriesIndex},{ExperimentalCondition.BrightFieldChannelIndex} ]!")
                ExperimentalCondition.AnalysisStatus |= DRGAnalysis_StatusCode.NoBrightFieldImage
                ValidCondition = False

            self.FluorescentImageFile = ExperimentalCondition.LIFFilePath
            self.FluorescentImage = ZStack.ZStack.FromLIF(ExperimentalCondition.LIFFilePath, SeriesIndex=ExperimentalCondition.NeuriteSeriesIndex, ChannelIndex=ExperimentalCondition.NeuriteChannelIndex)
            if ( self.FluorescentImage is None ):
                self._LogWriter.Errorln(f"Failed to open Fluorescent Image with Series and Channel Indices [ {ExperimentalCondition.NeuriteSeriesIndex},{ExperimentalCondition.NeuriteChannelIndex} ]!")
                ExperimentalCondition.AnalysisStatus |= DRGAnalysis_StatusCode.NoFluorescentImage
                ValidCondition = False

            self.OutputDirectory = os.path.splitext(ExperimentalCondition.LIFFilePath)[0] + f" - Analyzed {datetime.now().strftime('%Y-%m-%d %H-%M-%S')}"

        #   ...

        if ( not ValidCondition ):
            raise ValueError(f"Failed to properly extract analysis configuration state from the Experimental Condition details - {str(ExperimentalCondition.AnalysisStatus)}!")

        return self

    def ExtractArguments(self: Configuration, Arguments: argparse.Namespace) -> None:
        """
        ExtractArguments

        This function extracts all of the command-line arguments into the Configuration object, providing a single
        reference point for translation from command-line settings to application configuration state.

        Arguments:
            The argparse.Namespace resulting from parsing the command-line arguments.

        Return (None):
            None, the configuration object is modified with the results of the command-line arguments.
        """

        self.BrightFieldImageFile = Arguments.BrightField
        self.FluorescentImageFile = Arguments.MIPImage

        self.ApplyManualROISelection = Arguments.ManualROI
        self.ManualPreview = Arguments.PreCheck
        self.DRGBodyMaskFilename = Arguments.DRGBodyMask
        self.WellInteriorMaskFilename = Arguments.WellInteriorMask

        self.OutputDirectory = Arguments.OutputDirectory + f" - Analyzed {datetime.now().strftime('%Y-%m-%d %H-%M-%S')}"
        self.JSONDirectory = Arguments.JSONDirectory

        #   ...

        self.LogFile = Arguments.LogFile
        self.QuietMode = Arguments.Quiet

        self.DryRun = Arguments.DryRun
        self.ValidateOnly = Arguments.Validate
        self.HeadlessMode = Arguments.Headless
        self.SaveIntermediates = Arguments.SaveIntermediates

        return

    def ValidateArguments(self: Configuration) -> typing.Tuple[bool, bool]:
        """
        ValidateArguments

        This function will validate the arguments as given from the command-line to make sure they
        are semantically valid and sensible.s

        Return (Tuple):
            [0] - bool:
                Validated - A boolean indicating that the arguments have passed validation
            [1] - bool:
                ValidateOnly - A boolean indicating whether the script should ONLY peform validation and then exit.
        """

        Validated: bool = True

        if ( self.QuietMode ):
            self._LogWriter.SetOutputFilename("/dev/null")
        elif ( self.LogFile == "-" ):
            self._LogWriter.SetOutputStream(sys.stdout)
        else:
            self._LogWriter.SetOutputFilename(os.path.join(self.OutputDirectory, self.LogFile))

        if ( not os.path.exists(self.BrightFieldImageFile) ):
            self._LogWriter.Errorln(f"Bright Field image file does not exist!")
            Validated &= False
        else:
            self.BrightFieldImage = ZStack.ZStack.FromFile(self.BrightFieldImageFile)
            self._LogWriter.Println(f"Working with bright-field image file [ {self.BrightFieldImageFile} ]...")

        if ( not os.path.exists(self.FluorescentImageFile) ):
            self._LogWriter.Errorln(f"Fluorescent image file does not exist!")
            Validated &= False
        else:
            self.FluorescentImage = ZStack.ZStack.FromFile(self.FluorescentImageFile)
            self._LogWriter.Println(f"Working with fluorescent image file [ {self.FluorescentImageFile} ]...")

        if ( self.DRGBodyMaskFilename != "" ):
            self.DRGBodyMask = Utils.GammaCorrection(cv2.imread(self.DRGBodyMaskFilename), Minimum=0, Maximum=1)
            self._LogWriter.Println(f"Working with DRG Body mask image file [ {self.DRGBodyMaskFilename} ]...")

        if ( self.WellInteriorMaskFilename != "" ):
            self.WellInteriorMask = Utils.GammaCorrection(cv2.imread(self.WellInteriorMaskFilename), Minimum=0, Maximum=1)
            self._LogWriter.Println(f"Working with well interior mask image file [ {self.WellInteriorMaskFilename} ]...")

        if ( self.ApplyManualROISelection ) and ( self.HeadlessMode ):
            self.ApplyManualROISelection = False
            self._LogWriter.Warnln(f"Manual ROI Filtering is not compatible with headless operation... Skipping manual ROI selection and continuing.")

        if ( self.DryRun ):
            self.SaveIntermediates = False

        return (Validated, self.ValidateOnly)

    def Save(self: Configuration) -> bool:
        """
        Save

        This function saves out the Configuration instance
        for later review.

        Return (bool):
            Flag for whether the save operation was successful.
        """

        return self._SaveText()

    ### Private Methods
    def _SaveText(self: Configuration) -> bool:
        """
        _SaveText

        This function specifically saves out the configuration in text format.

        Return (bool):
            Boolean for whether the save operation was successful.
        """

        Stringified: str = str(self)

        if ( not self.DryRun ):
            if ( not os.path.exists(self.OutputDirectory) ):
                os.makedirs(self.OutputDirectory, mode=0o775, exist_ok=True)
            with open(os.path.join(self.OutputDirectory, f"Configuration Parameters.txt"), "w+") as ConfigFile:
                ConfigFile.write(Stringified)
        else:
            LogWriter.Write(Stringified)

        return True

class QuantificationIntermediates():
    """
    QuantificationIntermediates

    This class...
    """

    OriginalBrightField: ZStack.ZStack
    BrightFieldMinProjection: np.ndarray
    BrightFieldBinarized: np.ndarray
    BrightFieldExclusionMask: np.ndarray
    BodyCentroidLocation: typing.Tuple[int, int]

    OriginalFluorescent: ZStack.ZStack
    BinarizedFluorescent: ZStack.ZStack
    MaskedFluorescent: ZStack.ZStack
    FilteredFluorescent: ZStack.ZStack
    ManuallySelectedFluorescent: ZStack.ZStack

    NeuriteDistances: typing.Sequence[np.ndarray]
    MaximumNeuriteDistance: int
    ColourAnnotatedNeuriteLengths: ZStack.ZStack

    NeuriteOrientations: typing.Sequence[np.ndarray]
    ColourAnnotatedNeuriteOrientations: ZStack.ZStack

    def __init__(self: QuantificationIntermediates, LogWriter: Logger.Logger) -> None:
        """
        Constructor

        This function...

        LogWriter:
            ...

        Return (None):
            ...
        """

        self.OriginalBrightField = ZStack.ZStack(Name="Original Bright Field")
        self.BrightFieldMinProjection = np.array([])
        self.BrightFieldBinarized = np.array([])
        self.BrightFieldExclusionMask = np.array([])
        self.BodyCentroidLocation = ()

        self.OriginalFluorescent = ZStack.ZStack(Name="Original Fluorescent")
        self.BinarizedFluorescent = ZStack.ZStack(Name="Binarized Fluorescent")
        self.MaskedFluorescent = ZStack.ZStack(Name="Masked Binarized Fluorescent")
        self.FilteredFluorescent = ZStack.ZStack(Name="Component Filtered Binarized Fluorescent")
        self.ManuallySelectedFluorescent = ZStack.ZStack(Name="Component Filtered Binarized Fluorescent with Manual ROI Selection")

        self.NeuriteDistances = []
        self.MaximumNeuriteDistance = 0
        self.ColourAnnotatedNeuriteLengths = ZStack.ZStack(Name="Colour-Annotated Neurite Lengths")

        self.NeuriteOrientations = []
        self.ColourAnnotatedNeuriteOrientations = ZStack.ZStack(Name="Colour-Annotated Neurite Orientations")

        self._LogWriter = LogWriter

        return

    def Save(self: QuantificationIntermediates, Folder: str, DryRun: bool) -> bool:
        """
        Save

        This function...

        Folder:
            ...

        Return (bool):
            ...
        """

        if ( Folder is None ) or ( Folder == "" ):
            raise ValueError(f"Output folder must be specified!")

        if ( not DryRun ):
            if ( not os.path.exists(Folder) ):
                os.makedirs(Folder, 0o755, exist_ok=True)


            self.OriginalBrightField.SetName("Bright Field").SaveTIFF(Folder)
            Utils.WriteImage(self.BrightFieldMinProjection, os.path.join(Folder, "Bright Field Minimum Intensity.tif"))
            Utils.WriteImage(self.BrightFieldBinarized, os.path.join(Folder, "Bright Field Binarized.tif"))
            Utils.WriteImage(self.BrightFieldExclusionMask, os.path.join(Folder, "Bright Field Exclusion Mask.tif"))
            self.OriginalFluorescent.SaveTIFF(Folder)
            self.BinarizedFluorescent.SaveTIFF(Folder)
            self.MaskedFluorescent.SaveTIFF(Folder)
            self.FilteredFluorescent.SaveTIFF(Folder)
            if ( Config.ApplyManualROISelection ):
                self.ManuallySelectedFluorescent.SaveTIFF(Folder)
            self.ColourAnnotatedNeuriteLengths.SaveTIFF(Folder)
            self.ColourAnnotatedNeuriteOrientations.SaveTIFF(Folder)

            with open(os.path.join(Config.OutputDirectory, f"Neurite Lengths by Layer.csv"), "w+") as NeuriteLengthsFile:
                BinCount: int = int(round(max([np.max(x) if len(x) > 0 else 0 for x in self.NeuriteDistances])))
                if ( BinCount > 0 ):
                    NeuriteLengthsFile.writelines(",".join([
                        f"{x}" for x in range(0, BinCount)
                    ] + ["\n\n"]))
                    for Layer in self.NeuriteDistances:
                        hist, _ = np.histogram(Layer, bins=BinCount)
                        NeuriteLengthsFile.writelines(",".join([
                            f"{h}" for h in hist
                        ]) + "\n")

            with open(os.path.join(Config.OutputDirectory, f"Neurite Orientations By Layer.csv"), "w+") as NeuriteOrientationsFile:
                BinCount: int = Config.DistinctOrientations
                if ( BinCount > 0 ):
                    NeuriteOrientationsFile.writelines(",".join([
                        f"{x}" for x in range(0, 180, int(180 / BinCount))
                    ] + ["\n\n"]))
                    for Layer in self.NeuriteOrientations:
                        hist, bins = np.histogram(Layer, bins=BinCount)
                        NeuriteOrientationsFile.writelines(",".join([
                            f"{h}" for h in hist
                        ]) + "\n")

        return True

#   Define the globals to set by the command-line arguments
LogWriter: Logger.Logger = Logger.Logger(Prefix=os.path.basename(sys.argv[0]))
Config: Configuration = Configuration(LogWriter=LogWriter)
QuantificationStacks: QuantificationIntermediates = QuantificationIntermediates(LogWriter=LogWriter)
Results: DRGQuantificationResults = DRGQuantificationResults()

#   Main
#       This is the main entry point of the script.
def main() -> int:

    global ImageSequenceNumber
    global Results
    ImageSequenceNumber = 1

    if ( Config.ManualPreview ):
        return ManualPreviewImages(Config.BrightFieldImage.MinimumIntensityProjection(), Config.FluorescentImage.MaximumIntensityProjection())

    #   This script will take in two images, a MIP from the fluorescence Z-stack and a bright-field image
    #   of the same size, magnification, and ROI.

    #   Take the bright-field image and use this to identify the centroid of the DRG body,
    #   as well as to generate a mask to remove the DRG body from the fluorescent MIP.
    QuantificationStacks.OriginalBrightField = Config.BrightFieldImage
    CentroidLocation, DRGBodyMask, WellEdgeMask = ProcessBrightField(Config.BrightFieldImage.MinimumIntensityProjection())

    MasksStatus = SanityCheckMasks(DRGBodyMask, WellEdgeMask)
    if ( MasksStatus != DRGAnalysis_StatusCode.StatusSuccess ):

        #   Try to use the alternative mask generation algorithms?
        LogWriter.Println(f"Attempting to generate masks with alternative algorithms...")
        CentroidLocation, DRGBodyMask, WellEdgeMask = ProcessBrightField(Config.BrightFieldImage.MinimumIntensityProjection(), AlternativeMaskGeneration=True)
        MasksStatus = SanityCheckMasks(DRGBodyMask, WellEdgeMask)
        if ( MasksStatus != DRGAnalysis_StatusCode.StatusSuccess ):
            LogWriter.Warnln(f"Alternative mask generation algorithms also failed!")
            return MasksStatus
        LogWriter.Println(f"Alternative mask generation algorithms succeeded.")

    LogWriter.Println(f"Starting to process fluorescent image...")
    for Index, Layer in enumerate(Config.FluorescentImage.Layers()):
        LogWriter.Println(f"Processing Layer [ {Index+1}/{len(Config.FluorescentImage.Layers())} ]")

        #   Take the fluorescent image and segment out the neurite growth pixels
        Neurites: np.ndarray = ProcessFluorescent(Layer.copy(), DRGBodyMask, WellEdgeMask)

        #   If the user has selected they would like to apply manual ROI selection to exclude specific noise regions,
        #   perform this now.
        Neurites, ManualExclusionMask = ApplyManualROI(Neurites, Layer.copy())
        if ( Config.ApplyManualROISelection ):
            DisplayAndSaveImage(Utils.ConvertTo8Bit(ManualExclusionMask), "Polygon Exclusion Mask", Config.DryRun, Config.HeadlessMode)
            DisplayAndSaveImage(Utils.ConvertTo8Bit(Neurites), "Polygon Exclusion Masked Image", Config.DryRun, Config.HeadlessMode)
        QuantificationStacks.ManuallySelectedFluorescent.Append(Utils.ConvertTo8Bit(Neurites))

        #   With the centroid location and neurite pixels now identified, quantify the distribution of lengths of neurites
        QuantificationStacks.NeuriteDistances.append(QuantifyNeuriteLengths(Neurites, CentroidLocation))

        FeatureSizePx: float = 50 / 0.7644
        Config.DistinctOrientations = 90
        DistinctOrientations = Config.DistinctOrientations
        QuantificationStacks.NeuriteOrientations.append(QuantifyNeuriteOrientations(Layer.copy(), Neurites, CentroidLocation, FeatureSizePx, DistinctOrientations))

    LogWriter.Println(f"Finished processing fluorescent image.")

    QuantificationStacks.MaximumNeuriteDistance = int(round(max([np.max(x) if len(x) > 0 else 0 for x in QuantificationStacks.NeuriteDistances])))
    GenerateNeuriteLengthVisualization(QuantificationStacks.OriginalFluorescent, QuantificationStacks.ManuallySelectedFluorescent, QuantificationStacks.NeuriteDistances, CentroidLocation)

    CreateQuantificationFigures(list(itertools.chain.from_iterable(QuantificationStacks.NeuriteDistances)))

    #   Format and structure the intermediate results as computed here to be printed out to be further processed later
    Results = PrepareResults(Results)

    #   Save out the configuration state for possible later review.
    Config.Save()
    QuantificationStacks.Save(Folder=Config.OutputDirectory, DryRun=Config.DryRun)

    #   Save the results to the per-execution folder
    Results.Save(Folder=Config.OutputDirectory, DryRun=Config.DryRun)

    #   ... and to the potential global JSON directory if this is different.
    if ( Config.JSONDirectory != Config.OutputDirectory ):
        Results.Save(Folder=Config.JSONDirectory, DryRun=Config.DryRun)

    #   Quickly check that at least some neurites were identified in the analysis, otherwise return a special status code to indicate this.
    if ( max([np.max(x) if len(x) > 0 else 0 for x in QuantificationStacks.NeuriteDistances]) == 0 ):
        return DRGAnalysis_StatusCode.StatusNoNeurites

    return DRGAnalysis_StatusCode.StatusSuccess

def ManualPreviewImages(BrightFieldProjection: np.ndarray, FluorescentProjection: np.ndarray) -> int:
    """
    ManualPreviewImages

    This function...

    BrightFieldProjection:
        ...
    FluorescentProjection:
        ...

    Return (int):
        ...
    """

    LogWriter.Println(f"Manually previewing DRG images. Should these be processed further? (y/N)")

    KeyCode: int = Utils.DisplayImages(
        [
            ("Bright Field Minimum Intensity Projection", Utils.ConvertTo8Bit(BrightFieldProjection)),
            ("Fluorescent Maximum Intensity Projection", Utils.ConvertTo8Bit(FluorescentProjection)),
        ],
        0,
        True,
        True
    )

    if ( KeyCode in [ord(x) for x in 'yY'] ):
        return DRGAnalysis_StatusCode.StatusPreviewAccepted
    else:
        return DRGAnalysis_StatusCode.StatusPreviewRejected

def DisplayAndSaveImage(Image: np.ndarray, Description: str, DryRun: bool, Headless: bool) -> None:
    """
    DisplayAndSaveImage

    This function displays (if acceptable) a given image, and saves (if permitted) the image file to disk.

    Image:
        The image to display to the screen
    Description:
        A descriptor for the image. Used for the window name of the display, and used to construct the filename
        of the saved image.

    Return (None):
        None, the image is displayed and/or saved, as permitted by the --headless and --dry-run options.
    """

    global ImageSequenceNumber

    #   Display the image to the screen
    Utils.DisplayImage(f"{ImageSequenceNumber} - {Description}", Image, DEBUG_DISPLAY_TIMEOUT, True, (not Headless) and DEBUG_DISPLAY_ENABLED)

    #   Save the image to disk.
    if ( not DryRun ):
        if ( Utils.WriteImage(Image, os.path.join(Config.OutputDirectory, f"{ImageSequenceNumber} - {Description}.png")) ):
            LogWriter.Println(f"Wrote out image [ {ImageSequenceNumber} - {Description}.png ] to [ {Config.OutputDirectory}/ ]...")
        else:
            LogWriter.Errorln(f"Failed to write out image [ {ImageSequenceNumber} - {Description}.png ] to [ {Config.OutputDirectory}/ ]...")

    ImageSequenceNumber += 1

    return

def ProcessBrightField(BrightFieldImage: np.ndarray, AlternativeMaskGeneration: bool = False) -> typing.Tuple[typing.Tuple[int, int], np.ndarray, np.ndarray]:
    """
    ProcessBrightField

    This function handles the full processing of the bright-field or transmitted light image. This
    is where the centroid of the DRG body is identified, and masks for removing the DRG body and PDMS regions
    of the experimental chip are constructed.

    BrightFieldImage:
        The bright-field image to process, formatted as a bright background with dark features.
        This will be converted to an 8-bit greyscale image internally.

    Return (Tuple):
        [0] - Tuple[int, int]:
            The estimated position of the centroid of the DRG body within the image. (X, Y)
        [1] - np.ndarray:
            A numpy mask equal in size to the bright field image, which can be used to mask out the
            body of the DRG in the fluorescent image.
        [2] - np.ndarray:
            A numpy mask equal in size to the bright field image, which can be used to mask in the
            interior region of the chip wells wtihin the fluorescent image.s
    """

    LogWriter.Println(f"Starting to process Bright-Field image...")

    #   First, convert the bright field image to a full-range 8-bit image and assert that it is greyscale.
    Image: np.ndarray = Utils.ConvertTo8Bit(Utils.BGRToGreyscale(BrightFieldImage))
    DisplayAndSaveImage(Image, "Initial Bright-Field Image", not Config.SaveIntermediates, Config.HeadlessMode)
    QuantificationStacks.BrightFieldMinProjection = Image

    #   Next, assert that the image is generally a bright background with a dark foreground
    if ( np.median(Image) < np.mean(Image) ):
        Image = -Image
        DisplayAndSaveImage(Image, "Inverted Bright-Field Image", not Config.SaveIntermediates, Config.HeadlessMode)
        QuantificationStacks.BrightFieldMinProjection = Image

    #   Create a copy of the image to binarize in order to perform the search for the centroid of the DRG body
    BinarizedImage, ThresholdLevel = BinarizeBrightField(Image.copy())
    DisplayAndSaveImage(BinarizedImage, f"Binarized Image ({ThresholdLevel if ThresholdLevel >= 0 else float('NaN'):.0f})", not Config.SaveIntermediates, Config.HeadlessMode)
    QuantificationStacks.BrightFieldBinarized = BinarizedImage

    Centroid: typing.Tuple[int, int] = (0, 0)
    if ( Config.DRGBodyMask is not None ):
        Centroid = EstimateCentroid(Config.DRGBodyMask.copy())
    else:
        #   Using this binarized image, identify the centroid of the DRG body
        Centroid: typing.Tuple[int, int] = EstimateCentroid(BinarizedImage)

    #   Annotate where the centroid of the DRG body is found to be, and display this to the user...
    CentroidAnnotated: np.ndarray = cv2.circle(Utils.GreyscaleToBGR(Image.copy()), Centroid, 10, (0, 0, 255), -1)
    DisplayAndSaveImage(CentroidAnnotated, "Centroid Annotated DRG Body", not Config.SaveIntermediates, Config.HeadlessMode)
    QuantificationStacks.BodyCentroidLocation = Centroid

    DRGBodyMask: np.ndarray = None
    if ( Config.DRGBodyMask is not None ):
        #   With the centroid identified, review the bright field image and compute a mask which covers the
        #   body of the DRG.
        DRGBodyMask: np.ndarray = Config.DRGBodyMask.copy()
    else:
        if ( AlternativeMaskGeneration ):
            DRGBodyMask: np.ndarray = ComputeDRGMask_Alt1(BinarizedImage, Centroid)
        else:
            DRGBodyMask: np.ndarray = ComputeDRGMask(BinarizedImage, Centroid)
    DisplayAndSaveImage(Utils.ConvertTo8Bit(DRGBodyMask), "DRG Body Mask", not Config.SaveIntermediates, Config.HeadlessMode)

    WellEdgeMask: np.ndarray = None
    if ( Config.WellInteriorMask is not None ):
        WellEdgeMask = Config.WellInteriorMask.copy()
    else:
        #   Compute the mask of the well edge within the image, as this is typically the source of more noise signals than anywhere else
        if ( AlternativeMaskGeneration ):
            WellEdgeMask: np.ndarray = ComputeWellEdgeMask_Alt1(BinarizedImage, DRGBodyMask, Centroid)
        else:
            WellEdgeMask: np.ndarray = ComputeWellEdgeMask(BinarizedImage, Centroid)
    DisplayAndSaveImage(Utils.ConvertTo8Bit(WellEdgeMask), "Well Edge Mask", not Config.SaveIntermediates, Config.HeadlessMode)

    QuantificationStacks.BrightFieldExclusionMask = Utils.ConvertTo8Bit(DRGBodyMask * WellEdgeMask)

    LogWriter.Println(f"Finished processing Bright-Field Image.")
    return (Centroid, DRGBodyMask, WellEdgeMask)

def BinarizeBrightField(Image: np.ndarray) -> typing.Tuple[np.ndarray, int]:
    """
    BinarizeBrightField

    This function...

    Image:
        ...

    Return (Tuple):
        [0] - np.ndarray:
            ...
        [1] - int:
            ...
    """

    ThresholdLevel, BinarizedImage = cv2.threshold(Image.copy(), 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    return BinarizedImage, ThresholdLevel

def EstimateCentroid(ThresholdedImage: np.ndarray, CorrelationThreshold: float = 0.975, KernelStepSize: int = 10, InitialKernelSize: int = 51, CentroidJitterThreshold: int = 1) -> typing.Tuple[int, int]:
    """
    EstimateCentroid

    This function estimates the location of the centroid of the DRG within the provided pre-thresholded and binarized image,
    using an iterative cross-correlation search method to identify the location of the largest and most consistently circular-ish figure
    within the image.

    ThresholdedImage:
        The binarized image of the chip well, with the DRG body as a bright foreground on a dark background.
    CorrelationThreshold:
        The lower bound for what constitutes a "strongly correlated region", to use when identifying
        which regions should be included in the centroid calculation.
    KernelStepSize:
        How much the kernel diameter should increase in each step of the algorithm.
    InitialKernelSize:
        The original diameter of the kernel used to identify the DRG body.
    CentroidJitterThreshold:
        A threshold on how much the estimaged DRG centroid can vary between iterations before
        it is deemed to have converged.

    Return (Tuple):
        [0] - int:
            The X coordinate of the DRG centroid
        [1] - int:
            The Y coordinate of the DRG centroid
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
        ImageMoments = cv2.moments(CorrelationMap, binaryImage=False)

        #   Actually compute the centroid location
        if ( ImageMoments["m00"] != 0 ):
            Centroid_X = int(ImageMoments["m10"] / ImageMoments["m00"])
            Centroid_Y = int(ImageMoments["m01"] / ImageMoments["m00"])
        else:
            Centroid_X, Centroid_Y = 0, 0

        #   DEBUGGING
        # Utils.DisplayImage(f"DRG Centroid Correlation Map: [ K={KernelSize} ]", cv2.circle(Utils.GreyscaleToBGR(Utils.ConvertTo8Bit(CorrelationMap.copy())), (Centroid_X, Centroid_Y), 10, (0, 0, 255), -1), DEBUG_DISPLAY_TIMEOUT, True, DEBUG_DISPLAY_ENABLED and ( not Config.HeadlessMode ))
        #   DEBUGGING

        #   If the centroid does not exist, then the kernel is improperly sized so we exit.
        if ( Centroid_X == 0 ) and ( Centroid_Y == 0 ):
            LogWriter.Warnln(f"Failed to identify centroid location for kernel of size [ {KernelSize} ]...")
            break

        #   Check where the current averaged centroid location is...
        Current_X, Current_Y = 0, 0
        if ( len(Centroid_Xs) > 0 ) and ( len(Centroid_Ys) > 0 ):
            Current_X, Current_Y = tuple([int(np.mean(np.array(x))) for x in [Centroid_Xs, Centroid_Ys]])

        Centroid_Xs.append(Centroid_X)
        Centroid_Ys.append(Centroid_Y)
        Next_X, Next_Y = tuple([int(np.mean(np.array(x))) for x in [Centroid_Xs, Centroid_Ys]])

        # If centroid starts to converge, take its position as the estimated centroid
        if (abs(Next_X - Current_X) < CentroidJitterThreshold ) and ((Next_Y - Current_Y) < CentroidJitterThreshold ):
            return (Next_X, Next_Y)

    return tuple([int(np.mean(np.array(x))) for x in [Centroid_Xs, Centroid_Ys]])

def ComputeDRGMask(ThresholdedImage: np.ndarray, DRGCentroid: typing.Tuple[int, int]) -> np.ndarray:
    """
    ComputeDRGMask

    This function takes in the binarized bright-field image of the DRG body and computes
    a mask for excluding all DRG body pixels from the fluorescence image.

    ThresholdedImage:
        The pre-thresholded binary bright-field image of the DRG body within the chip well.
    DRGCentroid:
        The estimated location of the DRG centroid within the image, in (X, Y) pixel coordinates

    Return (np.ndarray):
        An image mask suitable to be applied to remove all pixels within the interior
        of the DRG body, without affecting any other pixels of the image.
    """

    #   Define all of the tunable parameters of this function in one place
    MinimumComponentArea: int = 100
    MaximumComponentExtent: int = ThresholdedImage.shape[0] * 2 / 3
    MaximumDistanceThreshold: float = 5
    MorphologyKernelSize: int = 25

    #   First, identify all of the connected components of the image, as the DRG body should be a single
    #   block of pixels
    NumberOfComponents, Labels, Stats, Centroids = cv2.connectedComponentsWithStats(ThresholdedImage, connectivity=8)

    #   Prepare a blank image to write the contents of the mask into.
    DRGMask: np.ndarray = np.ones_like(ThresholdedImage)

    #   Determine the distances of each component centroid from the previously identified DRG centroid
    OrderedComponents: np.ndarray = np.array(sorted([
        (x, np.linalg.norm(np.array(DRGCentroid) - np.array(Centroids[x]))) for x in range(1, NumberOfComponents)
    ], key=lambda x: x[1]))
    MaximumDistanceThreshold *= OrderedComponents[0,1]  #   Scale the maximum distance threshold by the distance between the centroid and the closest component.

    #   Look through each of the components identified and build up the mask of what to exclude from the fluorescent image
    for ComponentIndex, ComponentID in enumerate(OrderedComponents[:,0], start=1):
        ComponentID = int(ComponentID)

        #   Check the size of the component, to eliminate small spot-noise which often makes it through the thresholding
        #   This area is generally in units of pixels, so we want to eliminate things which are "small"
        ComponentArea = Stats[ComponentID, cv2.CC_STAT_AREA]
        if ( ComponentArea <= MinimumComponentArea ):
            # LogWriter.Println(f"Component [ {ComponentIndex}/{len(OrderedComponents[:,0])} ] rejected from DRG mask with too small of an area.")
            continue

        Width, Height = Stats[ComponentID, cv2.CC_STAT_WIDTH], Stats[ComponentID, cv2.CC_STAT_HEIGHT]
        if ( Width >= MaximumComponentExtent ) or ( Height >= MaximumComponentExtent ):
            # LogWriter.Println(f"Component [ {ComponentIndex}/{len(OrderedComponents[:,0])} ] rejected from DRG mask with too large of a height or width.")
            continue

        #   Next, check if this component is "close" to the DRG Centroid identified earlier
        Distance: float = np.linalg.norm(np.array(DRGCentroid) - np.array(Centroids[ComponentID]))
        if ( Distance >= MaximumDistanceThreshold ):
            # LogWriter.Println(f"Component [ {ComponentIndex}/{len(OrderedComponents[:,0])} ] rejected from DRG mask as too far from DRG centroid.")
            continue

        DRGMask[Labels == ComponentID] = 0

        #   DEBUGGING
        # Utils.DisplayImage(f"Current DRG Mask", Utils.ConvertTo8Bit(DRGMask), 0, True, True)
        #   DEBUGGING

    #   Finally, clean up the edges of the mask with an Opening morphological transformation
    DRGMask = cv2.morphologyEx(DRGMask, cv2.MORPH_OPEN, kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MorphologyKernelSize, MorphologyKernelSize)))
    return DRGMask

def ComputeDRGMask_Alt1(ThresholdedImage: np.ndarray, DRGCentroid: typing.Tuple[int, int]) -> np.ndarray:
    """
    ComputeDRGMask_Alt1

    This function...

    ThresholdedImage:
        ...
    DRGCentroid:
        ...

    Return (np.ndarray):
        ...
    """

    Mask: np.ndarray = np.zeros_like(Utils.BGRToGreyscale(ThresholdedImage))

    #   First, identify the connected components of the image.
    ComponentCount, Labels, Stats, Centroids = cv2.connectedComponentsWithStats(ThresholdedImage, connectivity=8)

    #   Next, remove any and all components occupying less than 1% of the area of the image
    FilteredComponentIndices: typing.List[int] = [
        x for x in range(1, ComponentCount) if Stats[x, cv2.CC_STAT_AREA] >= (0.01 * np.prod(ThresholdedImage.shape))
    ]

    #   Next, find any components where the previously identified DRG centroid is within
    #   the component's bounding box
    ComponentIDs, FilteredComponentIndices = FilteredComponentIndices, []
    for ComponentID in ComponentIDs:
        x, y, w, h = Stats[ComponentID, cv2.CC_STAT_LEFT], Stats[ComponentID, cv2.CC_STAT_TOP], Stats[ComponentID, cv2.CC_STAT_WIDTH], Stats[ComponentID, cv2.CC_STAT_HEIGHT]
        BoundingBox = np.array([[x,y], [x+w, y], [x+w, y+h], [x, y+h]])
        if ( cv2.pointPolygonTest(BoundingBox, DRGCentroid, False) > 0 ):
            FilteredComponentIndices.append(ComponentID)

    if ( len(FilteredComponentIndices) == 0 ):
        LogWriter.Errorln(f"Failed to identify a component with the DRG Centroid located inside!")
        return np.zeros_like(ThresholdedImage)

    #   If there are multiple such components, identify the one with the smallest
    #   moment of inertia, i.e. the one most spatially concentrated about the
    #   centroid of the component
    DRGComponentIndex = sorted([
        (
            x,
            cv2.HuMoments(
                cv2.moments(
                    (Labels == x).astype(np.uint8)
                )
            )[0]
        ) for x in FilteredComponentIndices
    ], key=lambda x: x[1])[0][0]

    #   With the single component containing the DRG centroid identified, now we just need
    #   to filter this down to ensure we only keep pixels of the actual DRG body and
    #   remove any segments of the well wall which are attached to the body itself.
    #
    #   To do this, consider the L2 normed distances of all identified pixels
    #   from the centroid. Within the body of the DRG, we expect to see more points
    #   at larger distances, as the shape is approximately circular. Once we start to
    #   examine radii larger than the radius of the DRG, the counts should decrease again,
    #   with a tail depending on what else is connected to this component.
    MaskCoordinates: np.ndarray = np.argwhere(Labels == DRGComponentIndex)
    Distances: np.ndarray = np.hypot(MaskCoordinates[:,0] - DRGCentroid[1], MaskCoordinates[:,1] - DRGCentroid[0])
    DistanceCounts: np.ndarray = np.histogram(Distances, bins=int(round(np.max(Distances))))[0]
    Skewness: np.ndarray = np.array([scipy.stats.skew(DistanceCounts[:x]) if x > np.argmax(DistanceCounts) else 0 for x in range(len(DistanceCounts))])
    MinIndex: int = np.argmin(Skewness)
    ZeroCrossing: int = MinIndex + np.argmin(Skewness[MinIndex:] <= 0)
    EffectiveRadius: int = int(round((MinIndex + ZeroCrossing) / 2))

    ### DEBUGGING
    F: Figure = Utils.PrepareFigure()
    Ax = F.add_subplot(111)
    Ax.set_title(f"Pixel Counts from DRG Centroid")
    Ax.set_xlabel(f"Distance (px)")
    Ax.set_ylabel(f"Pixel Counts")
    Ax.plot(DistanceCounts)
    DisplayAndSaveImage(Utils.FigureToImage(F), f"Pixel Counts versus Radial Distance", not Config.SaveIntermediates, Config.HeadlessMode)

    F.clear()
    Ax = F.add_subplot(111)
    Ax.set_title(f"Pixel Counts from DRG Centroid - Skewness versus Radial Distance")
    Ax.set_xlabel(f"Distance (px)")
    Ax.set_ylabel(f"Histogram Skewness")
    Ax.plot(Skewness, label="Histogram Skewness")
    Ax.vlines(MinIndex, ymin=np.min(Skewness), ymax=np.max(Skewness), colors='r', label=f"Minimum Skewness: {MinIndex}")
    Ax.vlines(ZeroCrossing, ymin=np.min(Skewness), ymax=np.max(Skewness), colors='g', label=f"Zero Skewness: {ZeroCrossing}")
    Ax.vlines(EffectiveRadius, ymin=np.min(Skewness), ymax=np.max(Skewness), colors='k', label=f"Effective DRG Radius: {EffectiveRadius}")
    DisplayAndSaveImage(Utils.FigureToImage(F), f"DRG Mask Skewness versus Distance", not Config.SaveIntermediates, Config.HeadlessMode)

    DisplayAndSaveImage(cv2.circle(Utils.GreyscaleToBGR(ThresholdedImage.copy()), DRGCentroid, MinIndex, (0, 255, 0), 5), f"Annotated DRG Radius (Minimum Skewness)", not Config.SaveIntermediates, Config.HeadlessMode)
    DisplayAndSaveImage(cv2.circle(Utils.GreyscaleToBGR(ThresholdedImage.copy()), DRGCentroid, EffectiveRadius, (0, 255, 255), 5), f"Annotated DRG Radius (Effective Radius)", not Config.SaveIntermediates, Config.HeadlessMode)
    DisplayAndSaveImage(cv2.circle(Utils.GreyscaleToBGR(ThresholdedImage.copy()), DRGCentroid, ZeroCrossing, (0, 0, 255), 5), f"Annotated DRG Radius (Zero Skewness)", not Config.SaveIntermediates, Config.HeadlessMode)
    F.clear()
    ### DEBUGGING

    #   Generate the mask from the component and restrict it to only the circle we associate
    #   with the DRG body.
    Mask[Labels == DRGComponentIndex] = 1
    Mask *= cv2.circle(np.zeros_like(Mask), DRGCentroid, EffectiveRadius, 1, -1)

    #   Now, invert this mask to follow the expected format of an exclusion mask, i.e.
    #   zeros where we want to remove and ones where we want to keep.
    return (~Mask - 254)

def ComputeWellEdgeMask(ThresholdedImage: np.ndarray, DRGCentroid: typing.Tuople[int, int]) -> np.ndarray:
    """
    ComputeWellEdgeMask

    This function takes in the binarized bright-field image of the DRG body and computes
    a mask for excluding all pixels outside of the chip well in view.

    ThresholdedImage:
        The pre-thresholded binary bright-field image of the DRG body within the chip well.
    DRGCentroid:
        The estimated location of the DRG centroid within the image, in (X, Y) pixel coordinates

    Return (np.ndarray):
        An image mask suitable to be applied to remove all pixels outside of the well
        interior within the chip.
    """

    #   Define all of the tunable parameters of this function in one place
    MinimumComponentExtent: int = ThresholdedImage.shape[0] * 2 / 3
    ContourMinimumAreaThreshold: int = 500
    MorphologyKernelSize: int = 25

    #   First, identify all of the connected components of the image, as the DRG body should be a single
    #   block of pixels
    NumberOfComponents, Labels, Stats, Centroids = cv2.connectedComponentsWithStats(ThresholdedImage, connectivity=8)

    #   Prepare a blank image to write the contents of the mask into.
    WellEdgeMask: np.ndarray = np.ones_like(ThresholdedImage)

    #   Determine the distances of each component centroid from the previously identified DRG centroid
    OrderedComponents: np.ndarray = np.array(sorted([
        (x, Stats[x, cv2.CC_STAT_AREA]) for x in range(1, NumberOfComponents)
    ], key=lambda x: x[1], reverse=True))

    #   Look through each of the components identified and build up the mask of what to exclude from the fluorescent image
    for ComponentID in OrderedComponents[:,0]:
        ComponentID = int(ComponentID)

        Height, Width = Stats[ComponentID, cv2.CC_STAT_HEIGHT], Stats[ComponentID, cv2.CC_STAT_WIDTH]
        if ( Height < MinimumComponentExtent ) and ( Width < MinimumComponentExtent ):
            continue

        WellEdgeMask[Labels == ComponentID] = 0

    # #   Write a border around the entire image one pixel wide, to separate the image into enclosed regions.
    WellEdgeMask[0, :] = 0
    WellEdgeMask[-1, :] = 0
    WellEdgeMask[:, 0] = 0
    WellEdgeMask[:, -1] = 0

    #   With the component corresponding to the well edge identified, determine the "inside" and "outside",
    #   so that the entire region not contained by the wells can be masked out.
    Contours, _ = cv2.findContours(WellEdgeMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    Contours = sorted(Contours, key=lambda x: cv2.contourArea(x), reverse=True)
    for Contour in Contours:
        ContourArea: float = cv2.contourArea(Contour)
        if ( ContourArea >= ContourMinimumAreaThreshold ) and ( cv2.pointPolygonTest(Contour, DRGCentroid, False) != 1 ):
            WellEdgeMask = cv2.drawContours(WellEdgeMask, [Contour], 0, 0, -1)

    #   Finally, clean up the edges of the mask with an Opening morphological transformation
    WellEdgeMask = cv2.morphologyEx(WellEdgeMask, cv2.MORPH_OPEN, kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MorphologyKernelSize, MorphologyKernelSize)))

    return WellEdgeMask

def ComputeWellEdgeMask_Alt1(ThresholdedImage: np.ndarray, DRGBodyMask: np.ndarray, DRGCentroid: typing.Tuple[int, int]) -> np.ndarray:
    """
    ComputeWellEdgeMask_Alt1

    This function...

    ThresholdedImage:
        ...

    Return (np.ndarray):
        ...
    """

    Mask: np.ndarray = np.zeros_like(ThresholdedImage)

    #   First, again we segment into connected components, but this time with a
    #   much smaller allowed area threshold as we're okay with relatively small
    #   regions of chip-edge making it through this selection.
    ComponentCount, Labels, Stats, Centroids = cv2.connectedComponentsWithStats(ThresholdedImage, connectivity=8)

    #   Next, remove any and all components occupying less than 0.05% of the area of the image
    FilteredComponentIndices: typing.List[int] = list(sorted([
        x for x in range(1, ComponentCount) if Stats[x, cv2.CC_STAT_AREA] >= (0.0005 * np.prod(ThresholdedImage.shape))
    ], key=lambda x: Stats[x, cv2.CC_STAT_AREA], reverse=True))

    #   Write out these filtered components to a new mask
    FilteredComponentMask: np.ndarray = np.ones_like(ThresholdedImage)
    for ComponentID in FilteredComponentIndices:
        FilteredComponentMask[Labels == ComponentID] = 0
        # Utils.DisplayImage(f"Current Components", Utils.ConvertTo8Bit(FilteredComponentMask), 0, True, True)

    #   Apply a very large-kernel close operation to close any little speckle noise
    #   on the interior of any components
    KernelSize: int = int(round(0.02 * np.min(ThresholdedImage.shape)))
    FilteredComponentMask = cv2.morphologyEx(FilteredComponentMask, cv2.MORPH_ERODE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (KernelSize, KernelSize)))

    Contours, Hierarchy = cv2.findContours(FilteredComponentMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    SortedContours = sorted([
            x for x in Contours if (cv2.contourArea(x) >= 0.01 * np.prod(ThresholdedImage.shape)) and (cv2.pointPolygonTest(cv2.boxPoints(cv2.minAreaRect(x)).astype(np.int_), DRGCentroid, False) > 0)
        ],
        key=lambda x: cv2.contourArea(x),
        reverse=True
    )

    ContourMoments: np.ndarray = np.array([cv2.HuMoments(cv2.moments(cv2.drawContours(np.zeros_like(Mask), [x], 0, 255, -1), True))[0] for x in SortedContours])

    for Index, Contour in enumerate(SortedContours):
        ContourMask: np.ndarray = cv2.drawContours(np.zeros_like(Mask), [Contour], 0, 1, -1)
        # Utils.DisplayImage(f"Current Test Contour", Utils.ConvertTo8Bit(ContourMask), 0, True, True)

        #   Check if the contour contains the DRG centroid itself.
        #   If the contour contains the DRG centroid, then this is clearly the interior of the well
        #   and we can use this region directly.
        if ( cv2.pointPolygonTest(Contour, DRGCentroid, False) > 0 ):
            Mask = cv2.drawContours(Mask, [Contour], 0, 1, -1)

        #   Otherwise, if this region does not contain the DRG centroid, we then need to
        #   look more carefully at what this region looks like and how it's distributed
        #   over the image.

        #   Check if the moment of inertia is less than the mean
        #   of the components containing the centroid within the bounding box
        elif ( ContourMoments[Index] <= np.mean(ContourMoments) ):
            Mask = cv2.drawContours(Mask, [Contour], 0, 1, -1)


        # Utils.DisplayImage(f"Current Mask", Utils.ConvertTo8Bit(Mask), 0, True, True)

    # Utils.DisplayImage(f"Current Mask", Utils.ConvertTo8Bit(Mask), 0, True, True)
    return Mask

def SanityCheckMasks(DRGBodyMask: np.ndarray, WellEdgeMask: np.ndarray) -> int:
    """
    SanityCheckMasks

    This function...

    DRGBodyMask:
        ...
    WellEdgeMask:
        ...

    Return (int):
        ...
    """

    MaskStatus: int = 0

    DRGMaskMinArea: float = 0.025
    DRGMaskMaxArea: float = 0.50

    WellInteriorMinArea: float = 0.15

    #   The DRG Body mask is 0's where the body is, and 1's everywhere else.
    #   A valid mask should have a "reasonable" fraction of the image covered,
    #   but not too much or too little.
    #   A simple check for the ratio of 0's to 1's is the mean value of the
    #   entire image.
    DRGBodyFraction: float = 1 - np.mean(DRGBodyMask)
    if ( DRGBodyFraction < DRGMaskMinArea ) or ( DRGBodyFraction > DRGMaskMaxArea ):
        MaskStatus |= DRGAnalysis_StatusCode.StatusBodyMaskFailed
        LogWriter.Warnln(f"DRG Body mask coverage fraction is concerningly high (or low)! [ {DRGBodyFraction:.2f} ]")
    #   ...

    #   As for the well interior mask, this should similarly constitute a meaningful
    #   fraction of the area of the image. In this case, the interior of the well
    #   is filled with 1's and the walls and exterior with 0's.
    WellInteriorFraction: float = np.mean(WellEdgeMask)
    if ( WellInteriorFraction < WellInteriorMinArea ):
        MaskStatus |= DRGAnalysis_StatusCode.StatusWellMaskFailed
        LogWriter.Warnln(f"Well interior mask coverage fraction is concerningly low! [ {WellInteriorFraction:.2f} ]")

    #   If both masks are acceptable, return a success code.
    if ( MaskStatus == 0 ):
        return DRGAnalysis_StatusCode.StatusSuccess

    return MaskStatus

def ProcessFluorescent(FluorescentImage: np.ndarray, DRGBodyMask: np.ndarray, WellEdgeMask: np.ndarray) -> np.ndarray:
    """
    ProcessFluorescent

    This function handles all of the processing to apply to the fluorescence image
    in order to extract out the neurite length and orientation information from the image.

    FluorescentImage:
        A flattened maximum intensity Z-projection of the original fluoresence Z-stack.
    DRGBodyMask:
        The pre-computed mask for removing pixels within the interior of the DRG body from
        the fluorescence image.
    WellEdgeMask:
        The pre-computed mask for removing pixels outside the bounds of the well of the chip.

    Return (np.ndarray):
        A binary image with the pixels corresponding to neurites set as non-zero values, while
        all non-neurite pixels are set to 0.
    """

    #   Define all of the tunable parameters in one place
    #   TODO: Make these configuration settings
    AdaptiveKernelSize: int = 45
    AdaptiveOffset: int = -5
    MaskExpansionSize: int = int(AdaptiveKernelSize / 4)
    SpeckleComponentAreaThreshold: int = 25
    NeuriteAspectRatioThreshold: float = 1.5
    NeuriteInfillFractionThreshold: float = 0.75 * (np.pi / 4)

    #   First, convert the image to a full-range 8-bit image and assert that it is greyscale.
    Image: np.ndarray = Utils.ConvertTo8Bit(Utils.BGRToGreyscale(FluorescentImage))
    DisplayAndSaveImage(Image, "Initial Fluorescent Image", not Config.SaveIntermediates, Config.HeadlessMode)
    QuantificationStacks.OriginalFluorescent.Append(Image)

    #   Apply an adaptive local threshold over the image to further select out the neurite pixels
    BinarizedImage: np.ndarray = BinarizeFluorescent(Image, AdaptiveKernelSize, AdaptiveOffset)
    DisplayAndSaveImage(BinarizedImage, f"Binarized Fluorescent Image (k={AdaptiveKernelSize}, C={AdaptiveOffset})", not Config.SaveIntermediates, Config.HeadlessMode)
    QuantificationStacks.BinarizedFluorescent.Append(BinarizedImage)

    #   Expand the original DRG Body mask again slightly, to not include artefacts from the "edge" introduced
    #   by applying the mask
    BinarizedImage = ApplyExclusionMask(BinarizedImage, cv2.erode(DRGBodyMask, kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(MaskExpansionSize, MaskExpansionSize))))
    DisplayAndSaveImage(BinarizedImage, f"DRG Body Mask Edge Removed", not Config.SaveIntermediates, Config.HeadlessMode)

    #   Expand the original Well Edge mask again slightly, to not include artefacts from the "edge" introduced
    #   by applying the mask
    BinarizedImage = ApplyExclusionMask(BinarizedImage, cv2.erode(WellEdgeMask, kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(MaskExpansionSize, MaskExpansionSize))))
    DisplayAndSaveImage(BinarizedImage, f"Well Edge Mask Edge Removed", not Config.SaveIntermediates, Config.HeadlessMode)
    QuantificationStacks.MaskedFluorescent.Append(BinarizedImage)

    #   Finally, separate out all of the components of the image and filter them to remove any which do not
    #   satisfy the expectations of neurites
    FilteredNeuriteComponents = FilterNeuriteComponents(BinarizedImage, SpeckleComponentAreaThreshold, NeuriteAspectRatioThreshold, NeuriteInfillFractionThreshold)
    DisplayAndSaveImage(Utils.ConvertTo8Bit(FilteredNeuriteComponents), "Filtered Connected Components after Local Thresholding", not Config.SaveIntermediates, Config.HeadlessMode)
    QuantificationStacks.FilteredFluorescent.Append(Utils.ConvertTo8Bit(FilteredNeuriteComponents))

    return FilteredNeuriteComponents

def BinarizeFluorescent(Image: np.ndarray, KernelSize: int, ThresholdValue: int) -> np.ndarray:
    """
    BinarizeFluorescent

    This function...

    Image:
        ...
    KernelSize:
        ...
    ThresholdValue:
        ...
    """

    ThresholdedImage: np.ndarray = cv2.adaptiveThreshold(Image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, KernelSize, ThresholdValue)

    return ThresholdedImage

def ApplyExclusionMask(Image: np.ndarray, Mask: np.ndarray) -> np.ndarray:
    """
    ApplyExclusionMask

    This function...

    Image:
        ...
    Mask:
        ...

    Return (np.ndarray):
        ...
    """

    Mask = Utils.GammaCorrection(Mask.astype(np.float64), Gamma=1, Minimum=0.0, Maximum=1.0)

    return Utils.ConvertTo8Bit(Image * Mask)

def FilterNeuriteComponents(Image: np.ndarray, SpeckleAreaThreshold: int, NeuriteAspectRatioThreshold: float, NeuriteInfillFraction: float) -> np.ndarray:
    """
    FilterNeuriteComponents

    This function...

    Image:
        ...

    Return (np.ndarray):
        ...
    """

    #   We want to removal little speckle noise and other very small regions which make it through the thresholding.
    #   Identify connected components and only accept those larger than some area threshold
    NumberOfComponents, Labels, Stats, Centroids = cv2.connectedComponentsWithStats(Image, connectivity=4)

    if ( NumberOfComponents == 0 ):
        raise RuntimeError(f"Failed to identify any components within the image!")

    #   Determine the area of each component, given by the number of pixels it contains.
    OrderedComponents: np.ndarray = np.array(sorted([
        (x, Stats[x, cv2.CC_STAT_AREA]) for x in range(1, NumberOfComponents) if Stats[x, cv2.CC_STAT_AREA] > SpeckleAreaThreshold
    ], key=lambda x: x[1], reverse=True))

    #   Re-build the components into a single image to work with, containing only the meaningful components of the image.
    FilteredComponents: np.ndarray = np.zeros_like(Image)
    for Index, (ComponentID, ComponentArea) in enumerate(OrderedComponents):

        #   Apply Filtering based off the circularity of the components.
        #   We'd expect neurites to be either high aspect ratio, or if the bounding box is close to square, the filled area would be low.
        ComponentMask: np.ndarray = (Labels == ComponentID).astype(np.uint8)

        Contours, _ = cv2.findContours(ComponentMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        #   We only look at the first contour, since we only draw a single component into the temporary image.
        if ( len(Contours) > 0 ):

            Rectangle: cv2.RotatedRect = cv2.minAreaRect(Contours[0])
            (Centre, (Width, Height), Orientation) = Rectangle

            if ( Width == 0 ) or ( Height == 0 ):
                continue

            AspectRatio: float = np.max([Height, Width]) / np.min([Height, Width])
            BoundingArea: int = Height * Width
            FilledFraction: float = ComponentArea / BoundingArea

            if ( AspectRatio < NeuriteAspectRatioThreshold ):
                if ( FilledFraction > NeuriteInfillFraction ):
                    continue

            FilteredComponents[Labels == ComponentID] = 1

    return FilteredComponents

def ApplyManualROI(ImageToFilter: np.ndarray, Background: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
    """
    ApplyManualROI

    This function allows the user to manually draw closed polygons over the image to construct
    a mask for removing any pixels or regions which have erroneously been considered as neurites.
    This builds up a new exclusion mask to apply to the image, removing all pixels within
    the polygons drawn by the user.

    ImageToFilter:
        The current image of identified neurite pixels, which can be manually further filtered by this function.
    Background:
        A background image to provide context into what the identified pixels correspond to, generally the
        fluorescent image.

    Return (Tuple):
        [0] - np.ndarray:
            The resulting masked image, with the user-selected regions removed.
        [1] - np.ndarray:
            The mask generated by the user, for visualization or potential later re-use.
    """

    #   Pre-fill the exclusion mask with all ones, to include all pixels.
    PolygonExclusionMask: np.ndarray = np.ones_like(Utils.BGRToGreyscale(ImageToFilter))

    if ( Config.ApplyManualROISelection ):

        F, Ax = plt.subplots()

        #   Prepare an overlaid image with the background being shown as-is, and the foreground neurite pixels
        #   only in the green channel.
        Base: np.ndarray = Utils.GammaCorrection(Utils.GreyscaleToBGR(Background).astype(np.float64), Minimum=0, Maximum=1.0)
        Foreground: np.ndarray = Utils.GammaCorrection(Utils.GreyscaleToBGR(ImageToFilter).astype(np.float64), Minimum=0, Maximum=1.0)
        Foreground[:,:,0] = 0
        Foreground[:,:,2] = 0
        Alpha: np.ndarray = Foreground.copy()

        Result = (Foreground * Alpha) + (Base * (1.0 - Alpha))

        #   Actually display the image and allow the user to select points on the image to draw the polygons to exclude.
        #   TODO: Update the candidate image between each exclusion mask application?
        AxIm = Ax.imshow(Result, origin='upper')

        #   Define the callback to run if and when the user draws a closed polygon.
        def UpdateExclusionMask(Vertices: np.ndarray) -> None:

            #   When a closed polygon is drawn, convert it into a set of (X,Y) coordinate points,
            #   and draw this as a contour over the image, infilling with 0's to mask out the
            #   interior region of the contour.
            nonlocal PolygonExclusionMask
            nonlocal Ax
            Vertices = np.array([
                (int(X), int(Y)) for (X, Y) in Vertices
            ])
            PolygonExclusionMask = cv2.drawContours(PolygonExclusionMask, [Vertices], 0, 0, -1)
            Utils.DisplayImage(f"Current Polygon Exclusion Result", Utils.ConvertTo8Bit(PolygonExclusionMask * ImageToFilter), DEBUG_DISPLAY_TIMEOUT, True, True)

            #   Prepare an overlaid image with the background being shown as-is, and the foreground neurite pixels
            #   only in the green channel.
            Base: np.ndarray = Utils.GammaCorrection(Utils.GreyscaleToBGR(Background).astype(np.float64), Minimum=0, Maximum=1.0)
            Foreground: np.ndarray = Utils.GammaCorrection(Utils.GreyscaleToBGR(PolygonExclusionMask * ImageToFilter).astype(np.float64), Minimum=0, Maximum=1.0)
            Foreground[:,:,0] = 0
            Foreground[:,:,2] = 0
            Alpha: np.ndarray = Foreground.copy()

            Result = (Foreground * Alpha) + (Base * (1.0 - Alpha))

            Ax.imshow(Utils.ConvertTo8Bit(Result), origin='upper')

        P = PolygonSelector(Ax, onselect=UpdateExclusionMask, props=dict(color='r', linestyle='-', linewidth=2))
        plt.show(block=True)


    PolygonMaskedImage: np.ndarray = Utils.BGRToGreyscale(ImageToFilter) * PolygonExclusionMask
    return PolygonMaskedImage, PolygonExclusionMask

def QuantifyNeuriteLengths(NeuritePixels: np.ndarray, Origin: typing.Tuple[int, int]) -> np.ndarray:
    """
    QuantifyNeuriteLengths

    This function handles the actual quantification operation for transforming the image
    of identified neurite pixels, and the centroid of the DRG body, into a distribution of lengths
    of neurites within the image.

    NeuritePixels:
        The image with identified neurite pixels as non-zero values.
    Origin:
        The (X, Y) coordinates to consider as the origin from which distances are to be measured.

    Return (np.ndarray):
        A 1D numpy array of the L2-normed distances as measured from the estimated DRG centroid location.
    """

    #   Identify the indices of the image corresponding to the neurite pixels
    NeuriteCoordinates = np.argwhere(NeuritePixels != 0)

    #   Compute the L2 norm from the origin point to each identified neurite pixel
    Distances: np.ndarray = np.hypot(NeuriteCoordinates[:,0] - Origin[1], NeuriteCoordinates[:,1] - Origin[0])

    #   Convert from pixel distances to physical units, assuming the "Spatial Resolution" value in the configuration object is non-zero.
    if ( Config.ExperimentalDetails is not None ):
        if ( Config.ExperimentalDetails.ImageResolution is not None ) and ( Config.ExperimentalDetails.ImageResolution != 0 ):
            Distances *= Config.ExperimentalDetails.ImageResolution

    return Distances

def GenerateNeuriteLengthVisualization(BaseImages: ZStack.ZStack, NeuritePixels: ZStack.ZStack, Distances: typing.Sequence[np.ndarray], Origin: typing.Tuple[int, int]) -> None:

    MaximumLength: float = max([np.max(x) if len(x) > 0 else 0 for x in Distances])
    if ( MaximumLength == 0 ):
        LogWriter.Warnln(f"No neurites were identified!")
        return

    for Index, (Layer, BaseImage, LayerDistances) in enumerate(zip(NeuritePixels.Layers(), BaseImages.Layers(), Distances)):
        NeuriteCoordinates = np.argwhere(Layer != 0)

        #   Prepare a visualization of the neurite lengths, with increasing length corresponding to varying hue
        NeuriteLengthVisualization: np.ndarray = Utils.GreyscaleToBGR(np.zeros_like(Layer))

        #   Add in the centroid location
        for Index, Distance in enumerate(LayerDistances):
            NeuriteLengthVisualization[NeuriteCoordinates[Index, 0], NeuriteCoordinates[Index, 1], :] = (180 * (Distance / MaximumLength), 255, 255)
        DisplayAndSaveImage(cv2.cvtColor(NeuriteLengthVisualization, cv2.COLOR_HSV2BGR), "Neurite Length Visualization", not Config.SaveIntermediates, Config.HeadlessMode)

        Base: np.ndarray = Utils.GammaCorrection(Utils.GreyscaleToBGR(BaseImage.copy()).astype(np.float64), Minimum=0, Maximum=1.0)
        Foreground: np.ndarray = Utils.GammaCorrection(cv2.cvtColor(NeuriteLengthVisualization, cv2.COLOR_HSV2BGR).astype(np.float64), Minimum=0, Maximum=1.0)
        Alpha: np.ndarray = Utils.GammaCorrection(Utils.GreyscaleToBGR(Layer).astype(np.float64), Minimum=0, Maximum=1.0)

        Result = (Foreground * Alpha) + (Base * (1.0 - Alpha))
        Result = cv2.circle(Result, Origin, 10, (0, 0, 1.0), -1)
        DisplayAndSaveImage(Utils.ConvertTo8Bit(Result), "Colour Annotated Identified Neurites", not Config.SaveIntermediates, Config.HeadlessMode)
        QuantificationStacks.ColourAnnotatedNeuriteLengths.Append(Utils.ConvertTo8Bit(Result))

    return

def QuantifyNeuriteOrientations(BaseImage: np.ndarray, NeuritePixels: np.ndarray, CentroidLocation: typing.Tuple[int, int], FeatureSizePx: float, DistinctOrientations: int) -> np.ndarray:
    """
    QuantifyNeuriteOrientations

    This function...


    Return
    """

    #   TODO: This also needs to be extracted into a dedicated function...
    #   TODO: This needs to be derived from command-line settings, and extracted into a dedicated function for orientation details.
    #   Finally, compute the orientation details of the neurites within the image.
    EllipticalFilterKernelSize: int     = Utils.RoundUpKernelToOdd(int(round(FeatureSizePx * 1.15)))
    EllipticalFilterSigma: float        = EllipticalFilterKernelSize / 2.0
    EllipticalFilterMinSigma: float     = EllipticalFilterSigma / 50.0
    EllipticalFilterScaleFactor: float  = EllipticalFilterKernelSize / 20.0

    #   Prepare the base elliptical kernel to work with.
    EllipticalKernel: np.ndarray = PrepareEllipticalKernel(EllipticalFilterKernelSize, EllipticalFilterSigma, EllipticalFilterMinSigma, EllipticalFilterScaleFactor)

    #   Apply the elliptical convoluation and identify all of the orientations
    Mask, RawOrientations = ApplyEllipticalConvolution(NeuritePixels, DistinctOrientations, EllipticalKernel)

    Mask[NeuritePixels == 0] = 0
    OrientationVisualization: np.ndarray = CreateOrientationVisualization(Utils.ConvertTo8Bit(BaseImage), RawOrientations, Mask)

    OrientationVisualization = cv2.circle(OrientationVisualization, CentroidLocation, 10, (0, 0, 255), -1)
    DisplayAndSaveImage(OrientationVisualization, "Colour Annotated Neurite Orientations", not Config.SaveIntermediates, Config.HeadlessMode)
    QuantificationStacks.ColourAnnotatedNeuriteOrientations.Append(OrientationVisualization)

    Count, AlignmentFraction, MeanOrientation, AngularStDev, Orientations = ComputeAlignmentMetric(RawOrientations[Mask != 0].flatten())
    Angles: AngleTracker = AngleTracker(LogWriter, Config.OutputDirectory, Video=False, DryRun=Config.DryRun).SetAngularResolution(180.0 / DistinctOrientations)
    OrientationPlot, _= Angles.Update(Orientations, MeanOrientation, AngularStDev, Count, AlignmentFraction, Config.HeadlessMode)
    DisplayAndSaveImage(Utils.FigureToImage(OrientationPlot), "Neurite Orientations", not Config.SaveIntermediates, Config.HeadlessMode)

    return RawOrientations[Mask != 0].flatten()

def CreateQuantificationFigures(NeuriteLengths: np.ndarray) -> None:
    """
    CreateQuantificationFigures

    This function creates the figures used to report the quantified neurite lengths as identified
    from the images.

    NeuriteLengths:
        A 1D numpy array of the raw distance values for neurite pixels.

    Return (None):
        None, the figures are generated and optionally displayed and saved to disk.
    """

    BinCount: int = 100
    n, bins = np.histogram(NeuriteLengths, bins=BinCount, density=True)
    median, mean, stdev = np.median(NeuriteLengths), np.mean(NeuriteLengths), np.std(NeuriteLengths)

    F: Figure = Utils.PrepareFigure(Interactive=(not Config.HeadlessMode))
    A: Axes = F.add_subplot(111)

    F.suptitle(f"<Experimental Identification Here>")
    A.set_title(f"Neurite Length Quantification - Total Pixel Count {len(NeuriteLengths)}")
    A.set_xlabel(f"Neurite Length (px)")
    A.set_ylabel(f"Normalized Pixel Count")

    A.plot(bins[:-1], n, color='b')
    A.vlines(median, ymin=0, ymax=np.max(n), label=f"Median Length ({median:.0f}px)", color='g')
    A.vlines(mean, ymin=0, ymax=np.max(n), label=f"Mean Length ({mean:.0f}px)", color='r')
    A.vlines([mean + stdev, mean - stdev], ymin=0, ymax=np.max(n), label=f"1σ Length ({stdev:.0f}px)", color='k')
    A.legend()

    DisplayAndSaveImage(Utils.FigureToImage(F), "Neurite Length Distribution", not Config.DryRun, Config.HeadlessMode)

    return

def PrepareResults(Results: DRGQuantificationResults) -> DRGQuantificationResults:
    """
    PrepareResults

    This function...

    Results:
        ...

    Return (DRGQuantificationResults):
        ...
    """

    Results.Processed = True

    #   Compute a hash of the source file (Bright Field versus Fluorescent is arbitrary) to generate a unique ID for file naming.
    Results.SourceHash = Utils.Sha256Sum(Config.BrightFieldImageFile)

    #   Store the location of the DRG centroid within the image
    Results.DRGCentroidLocation = list(QuantificationStacks.BodyCentroidLocation)

    #   Compute the fraction of the image area in which growth is considered possible.
    GrowthRegionSize: int = np.count_nonzero(QuantificationStacks.BrightFieldExclusionMask) #   The total number of pixels within the image in which growth is considered possible.
    Results.InclusionMaskFraction = float(GrowthRegionSize / np.prod(QuantificationStacks.BrightFieldExclusionMask.shape))

    #   For each layer of the fluorescent stack, compute the count of neurite pixels at each integer distance from the DRG centroid.
    Results.NeuriteDistancesByLayer = {
        f"{LayerIndex}": (np.histogram(Z, bins=QuantificationStacks.MaximumNeuriteDistance)[0]).tolist() if len(Z) > 0 else [] for LayerIndex, Z in enumerate(QuantificationStacks.NeuriteDistances, start=1)
    }

    #   For each layer of the fluorescent stack, compute the median distance of the neurite pixels from the DRG centroid.
    Results.MedianNeuriteDistancesByLayer = {
        f"{LayerIndex}": int(np.median(Z)) if len(Z) > 0 else 0 for LayerIndex, Z in enumerate(QuantificationStacks.NeuriteDistances, start=1)
    }

    #   Collapse all of the neurite distance values, and compute the median distance across the entire 3D volume.
    Results.MedianNeuriteDistance = float(np.median(np.array(list(itertools.chain.from_iterable(QuantificationStacks.NeuriteDistances)))))

    #   Compute the fraction of the possible growth area actually occupied by neurites within each layer of the fluorescent stack.
    Results.NeuriteDensityByLayer = {
        f"{LayerIndex}": float(np.count_nonzero(Z) / GrowthRegionSize) for LayerIndex, Z in enumerate(QuantificationStacks.FilteredFluorescent.Layers(), start=1)
    }

    #   Take the maximum intensity projection of the neurite pixel stack and compute the fraction of the available space actually occupied by neurites.
    Results.NeuriteDensity = float(np.count_nonzero(QuantificationStacks.FilteredFluorescent.MaximumIntensityProjection()) / GrowthRegionSize)

    #   Report how many possible orientations were checked during the neurite orientation analysis
    Results.OrientationAngularResolution = 180.0 / Config.DistinctOrientations

    #   Report the count of neurite pixels with each of the possible orientations, for each layer of the fluorescent stack.
    Results.NeuriteOrientationsByLayer = {
        f"{LayerIndex}": (np.histogram(Z, bins=Config.DistinctOrientations)[0]).tolist() if len(Z) > 0 else [] for LayerIndex, Z in enumerate(QuantificationStacks.NeuriteOrientations, start=1)
    }

    #   Report additionally the count of pixels, mean orientation, angular standard deviation, and "alignment fraction" values for each layer.
    Results.OrientationMetricsByLayer = {
        f"{LayerIndex}": tuple(ComputeAlignmentMetric(Z)[:-1]) for LayerIndex, Z in enumerate(QuantificationStacks.NeuriteOrientations, start=1) if len(Z) > 0
    }

    #   Report the same four metrics as above, but for the total set of neurite orientations as identified from the full stack.
    Results.OrientationMetrics = tuple(ComputeAlignmentMetric(np.array(list(itertools.chain.from_iterable(QuantificationStacks.NeuriteOrientations))))[:-1])

    return Results

def HandleArguments() -> bool:
    """
    HandleArguments

    This function sets up, parses, extracts, and validates the command-line arguments for the script.

    Return (bool):
        A boolean indicating whether or not the script should continue executing after this function returns.
    """

    #   Prepare the argument parser
    Flags: argparse.ArgumentParser = argparse.ArgumentParser(description="wlkhg")

    #   Add in the command-line flags to accept
    #   Add in the flags for the bright-field and fluorescent MIP images to work from.
    Flags.add_argument("--bf-image",  dest="BrightField", metavar="file-path", type=str, required=True, help="The file path to the bright-field image file to work with.")
    Flags.add_argument("--mip-image", dest="MIPImage",    metavar="file-path", type=str, required=True, help="The file path to the maximum intensity image computed from the fluorescent Z-stack to work with.")
    Flags.add_argument("--manual-roi", dest="ManualROI", action="store_true", required=False, default=False, help="...")

    Flags.add_argument("--qualitative-pre-check", dest="PreCheck", action="store_true", required=False, default=False, help="Preview the images before processing, to check whether or not they are worth processing.")
    Flags.add_argument("--drg-mask", dest="DRGBodyMask", metavar="file-path", type=str, required=False, default="", help="The file path to a pre-defined image mask to remove the DRG Body.")
    Flags.add_argument("--well-mask", dest="WellInteriorMask", metavar="file-path", type=str, required=False, default="", help="The file path to a pre-defined image mask to include only the well interior.")

    #   Add in the flag specifying where the results generated by this script should be written out to.
    Flags.add_argument("--results-directory", dest="OutputDirectory", metavar="folder-path", type=str, required=False, default=os.getcwd(), help="The path to the base folder into which results will be written on a per-execution basis.")
    Flags.add_argument("--json-directory", dest="JSONDirectory", metavar="folder-path", type=str, required=False, default=os.getcwd(), help="The path to the base folder into which the compiled JSON results will be written on a per-execution basis.")
    #   ...

    #   Add in flags for manipulating the logging functionality of the script.
    Flags.add_argument("--log-file", dest="LogFile", metavar="file-path", type=str, required=False, default="-", help="File path to the file to write all log messages of this program to.")
    Flags.add_argument("--quiet",    dest="Quiet",   action="store_true",  required=False, default=False, help="Enable quiet mode, disabling logging of eveything except fatal errors.")

    #   Finally, add in scripts for modifying the basic environment or end-state of the script.
    Flags.add_argument("--dry-run",             dest="DryRun",              action="store_true", required=False, default=False, help="Enable dry-run mode, where no file-system alterations or heavy computations are performed.")
    Flags.add_argument("--save-intermediates",  dest="SaveIntermediates",   action="store_true", required=False, default=False, help="Enable saving of intermediate processing artefacts.")
    Flags.add_argument("--validate",            dest="Validate",            action="store_true", required=False, default=False, help="Only validate the command-line arguments, do not proceed with the remainder of the program.")
    Flags.add_argument("--headless",            dest="Headless",            action="store_true", required=False, default=False, help="Run in 'headless' mode, where nothing is displayed to the screen.")
    #   ...

    #   Parse out the arguments
    Arguments: argparse.Namespace = Flags.parse_args()

    #   Extract out the arguments into the global configuration object
    Config.ExtractArguments(Arguments)

    #   Validate the arguments as extracted
    Validated, ValidateOnly = Config.ValidateArguments()

    #   Indicate whether or not to continue processing
    return ( Validated ) and ( not ValidateOnly )

#   Allow this script to be called from the command-line and execute the main function.
#   If anything needs to happen before executing main, add it here before the call.
if __name__ == "__main__":
    if ( HandleArguments() ):
        try:
            main()
        except Exception as e:
            LogWriter.Fatalln(f"Exception raised in main(): [ {e} ]\n\n{''.join(traceback.format_exception(e, value=e, tb=e.__traceback__))}")
            sys.exit(DRGAnalysis_StatusCode.StatusUnknownException)
    else:
        if ( Config.ValidateOnly ):
            sys.exit(DRGAnalysis_StatusCode.StatusSuccess)

        sys.exit(DRGAnalysis_StatusCode.StatusValidationFailed)
