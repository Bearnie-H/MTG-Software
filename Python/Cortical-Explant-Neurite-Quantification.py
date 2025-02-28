#!/usr/bin/env python3

#   Author: Joseph Sadden
#   Date:   24th January, 2025

#   Script Purpose: This script provides an automated, objective analysis pipeline
#                       for analyzing and quantifying the growth length and directionality
#                       of neurites originating from Cortical Explants.

#   Import the necessary standard library modules
from __future__ import annotations
import argparse
import itertools
import os
import sys
import traceback
import typing

#   Import the necessary third-part modules
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.widgets import PolygonSelector
import numpy as np
import cv2
from scipy.signal import correlate

#   Import the desired locally written modules
from MTG_Common import Logger
from MTG_Common import Utils
from MTG_Common import ZStack
from Alignment_Analysis import PrepareEllipticalKernel, ApplyEllipticalConvolution, CreateOrientationVisualization, ComputeAlignmentMetric, AngleTracker

DEBUG_DISPLAY_ENABLED: bool = True
DEBUG_DISPLAY_TIMEOUT: float = 2

#   Add a sequence number to the images as generated and exported from this script.
ImageSequenceNumber: int = 1

#   Add in top-level return codes for the status of processing the script.
#   These signal to the environment whether or not everything processed
#   correctly, or if the script encountered an error during processing.
STATUS_SUCCESS: int                         = 0
STATUS_NO_CHIP_MASK: int                    = 1
STATUS_NO_EXPLANT_MASK: int                 = 2

class Configuration():
    """
    Configuration

    This class represents the full configuration state of the script, and all of the
    behaviours and options available to tune or modify the performance of this script.
    """

    #   Public Class Members

    BrightFieldImageFile: str
    NuclearStainedImageFile: str
    NeuriteStainedImageFile: str
    RodsStainedImageFile: str

    BrightFieldImage: ZStack.ZStack
    NuclearStainedImage: ZStack.ZStack
    NeuriteStainedImage: ZStack.ZStack
    RodsStainedImage: ZStack.ZStack

    EnableManualROISelection: bool

    OutputDirectory: str

    LogFile: str
    QuietMode: bool
    DryRun: bool
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
        self.NuclearStainedImageFile = ""
        self.NeuriteStainedImageFile = ""
        self.RodsStainedImageFile = ""

        self.BrightFieldImage = None
        self.NuclearStainedImage = None
        self.NeuriteStainedImage = None
        self.RodsStainedImage = None

        self.EnableManualROISelection = False

        self.LogFile = ""
        self.QuietMode = False
        self.DryRun = False
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
        ])

    ### Public Methods
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
        self.NuclearStainedImageFile = Arguments.NuclearStain
        self.NeuriteStainedImageFile = Arguments.NeuriteStain
        self.RodsStainedImageFile = Arguments.RodsStain

        self.EnableManualROISelection = Arguments.ManualROI

        self.OutputDirectory = Arguments.OutputDirectory    #   TODO: disambiguate by time of execution

        #   ...

        self.LogFile = Arguments.LogFile
        self.QuietMode = Arguments.Quiet

        self.DryRun = Arguments.DryRun
        self.ValidateOnly = Arguments.Validate
        self.HeadlessMode = Arguments.Headless

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

        self.NeuriteStainedImage = ZStack.ZStack.FromFile(self.NeuriteStainedImageFile)
        if ( self.NeuriteStainedImage is None ):
            self._LogWriter.Errorln(f"Failed to open neurite image file [ {self.NeuriteStainedImageFile} ]!")
            Validated &= False

        if ( self.BrightFieldImageFile is not None ) and ( self.BrightFieldImageFile != "" ):
            self.BrightFieldImage = ZStack.ZStack.FromFile(self.BrightFieldImageFile)
            if ( self.BrightFieldImage is None ):
                self._LogWriter.Errorln(f"Failed to open Bright-Field image file [ {self.BrightFieldImageFile} ]!")
                Validated &= False

        if ( self.NuclearStainedImageFile is not None ) and ( self.NuclearStainedImageFile != "" ):
            self.NuclearStainedImage = ZStack.ZStack.FromFile(self.NuclearStainedImageFile)
            if ( self.NuclearStainedImage is None ):
                self._LogWriter.Errorln(f"Failed to open nuclear-stained image file [ {self.NuclearStainedImageFile} ]!")
                Validated &= False

        if ( self.RodsStainedImageFile is not None ) and ( self.RodsStainedImageFile != "" ):
            self.RodsStainedImage = ZStack.ZStack.FromFile(self.RodsStainedImageFile)
            if ( self.RodsStainedImage is None ):
                self._LogWriter.Errorln(f"Failed to open image file [ {self.RodsStainedImageFile} ]!")
                Validated &= False

        if ( self.EnableManualROISelection ) and ( self.HeadlessMode ):
            self.EnableManualROISelection = False
            self._LogWriter.Warnln(f"Manual ROI Filtering is not compatible with headless operation... Skipping manual ROI selection and continuing.")

        return (Validated, self.ValidateOnly)

    def Save(self: Configuration, Text: bool, JSON: bool) -> bool:
        """
        Save

        This function saves out the Configuration instance in either or both text or JSON format
        for later review.

        Text:
            Boolean for whether to save a text copy of the Configuration instance.
        JSON:
            Boolean for whether to save a JSON copy of the Configuration instance.

        Return (bool):
            Flag for whether the save operation(s) were successful.
        """

        Success: bool = True

        if ( Text ):
            Success &= self._SaveText()
        if ( JSON ):
            Success &= self._SaveJSON()

        if ( not ( Text or JSON ) ):
            self._LogWriter.Warnln(f"Save() method called without specifying Text or JSON format.")

        return Success

    ### Private Methods
    def _SaveText(self: Configuration) -> bool:
        """
        _SaveText

        This function specifically saves out the configuration in text format.

        Return (bool):
            Boolean for whether the save operation was successful.
        """
        Success: bool = False

        #   ...

        return Success

    def _SaveJSON(self: Configuration) -> bool:
        """
        _SaveJSON

        This function specifically saves out the configuration in JSON format.

        Return (bool):
            Boolean for whether the save operation was successful.
        """
        Success: bool = False

        #   ...

        return Success

class QuantificationResults():
    """
    QuantificationResults

    This class...
    """

    BrightFieldStack: ZStack.ZStack
    BrightFieldMinimumProjection: np.ndarray

    ChipEdgeMask: np.ndarray

    NuclearStainStack: ZStack.ZStack
    NuclearStainStackStainMaximumProjection: np.ndarray

    ExplantBodyMask: np.ndarray
    ExplantBodyCentroid: typing.Tuple[int, int]

    NeuriteStainStack: ZStack.ZStack
    NeuriteStainMaximumProjection: np.ndarray
    FilteredIdentifiedNeurites: ZStack.ZStack

    RodStainStack: ZStack.ZStack
    RodStainMaximumProjection: np.ndarray
    FilteredIdentifiedRods: ZStack.ZStack

    NeuriteDistanceStack: ZStack.ZStack
    NeuriteOrientationsStack: ZStack.ZStack
    RodOrientationsStack: ZStack.ZStack

    NeuriteDistanceVisualizationStack: ZStack.ZStack
    NeuriteDistancePlotStack: ZStack.ZStack
    NeuriteDistancePlotFlattened: Figure

    NeuriteOrientationVisualizationStack: ZStack.ZStack
    NeuriteOrientationPlotStack: ZStack.ZStack
    NeuriteOrientationPlotFlattened: Figure

    RodOrientationVisualizationStack: ZStack.ZStack
    RodOrientationPlotStack: ZStack.ZStack
    RodOrientationPlotFlattened: Figure

    #   Derived and Exported Metrics
    #   ...

    def __init__(self: QuantificationResults, LogWriter: Logger.Logger) -> None:
        """
        Constructor

        This function...

        LogWriter:
            ...

        Return (None):
            ...
        """

        self._LogWriter = LogWriter

        self.BrightFieldStack = ZStack.ZStack(self._LogWriter, "Bright Field")
        self.BrightFieldMinimumProjection = None

        self.ChipEdgeMask = None

        self.NuclearStainStack = ZStack.ZStack(self._LogWriter, "Nuclear Stain")
        self.NuclearStainStackStainMaximumProjection = None

        self.ExplantBodyMask = None
        self.ExplantBodyCentroid = (-1, -1)

        self.NeuriteStainStack = ZStack.ZStack(self._LogWriter, "Neurite Stain")
        self.NeuriteStainMaximumProjection = None
        self.FilteredIdentifiedNeurites = ZStack.ZStack(self._LogWriter, "Identified Neurites")

        self.RodStainStack = ZStack.ZStack(self._LogWriter, "Rod Stain")
        self.RodStainMaximumProjection = None
        self.FilteredIdentifiedRods = ZStack.ZStack(self._LogWriter, "Identified Rods")

        self.NeuriteDistanceStack = ZStack.ZStack(self._LogWriter, "Neurite Distances")
        self.NeuriteDistanceVisualizationStack = ZStack.ZStack(self._LogWriter, "Colour-Annotated Neurite Distances")
        self.NeuriteDistancePlotStack = ZStack.ZStack(self._LogWriter, "Neurite Distances by Layer")
        self.NeuriteDistancePlotFlattened = None

        self.NeuriteOrientationsStack = ZStack.ZStack(self._LogWriter, "Neurite Orientations")
        self.NeuriteOrientationVisualizationStack = ZStack.ZStack(self._LogWriter, "Colour-Annotated Neurite Orientations")
        self.NeuriteOrientationPlotStack = ZStack.ZStack(self._LogWriter, "Neurite Orientations by Layer")
        self.NeuriteOrientationPlotFlattened = None

        self.RodOrientationsStack = ZStack.ZStack(self._LogWriter, "Rod Orientations")
        self.RodOrientationVisualizationStack = ZStack.ZStack(self._LogWriter, "Colour-Annotated Rod Orientations")
        self.RodOrientationPlotStack = ZStack.ZStack(self._LogWriter, "Rod Orientations by Layer")
        self.RodOrientationPlotFlattened = None

        return

    def Quantify(self: QuantificationResults) -> None:
        """
        Quantify

        This function...

        Return (None):
            ...
        """

        #   First, compute all of the relevant projections
        self.BrightFieldMinimumProjection               = self.BrightFieldStack.MinimumIntensityProjection()
        self.NuclearStainStackStainMaximumProjection    = self.NuclearStainStack.MaximumIntensityProjection()
        self.NeuriteStainMaximumProjection              = self.NeuriteStainStack.MaximumIntensityProjection()
        self.RodStainMaximumProjection                  = self.RodStainStack.MaximumIntensityProjection()

        #   Next, compute the neurite distance stack
        self.NeuriteDistanceStack = ComputeNeuriteDistances(self.FilteredIdentifiedNeurites, self.ExplantBodyCentroid)

        #   Next, compute the neurite orientation stack
        self.NeuriteOrientationsStack = ComputeOrientations(self.FilteredIdentifiedNeurites)

        #   Next, compute the rod orientation stack
        self.RodOrientationsStack = ComputeOrientations(self.FilteredIdentifiedRods)

        #   Compute the colour-annotated neurite length stack
        self.NeuriteDistanceVisualizationStack = ColourNeuriteDistances(self.NeuriteStainStack, self.NeuriteDistanceStack, self.ExplantBodyCentroid)

        #   Compute the colour-annotated neurite orientation stack
        self.NeuriteOrientationVisualizationStack = ColourOrientations(self.NeuriteStainStack, self.NeuriteOrientationsStack)

        #   Compute the colour-annotated rod orientation stack
        self.RodOrientationVisualizationStack = ColourOrientations(self.RodStainStack, self.RodOrientationsStack)

        #   Compute the per-layer neurite length plots
        self.NeuriteDistancePlotStack = PlotNeuriteDistances(self.NeuriteDistanceStack, flatten=False)

        #   Compute the per-layer neurite orientation plots
        self.NeuriteOrientationPlotStack = PlotNeuriteOrientations(self.NeuriteOrientationsStack, flatten=False)

        #   Compute the per-layer rod orientation plots
        self.RodOrientationPlotStack = PlotRodOrientations(self.RodOrientationsStack, flatten=False)

        #   Compute the overall neurite length plots
        self.NeuriteDistancePlotStack = PlotNeuriteDistances(self.NeuriteDistanceStack, flatten=True)

        #   Compute the overall neurite orientation plots
        self.NeuriteOrientationPlotStack = PlotNeuriteOrientations(self.NeuriteOrientationsStack, flatten=True)

        #   Compute the overall rod orientation plots
        self.RodOrientationPlotStack = PlotRodOrientations(self.RodOrientationsStack, flatten=True)

        return

    def Save(self: QuantificationResults, Folder: str, DryRun: bool) -> bool:
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

        return True


#   Define the globals to set by the command-line arguments
LogWriter: Logger.Logger = Logger.Logger(Prefix=os.path.basename(sys.argv[0]))
Config: Configuration = Configuration(LogWriter=LogWriter)
Results: QuantificationResults = QuantificationResults(LogWriter=LogWriter)

#   Main
#       This is the main entry point of the script.
def main() -> int:

    ChipExclusionMask: np.ndarray = None
    ExplantBodyExclusionMask: np.ndarray = None

    #   First, let us identify the bounds of the well or chip being used, to generate a mask for excluding all areas where
    #   neurites cannot possibly grow
    if ( Config.BrightFieldImage is not None ):
        LogWriter.Println(f"Processing bright-field image to identify the chip interior region mask.")
        Results.BrightFieldStack = Config.BrightFieldImage.Copy().SetName("Bright Field")
        ChipExclusionMask = ProcessBrightField(Config.BrightFieldImage.MinimumIntensityProjection(), Results)
    else:
        LogWriter.Println(f"No bright-field image is provided, chip exclusion mask cannot be identified and will be skipped...")
        ChipExclusionMask = np.ones_like(Utils.BGRToGreyscale(Config.NeuriteStainedImage.MaximumIntensityProjection()))

    #   Next, assuming that we have a nuclear-stained fluorescent image, attempt to extract a mask
    #   corresponding to the explant body from this to further mask and reduce the neurite search space
    if ( Config.NuclearStainedImage is not None ):
        LogWriter.Println(f"Processing nuclear-stained image to identify the explant body mask.")
        Results.NuclearStainStack = Config.NuclearStainedImage.Copy()
        ExplantBodyExclusionMask = ProcessNuclearStain(Config.NuclearStainedImage.MaximumIntensityProjection(), ChipExclusionMask, Results)
    else:
        if ( Config.BrightFieldImage is not None ):
            LogWriter.Println(f"No nuclear-stained image provided to extract the explant body mask. Attempting to extract this from the bright-field image.")
            ExplantBodyExclusionMask = ExtractBrightFieldExplantMask(Config.BrightFieldImage.MinimumIntensityProjection(), ChipExclusionMask, Results)
        else:
            LogWriter.Println(f"No bright-field image is provided, explany body exclusion mask cannot be identified and will be skipped...")
            ExplantBodyExclusionMask = np.ones_like(Utils.BGRToGreyscale(Config.NeuriteStainedImage.MaximumIntensityProjection()))

    if ( ChipExclusionMask is None ):
        return STATUS_NO_CHIP_MASK

    if ( ExplantBodyExclusionMask is None ):
        return STATUS_NO_EXPLANT_MASK

    #   Combine the exclusion masks into one mask for all future uses
    NeuriteExclusionMask: np.ndarray = ChipExclusionMask * ExplantBodyExclusionMask

    #   Now, with the regions known not to correspond to neurites identified, work throughe each layer of the neurite Z-stack
    #   and extract out the set of neurite pixels from each layer.
    Results.NeuriteStainStack = Config.NeuriteStainedImage.Copy()
    LogWriter.Println(f"Working with neurite-stained image to identify neurites.")
    for LayerIndex, NeuriteLayer in enumerate(Config.NeuriteStainedImage.Layers()):

        LogWriter.Println(f"Processing layer [ {LayerIndex+1}/{Config.NeuriteStainedImage.LayerCount()} ]...")
        ProcessNeuriteStain(NeuriteLayer, NeuriteExclusionMask, Results)

    #   The final processing of the stacks involves working with the rods image to identify the rod orientations
    if ( Config.RodsStainedImage is not None ):
        LogWriter.Println(f"Working with rod-stained image to identify rod orientations.")
        for LayerIndex, RodLayer in enumerate(Config.RodsStainedImage.Layers()):

            LogWriter.Println(f"Processing layer [ {LayerIndex+1}/{Config.RodsStainedImage.LayerCount()} ]...")
            ProcessRodStain(RodLayer, NeuriteExclusionMask, Results)

    #   With all of the information extracted from the images acquired, compute the resulting metrics
    #   and report them back out to the user.
    Results.Quantify()

    #   Save out the configuration state for possible later review.
    Config.Save(Text=True, JSON=True)
    Results.Save(Folder=Config.OutputDirectory, DryRun=False)

    return 0

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

def ContinuePromptToBoolean(Key: int) -> bool:
    """
    ContinuePromptToBoolean

    Key:
        ...

    Return (bool):
        ...
    """

    return Key in [ord(x) for x in 'yY']

def ProcessBrightField(Image: np.ndarray, Results: QuantificationResults) -> np.ndarray:
    """
    ProcessBrightField

    This function...

    Return ():
        ...
    """

    #   First, normalize the image to a well-defined format. Bright background with dark foreground,
    #   linearly rescaled to the full 8-bit range.
    NormalizedImage = Utils.ConvertTo8Bit(Image)

    #   If the median pixel is darker than the mean pixel, that means most of the image is dark.
    #   We interpret this to mean a dark background
    if ( np.median(NormalizedImage) < np.mean(NormalizedImage) ):
        NormalizedImage = -NormalizedImage

    #   Binarize the image, to segment the foreground from background
    BinarizedImage, ThresholdLevel = BinarizeBrightField(NormalizedImage.copy())

    #   With the binarized image, try to determine the bounds of the chip iterior
    #   in order to prepare an exclusion mask for everywhere else.
    ChipExclusionMask = ExtractBrightFieldChipExclusionMask(BinarizedImage, Image)

    #   Cache the results generated from this phase for later storage and quantification
    Results.BrightFieldMinimumProjection = NormalizedImage
    Results.ChipEdgeMask = ChipExclusionMask

    return ChipExclusionMask

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

    ThresholdLevel, BinarizedImage = cv2.threshold(Image, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)

    return BinarizedImage, ThresholdLevel

def ExtractBrightFieldChipExclusionMask(Image: np.ndarray, Background: np.ndarray) -> np.ndarray:
    """
    ExtractBrightFieldChipExclusionMask

    This function...

    Image:
        ...

    Return (np.ndarray):
        ...
    """

    MinimumComponentAreaThreshold: int = 1000
    MinimumContourAreaThreshold: int = 500

    #   Pre-initialize the exclusion mask to begin by including all pixels of the image, so that
    #   we can affirmatively remove known invalid locations
    ExclusionMask: np.ndarray = np.ones_like(Image)

    #   First, segment the image by component
    ComponentCount, Labels, Stats, Centroids = cv2.connectedComponentsWithStats(Image, connectivity=8)

    #   Next, sort the components by size, with the largest components at the beginning. We know
    #   that the main regions of the image we are interested in will be large, so this avoids us
    #   wasting time on little speckle regions
    SortedComponents: typing.List[typing.Tuple[int, int]] = sorted([
        (x, Stats[x, cv2.CC_STAT_AREA]) for x in range(1, ComponentCount) if Stats[x, cv2.CC_STAT_AREA] >= MinimumComponentAreaThreshold
    ], key=lambda x: x[1], reverse=True)

    #   Iterate over the sorted components, checking whether it corresponds to the interior
    #   of the chip.
    for ComponentIndex, (ComponentID, ComponentArea) in enumerate(SortedComponents):
        ComponentMask: np.ndarray = (Labels == ComponentID).astype(np.uint8)
        if ( ContinuePromptToBoolean(Utils.DisplayImage(f"Include Component: {ComponentID} ({ComponentIndex+1}/{len(SortedComponents)}) as part of chip exterior (White Region)? (y/N)", Utils.ConvertTo8Bit(ComponentMask), 0, True, True)) ):
            ExclusionMask[Labels == ComponentID] = 0

    #   ...
    Contours, _ = cv2.findContours(ExclusionMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    Contours = list(filter(lambda x: cv2.contourArea(x) > MinimumContourAreaThreshold, sorted(Contours, key=lambda x: cv2.contourArea(x), reverse=True)))
    for ContourIndex, Contour in enumerate(Contours):
        ContourMask = cv2.drawContours(np.zeros_like(ExclusionMask), [Contour], 0, 1, -1)
        if ( ContinuePromptToBoolean(Utils.DisplayImage(f"Include Contour: ({ContourIndex+1}/{len(Contours)}) - {cv2.contourArea(Contour)} as part of chip exterior (White Region)? (y/N)", Utils.ConvertTo8Bit(ContourMask), 0, True, True)) ):
            ExclusionMask[ContourMask != 0] = 0

    return ExclusionMask

def ProcessNuclearStain(Image: np.ndarray, ChipExclusionMask: np.ndarray, Results: QuantificationResults) -> np.ndarray:
    """
    ProcessNuclearStain

    This function...

    Return ():
        ...
    """

    LogWriter.Errorln(f"Nuclear Stained image processing has not yet been implemented!")

    #   Cache the results into the outgoing Results value
    Results.NuclearStainStackStainMaximumProjection = Image

    return None

def ExtractBrightFieldExplantMask(Image: np.ndarray, ChipExclusionMask: np.ndarray, Results: QuantificationResults) -> np.ndarray:
    """
    ExtractBrightFieldExplantMask

    This function...

    Return ():
        ...
    """

    if ( ContinuePromptToBoolean(Utils.DisplayImage(f"Has the explant body already been masked out? (y/N)", ApplyImageMask(Image, ChipExclusionMask), 0, True, True))):
        return np.ones_like(Image)

    MinimumComponentAreaThreshold: int = 1000
    MinimumContourAreaThreshold: int = 500

    #   First, normalize the image to a well-defined format. Bright background with dark foreground,
    #   linearly rescaled to the full 8-bit range.
    NormalizedImage = Utils.ConvertTo8Bit(Image)

    #   If the median pixel is darker than the mean pixel, that means most of the image is dark.
    #   We interpret this to mean a dark background
    if ( np.median(NormalizedImage) < np.mean(NormalizedImage) ):
        NormalizedImage = -NormalizedImage

    MaskedImage, InclusionMask = ApplyManualROI(ApplyImageMask(NormalizedImage, ChipExclusionMask), ApplyImageMask(Image, ChipExclusionMask), InclusionMask=True)

    #   Binarize the image, to segment the foreground from background
    BinarizedImage, ThresholdLevel = BinarizeBrightField(NormalizedImage)

    #   Apply the chip exclusion mask to remove contours and components we know can't be the explant body.
    BinarizedImage = ApplyImageMask(BinarizedImage, ChipExclusionMask * InclusionMask)

    ExclusionMask: np.ndarray = np.ones_like(Image)

    #   Segment the image by components
    ComponentCount, Labels, Stats, Centroids = cv2.connectedComponentsWithStats(BinarizedImage, connectivity=8)

    #   Next, sort the components by size, with the largest components at the beginning. We know
    #   that the main regions of the image we are interested in will be large, so this avoids us
    #   wasting time on little speckle regions
    SortedComponents: typing.List[typing.Tuple[int, int]] = sorted([
        (x, Stats[x, cv2.CC_STAT_AREA]) for x in range(1, ComponentCount) if Stats[x, cv2.CC_STAT_AREA] >= MinimumComponentAreaThreshold
    ], key=lambda x: x[1], reverse=True)

    #   Iterate over the sorted components, checking whether it corresponds to the interior
    #   of the chip.
    for ComponentIndex, (ComponentID, ComponentArea) in enumerate(SortedComponents):
        ComponentMask: np.ndarray = (Labels == ComponentID).astype(np.uint8)
        if ( ContinuePromptToBoolean(Utils.DisplayImage(f"Include Component: {ComponentID} ({ComponentIndex+1}/{len(SortedComponents)})? (y/N)", ApplyImageMask(BinarizedImage, ComponentMask), 0, True, True)) ):
            ExclusionMask[Labels == ComponentID] = 0

    #   ...
    Contours, _ = cv2.findContours(ExclusionMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    Contours = list(filter(lambda x: cv2.contourArea(x) > MinimumContourAreaThreshold, sorted(Contours, key=lambda x: cv2.contourArea(x), reverse=True)))
    for ContourIndex, Contour in enumerate(Contours):
        ContourArea: int = cv2.contourArea(Contour)
        if ( ContourArea > 0.75 * np.prod(ExclusionMask.shape)):
            continue
        ContourMask = cv2.drawContours(np.zeros_like(ExclusionMask), [Contour], 0, 1, -1)
        if ( ContinuePromptToBoolean(Utils.DisplayImage(f"Include Contour: ({ContourIndex+1}/{len(Contours)}) - {cv2.contourArea(Contour)}? (y/N)", ApplyImageMask(BinarizedImage, ContourMask), 0, True, True)) ):
            ExclusionMask[ContourMask != 0] = 0

    #   Cache the results...
    Results.ExplantBodyMask = ExclusionMask
    Results.ExplantBodyCentroid = DetermineExplantCentroid(~ExclusionMask)
    return ExclusionMask

def DetermineExplantCentroid(ExplantBodyMask: np.ndarray) -> typing.Tuple[int, int]:
    """
    DetermineExplantCentroid

    This function...

    ExplantBodyMask:
        ...

    Return (Tuple[int, int]):
        ...
    """

    ComponentCount, Labels, Stats, Centroids = cv2.connectedComponentsWithStats(Utils.ConvertTo8Bit(ExplantBodyMask), connectivity=8)
    return tuple((int(x) for x in Centroids[1]))

def ProcessNeuriteStain(Image: np.ndarray, ExclusionMask: np.ndarray, Results: QuantificationResults) -> None:
    """
    ProcessNeuriteStain

    This function...

    Return ():
        ...
    """

    AdaptiveKernelSize: int = 45
    AdaptiveOffset: int = -5
    MaskExpansionSize: int = int(AdaptiveKernelSize / 4)
    SpeckleComponentAreaThreshold: int = 25
    NeuriteAspectRatioThreshold: float = 1.5
    NeuriteInfillFractionThreshold: float = 0.75 * (np.pi / 4)

    #   First, convert the image to a full-range 8-bit image and assert that it is greyscale.
    Image: np.ndarray = Utils.ConvertTo8Bit(Utils.BGRToGreyscale(Image))

    #   Apply an adaptive local threshold over the image to further select out the neurite pixels
    BinarizedImage: np.ndarray = BinarizeNeuriteStain(Image, AdaptiveKernelSize, AdaptiveOffset)

    BinarizedImage = ApplyImageMask(BinarizedImage, cv2.erode(ExclusionMask, kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(MaskExpansionSize, MaskExpansionSize))))

    #   Finally, separate out all of the components of the image and filter them to remove any which do not
    #   satisfy the expectations of neurites
    FilteredNeuriteComponents = FilterNeuriteComponents(BinarizedImage, SpeckleComponentAreaThreshold, NeuriteAspectRatioThreshold, NeuriteInfillFractionThreshold)

    Results.FilteredIdentifiedNeurites.Append(Utils.ConvertTo8Bit(FilteredNeuriteComponents))

    return

def BinarizeNeuriteStain(Image: np.ndarray, KernelSize: int, ThresholdValue: int) -> np.ndarray:
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

            AspectRatio: float = np.max([Height, Width]) / np.min([Height, Width])
            BoundingArea: int = Height * Width
            FilledFraction: float = ComponentArea / BoundingArea

            if ( AspectRatio < NeuriteAspectRatioThreshold ) and ( FilledFraction > NeuriteInfillFraction ):
                continue

            FilteredComponents[Labels == ComponentID] = 1

    return FilteredComponents

def ProcessRodStain(Image: np.ndarray, ExclusionMask: np.ndarray, Results: QuantificationResults) -> None:
    """
    ProcessRodStain

    This function...

    Return ():
        ...
    """

    LogWriter.Errorln(f"Rod Stained image processing has not yet been implemented!")

    return

def EstimateCentroid(ThresholdedImage: np.ndarray, CorrelationThreshold: float = 0.975, KernelStepSize: int = 10, InitialKernelSize: int = 11, CentroidJitterThreshold: int = 1) -> typing.Tuple[int, int]:
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
        Centroid_X = int(ImageMoments["m10"] / ImageMoments["m00"])
        Centroid_Y = int(ImageMoments["m01"] / ImageMoments["m00"])

        # Utils.DisplayImage(f"DRG Centroid Correlation Map: [ K={KernelSize} ]", cv2.circle(Utils.GreyscaleToBGR(Utils.ConvertTo8Bit(CorrelationMap.copy())), (Centroid_X, Centroid_Y), 10, (0, 0, 255), -1), DEBUG_DISPLAY_TIMEOUT, True, DEBUG_DISPLAY_ENABLED and ( not Config.HeadlessMode ))

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

def ApplyImageMask(Image: np.ndarray, Mask: np.ndarray) -> np.ndarray:
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

def ApplyManualROI(ImageToFilter: np.ndarray, Background: np.ndarray, InclusionMask: bool = False) -> typing.Tuple[np.ndarray, np.ndarray]:
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
    InclusionMask:
        ...

    Return (Tuple):
        [0] - np.ndarray:
            The resulting masked image, with the user-selected regions removed or extracted.
        [1] - np.ndarray:
            The mask generated by the user, for visualization or potential later re-use.
    """

    #   Pre-fill the exclusion mask with all ones, to include all pixels.
    if ( InclusionMask ):
        PolygonSelectionMask: np.ndarray = np.zeros_like(Utils.BGRToGreyscale(ImageToFilter))
    else:
        PolygonSelectionMask: np.ndarray = np.ones_like(Utils.BGRToGreyscale(ImageToFilter))

    if ( Config.EnableManualROISelection ):

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
        #   TODO: Update the candidate image between each mask application?
        AxIm = Ax.imshow(Result, origin='upper')

        #   Define the callback to run if and when the user draws a closed polygon.
        def UpdateSelectionMask(Vertices: np.ndarray) -> None:

            #   When a closed polygon is drawn, convert it into a set of (X,Y) coordinate points,
            #   and draw this as a contour over the image, infilling with 0's to mask out the
            #   interior region of the contour.
            nonlocal PolygonSelectionMask
            nonlocal Ax

            Vertices = np.array([
                (int(X), int(Y)) for (X, Y) in Vertices
            ])
            if ( InclusionMask ):
                PolygonSelectionMask = cv2.drawContours(PolygonSelectionMask, [Vertices], 0, 1, -1)
            else:
                PolygonSelectionMask = cv2.drawContours(PolygonSelectionMask, [Vertices], 0, 0, -1)

            #   Prepare an overlaid image with the background being shown as-is, and the foreground neurite pixels
            #   only in the green channel.
            Base: np.ndarray = Utils.GammaCorrection(Utils.GreyscaleToBGR(Background).astype(np.float64), Minimum=0, Maximum=1.0)
            Foreground: np.ndarray = Utils.GammaCorrection(Utils.GreyscaleToBGR(PolygonSelectionMask * ImageToFilter).astype(np.float64), Minimum=0, Maximum=1.0)
            Foreground[:,:,0] = 0
            Foreground[:,:,2] = 0
            Alpha: np.ndarray = Foreground.copy()

            Result = (Foreground * Alpha) + (Base * (1.0 - Alpha))

            Ax.clear()
            Ax.imshow(Utils.ConvertTo8Bit(Result), origin='upper')

        P = PolygonSelector(Ax, onselect=UpdateSelectionMask, props=dict(color='r', linestyle='-', linewidth=2))
        plt.show(block=True)

    PolygonMaskedImage: np.ndarray = ApplyImageMask(Utils.BGRToGreyscale(ImageToFilter), PolygonSelectionMask)

    return PolygonMaskedImage, PolygonSelectionMask

def ComputeNeuriteDistances(IdentifiedPixels: ZStack.ZStack, Origin: typing.Tuple[int, int]) -> ZStack.ZStack:
    """
    ComputeNeuriteDistances

    This function...

    Return ():
        ...
    """

    Distances: ZStack.ZStack = ZStack.ZStack(LogWriter, "Neurite Distances")
    for Layer in IdentifiedPixels.Layers():
        NeuriteCoordinates: np.ndarray = np.argwhere(Layer != 0)
        LayerDistances: np.ndarray = np.zeros_like(Layer)
        LayerDistances[Layer == 0] = np.NaN
        LayerDistances[Layer != 0] = np.hypot(NeuriteCoordinates[:,0] - Origin[1], NeuriteCoordinates[:,1] - Origin[0])
        Distances.Append(LayerDistances)

    return

def ComputeOrientations(IdentifiedPixels: ZStack.ZStack) -> ZStack.ZStack:
    """
    ComputeOrientations

    This function...

    Return ():
        ...
    """

    return

def ColourNeuriteDistances(BackgroundImage: ZStack.ZStack, Distances: ZStack.ZStack, Origin: typing.Tuple[int, int]) -> ZStack.ZStack:
    """
    ColourNeuriteDistances

    This function...

    Return ():
        ...
    """

    MaximumDistance: float = np.nanmax(Distances)
    ColouredNeuriteDistances: ZStack.ZStack = ZStack.ZStack(LogWriter, "Colour-Annotated Neurite Distances")

    for Background, Distance in zip(BackgroundImage, Distances):

        pass

    return ColouredNeuriteDistances

def ColourOrientations(BackgroundImage: ZStack.ZStack, Orientations: ZStack.ZStack) -> ZStack.ZStack:
    """
    ColourOrientations

    This function...

    Return ():
        ...
    """

    return

def PlotNeuriteDistances(DistancesByLayer: ZStack.ZStack, flatten: bool = False) -> ZStack.ZStack | Figure:
    """
    PlotNeuriteDistances

    This function...

    Return ():
        ...
    """

    return

def PlotNeuriteOrientations(OrientationsByLayer: ZStack.ZStack, flatten: bool = False) -> ZStack.ZStack | Figure:
    """
    PlotNeuriteOrientations(self.NeuriteOrientationsStack, flatten=False)

    This function...

    Return ():
        ...
    """

    return

def PlotRodOrientations(OrientationsByLayer: ZStack.ZStack, flatten: bool = False) -> ZStack.ZStack | Figure:
    """
    PlotRodOrientations(self.RodOrientationsStack, flatten=False)

    This function...

    Return ():
        ...
    """

    return

def HandleArguments() -> bool:
    """
    HandleArguments

    This function sets up, parses, extracts, and validates the command-line arguments for the script.

    Return (bool):
        A boolean indicating whether or not the script should continue executing after this function returns.
    """

    #   Prepare the argument parser
    Flags: argparse.ArgumentParser = argparse.ArgumentParser(description="A script for quantifying the the growth length and directionality of the neurite extensions from cortical explants, and the micro-rods potentially used to align them.")

    #   Add in the command-line flags to accept
    #   Add in the flags for the images to work with.
    Flags.add_argument("--neurite-stain", dest="NeuriteStain", metavar="file-path", type=str, required=True, help="The neurite-stained image of the cortical explant, showing the neurite growth throughout the chip.")
    Flags.add_argument("--bright-field", dest="BrightField", metavar="file-path", type=str, required=False, default=None, help="The bright-field/transmitted light image of the cortical explant, showing the bounds of the well or chip housing the explant.")
    Flags.add_argument("--nuclear-stain", dest="NuclearStain", metavar="file-path", type=str, required=False, default=None, help="The nuclear-stained image of the cortical explant, showing the explant body to allow masking and removal of the region.")
    Flags.add_argument("--rods-stain", dest="RodsStain", metavar="file-path", type=str, required=False, default=None, help="The image showing the micro-scale magnetic rods used to align and orient the neurite growth.")

    Flags.add_argument("--manual-roi", dest="ManualROI", action="store_true", required=False, default=True, help="...")

    #   Add in the flag specifying where the results generated by this script should be written out to.
    Flags.add_argument("--results-directory", dest="OutputDirectory", metavar="folder-path", type=str, required=False, default=os.path.dirname(sys.argv[0]), help="The path to the base folder into which results will be written on a per-execution basis.")
    #   ...

    #   Add in flags for manipulating the logging functionality of the script.
    Flags.add_argument("--log-file", dest="LogFile", metavar="file-path", type=str, required=False, default="-", help="File path to the file to write all log messages of this program to.")
    Flags.add_argument("--quiet",    dest="Quiet",   action="store_true",  required=False, default=False, help="Enable quiet mode, disabling logging of eveything except fatal errors.")

    #   Finally, add in scripts for modifying the basic environment or end-state of the script.
    Flags.add_argument("--dry-run",  dest="DryRun",   action="store_true", required=False, default=False, help="Enable dry-run mode, where no file-system alterations or heavy computations are performed.")
    Flags.add_argument("--validate", dest="Validate", action="store_true", required=False, default=False, help="Only validate the command-line arguments, do not proceed with the remainder of the program.")
    Flags.add_argument("--headless", dest="Headless", action="store_true", required=False, default=False, help="Run in 'headless' mode, where nothing is displayed to the screen.")
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
            sys.exit(main())
        except Exception as e:
            LogWriter.Fatalln(f"Exception raised in main(): [ {e} ]\n\n{''.join(traceback.format_exception(e, value=e, tb=e.__traceback__))}")
    else:
        sys.exit(-1)
