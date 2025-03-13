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
from scipy.stats import circmean, circstd

#   Import the desired locally written modules
from MTG_Common import Logger
from MTG_Common import Utils
from MTG_Common import ZStack
from Alignment_Analysis import PrepareEllipticalKernel, ApplyEllipticalConvolution, CreateOrientationVisualization, ComputeAlignmentMetric, AngleTracker

DEBUG_DISPLAY_ENABLED: bool = True
DEBUG_DISPLAY_TIMEOUT: float = 0

#   Add a sequence number to the images as generated and exported from this script.
ImageSequenceNumber: int = 1

#   Add in top-level return codes for the status of processing the script.
#   These signal to the environment whether or not everything processed
#   correctly, or if the script encountered an error during processing.
STATUS_SUCCESS: int                         = 0

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

        if ( self.NeuriteStainedImageFile is not None ) and ( self.NeuriteStainedImageFile != "" ):
            self._LogWriter.Println(f"Attempting to open neurite-stained image file [ {self.NeuriteStainedImageFile} ]...")
            self.NeuriteStainedImage = ZStack.ZStack.FromFile(self.NeuriteStainedImageFile).SetName("Neurite Stain Stack")
            if ( self.NeuriteStainedImage is None ):
                self._LogWriter.Errorln(f"Failed to open neurite-stained image file [ {self.NeuriteStainedImageFile} ]!")
                Validated &= False

        if ( self.BrightFieldImageFile is not None ) and ( self.BrightFieldImageFile != "" ):
            self._LogWriter.Println(f"Attempting to open bright-field image file [ {self.BrightFieldImageFile} ]..")
            self.BrightFieldImage = ZStack.ZStack.FromFile(self.BrightFieldImageFile)
            if ( self.BrightFieldImage is None ):
                self._LogWriter.Errorln(f"Failed to open bright-field image file [ {self.BrightFieldImageFile} ]!")
                Validated &= False

        if ( self.NuclearStainedImageFile is not None ) and ( self.NuclearStainedImageFile != "" ):
            self._LogWriter.Println(f"Attempting to open nuclear-stained image file [ {self.NuclearStainedImageFile} ]...")
            self.NuclearStainedImage = ZStack.ZStack.FromFile(self.NuclearStainedImageFile)
            if ( self.NuclearStainedImage is None ):
                self._LogWriter.Errorln(f"Failed to open nuclear-stained image file [ {self.NuclearStainedImageFile} ]!")
                Validated &= False

        if ( self.RodsStainedImageFile is not None ) and ( self.RodsStainedImageFile != "" ):
            self._LogWriter.Println(f"Attempting to open rod-stained image file [ {self.RodsStainedImageFile} ]...")
            self.RodsStainedImage = ZStack.ZStack.FromFile(self.RodsStainedImageFile)
            if ( self.RodsStainedImage is None ):
                self._LogWriter.Errorln(f"Failed to open rod-stained image file [ {self.RodsStainedImageFile} ]!")
                Validated &= False

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

    def _ToString(self: Configuration) -> str:
        """
        _ToString

        This function...

        return (str):
            ...
        """

        return "\n".join([
            f""
        ])

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

    def _ToJSON(self: Configuration) -> str:
        """
        _ToJSON

        This function..

        Return (str):
            ...
        """

        return ""

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

    NeuriteDistances: typing.Sequence[np.ndarray]
    NeuriteOrientations: ZStack.ZStack
    RodOrientations: ZStack.ZStack

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
    MedianNeuriteLengths: typing.Sequence[float]

    MeanNeuriteOrientations: typing.Sequence[float]
    NeuriteOrientationStDevs: typing.Sequence[float]
    NeuriteAlignmentMetrics: typing.Sequence[float]

    MeanRodOrientations: typing.Sequence[float]
    RodOrientationStDevs: typing.Sequence[float]
    RodAlignmentMetrics: typing.Sequence[float]
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

        self.BrightFieldStack = None
        self.BrightFieldMinimumProjection = None

        self.ChipEdgeMask = None

        self.NuclearStainStack = None
        self.NuclearStainStackStainMaximumProjection = None

        self.ExplantBodyMask = None
        self.ExplantBodyCentroid = (-1, -1)

        self.NeuriteStainStack = None
        self.NeuriteStainMaximumProjection = None
        self.FilteredIdentifiedNeurites = ZStack.ZStack(self._LogWriter, "Identified Neurites")

        self.RodStainStack = None
        self.RodStainMaximumProjection = None
        self.FilteredIdentifiedRods = ZStack.ZStack(self._LogWriter, "Identified Rods")

        self.NeuriteDistances = []
        self.NeuriteDistanceVisualizationStack = ZStack.ZStack(self._LogWriter, "Colour-Annotated Neurite Distances")
        self.NeuriteDistancePlotStack = ZStack.ZStack(self._LogWriter, "Neurite Distances by Layer")
        self.NeuriteDistancePlotFlattened = None

        self.NeuriteOrientations = ZStack.ZStack(self._LogWriter, "Neurite Orientations")
        self.NeuriteOrientationVisualizationStack = ZStack.ZStack(self._LogWriter, "Colour-Annotated Neurite Orientations")
        self.NeuriteOrientationPlotStack = ZStack.ZStack(self._LogWriter, "Neurite Orientations by Layer")
        self.NeuriteOrientationPlotFlattened = None

        self.RodOrientations = ZStack.ZStack(self._LogWriter, "Rod Orientations")
        self.RodOrientationVisualizationStack = ZStack.ZStack(self._LogWriter, "Colour-Annotated Rod Orientations")
        self.RodOrientationPlotStack = ZStack.ZStack(self._LogWriter, "Rod Orientations by Layer")
        self.RodOrientationPlotFlattened = None

        #   ---

        self.MedianNeuriteLengths = []

        self.MeanNeuriteOrientations = []
        self.NeuriteOrientationStDevs = []
        self.NeuriteAlignmentMetrics = []

        self.MeanRodOrientations = []
        self.RodOrientationStDevs = []
        self.RodAlignmentMetrics = []

        return

    def Quantify(self: QuantificationResults) -> None:
        """
        Quantify

        This function...

        Return (None):
            ...
        """

        self._QuantifyBrightField()

        self._QuantifyNuclearStain()

        self._QuantifyNeuriteStain()

        self._QuantifyRodStain()

        return

    def _QuantifyBrightField(self: QuantificationResults) -> None:
        """
        _QuantifyBrightField

        This function...

        Return (None):
            ...
        """

        #   If we have a bright-field image, extract all of the information from it
        if ( self.BrightFieldStack is not None ):
            self.BrightFieldMinimumProjection = self.BrightFieldStack.MinimumIntensityProjection()

        return

    def _QuantifyNuclearStain(self: QuantificationResults) -> None:
        """
        _QuantifyNuclearStain

        This function...

        Return (None):
            ...
        """

        #   If we have a nuclear-stained stack, extract what we can from it
        if ( self.NuclearStainStack is not None ):
            self.NuclearStainStackStainMaximumProjection = self.NuclearStainStack.MaximumIntensityProjection()

        return

    def _QuantifyNeuriteStain(self: QuantificationResults) -> None:
        """
        _QuantifyNeuriteStain

        This function...

        Return (None):
            ...
        """

        #   If we have the neurite-stained stack, extract what we can from it.
        if ( self.NeuriteStainStack is not None ):
            self.NeuriteStainMaximumProjection = self.NeuriteStainStack.MaximumIntensityProjection()

            self.NeuriteDistances = ComputeNeuriteDistances(self.FilteredIdentifiedNeurites, self.ExplantBodyCentroid)
            self.NeuriteDistanceVisualizationStack = ColourNeuriteDistances(self.NeuriteStainStack, self.FilteredIdentifiedNeurites, self.ExplantBodyCentroid, max([np.max(x) for x in self.NeuriteDistances])).SetName(f"Neurite Distance Visualization")
            self.NeuriteDistancePlotStack = PlotNeuriteDistances(self.NeuriteDistances, flatten=False).SetName("Neurite Distance Plots")
            self.NeuriteDistancePlotFlattened = PlotNeuriteDistances(self.NeuriteDistances, flatten=True)

            self.NeuriteOrientations = ComputeOrientations(self.FilteredIdentifiedNeurites).SetName("Neurite Orientations")
            self.NeuriteOrientationVisualizationStack = ColourOrientations(self.NeuriteStainStack, self.FilteredIdentifiedNeurites, self.NeuriteOrientations).SetName("Neurite Orientation Visualization")
            self.NeuriteOrientationPlotStack = PlotNeuriteOrientations(self.NeuriteOrientations, flatten=False).SetName("Neurite Orientation Plots")
            self.NeuriteOrientationPlotFlattened = PlotNeuriteOrientations(self.NeuriteOrientations, flatten=True)

            #   Once all of the stacks and immediate results have been quantified, then compute the derived metrics and outputs
            self.MedianNeuriteLengths = [np.median(x) for x in self.NeuriteDistances]
            self.MeanNeuriteOrientations = [circmean(x[x != 255].flatten(), high=180, low=0) for x in self.NeuriteOrientations.Layers()]
            self.NeuriteOrientationStDevs = [circstd(x[x != 255].flatten(), high=180, low=0) for x in self.NeuriteOrientations.Layers()]
            AlignmentFractions: typing.List[float] = [len(x[abs(x[x != 255] - Mean) < StDev]) / len(x[x != 255]) for (x, Mean, StDev) in zip(self.NeuriteOrientations.Layers(), self.MeanNeuriteOrientations, self.NeuriteOrientationStDevs)]
            self.NeuriteAlignmentMetrics = [Fraction / StDev for (Fraction, StDev) in zip(AlignmentFractions, self.NeuriteOrientationStDevs)]

        return

    def _QuantifyRodStain(self: QuantificationResults) -> None:
        """
        _QuantifyRodStain

        This function...

        Return (None):
            ...
        """

        #   If we have the rod-stained stack, extract what we can from it.
        if ( self.RodStainStack is not None ):
            self.RodStainMaximumProjection = self.RodStainStack.MaximumIntensityProjection()

            self.RodOrientations = ComputeOrientations(self.FilteredIdentifiedRods).SetName("Rod Orientations")
            self.RodOrientationVisualizationStack = ColourOrientations(self.RodStainStack, self.FilteredIdentifiedRods, self.RodOrientations).SetName("Rod Orientation Visualization")
            self.RodOrientationPlotStack = PlotRodOrientations(self.RodOrientations, flatten=False).SetName("Rod Orientation Plots")
            self.RodOrientationPlotFlattened = PlotRodOrientations(self.RodOrientations, flatten=True)

            #   Once all of the stacks and immediate results have been quantified, then compute the derived metrics and outputs
            self.MeanRodOrientations = [circmean(x[x != 255].flatten(), high=180, low=0) for x in self.RodOrientations.Layers()]
            self.RodOrientationStDevs = [circstd(x[x != 255].flatten(), high=180, low=0) for x in self.RodOrientations.Layers()]
            # AlignmentFractions: typing.List[float] = [len(x[abs(x[x != 255] - Mean) < StDev]) / len(x[x != 255]) for (x, Mean, StDev) in zip(self.RodOrientations.Layers(), self.MeanRodOrientations, self.RodOrientationStDevs)]
            # self.NeuriteAlignmentMetrics = [Fraction / StDev for (Fraction, StDev) in zip(AlignmentFractions, self.RodOrientationStDevs)]

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

        if ( self.BrightFieldStack is not None ):
            self.BrightFieldStack.SaveTIFF(Folder)
            if ( self.BrightFieldMinimumProjection is not None ):
                Utils.WriteImage(Utils.ConvertTo8Bit(self.BrightFieldMinimumProjection), os.path.join(Folder, f"Bright Field Minimum Intensity Projection.tiff"))

        if ( self.ChipEdgeMask is not None ):
            Utils.WriteImage(Utils.ConvertTo8Bit(self.ChipEdgeMask), os.path.join(Folder, f"Chip Edges Mask.tiff"))

        if ( self.NuclearStainStack is not None ):
            self.NuclearStainStack.SaveTIFF(Folder)
            if ( self.NuclearStainStackStainMaximumProjection is not None ):
                Utils.WriteImage(Utils.ConvertTo8Bit(self.NuclearStainStackStainMaximumProjection), os.path.join(Folder, f"Nuclear Stain Maximum Intensity Projection.tiff"))

        if ( self.ExplantBodyMask is not None ):
            Utils.WriteImage(Utils.ConvertTo8Bit(self.ExplantBodyMask), os.path.join(Folder, f"Explant Body Mask.tiff"))

        if ( self.NeuriteStainStack is not None ):
            self.NeuriteStainStack.SaveTIFF(Folder)
            if ( self.NeuriteStainMaximumProjection is not None ):
                Utils.WriteImage(Utils.ConvertTo8Bit(self.NeuriteStainMaximumProjection), os.path.join(Folder, f"Neurite Stain Maximum Intensity Projection.tiff"))
            self.FilteredIdentifiedNeurites.SaveTIFF(Folder)

            self.NeuriteDistanceVisualizationStack.SaveTIFF(Folder)
            Utils.WriteImage(Utils.ConvertTo8Bit(self.NeuriteDistanceVisualizationStack.MaximumIntensityProjection()), os.path.join(Folder, f"Neurite Distance Visualization - Flattened.tiff"))

            self.NeuriteDistancePlotStack.SaveTIFF(Folder)
            if ( self.NeuriteDistancePlotFlattened is not None ):
                Utils.WriteImage(Utils.ConvertTo8Bit(Utils.FigureToImage(self.NeuriteDistancePlotFlattened)), os.path.join(Folder, f"Flattened Neurite Distances.tiff"))

            self.NeuriteOrientationVisualizationStack.SaveTIFF(Folder)
            Utils.WriteImage(Utils.ConvertTo8Bit(self.NeuriteOrientationVisualizationStack.MaximumIntensityProjection()), os.path.join(Folder, f"Neurite Orientation Visualization - Flattened.tiff"))

            self.NeuriteOrientationPlotStack.SaveTIFF(Folder)
            if ( self.NeuriteOrientationPlotFlattened is not None ):
                Utils.WriteImage(Utils.ConvertTo8Bit(Utils.FigureToImage(self.NeuriteOrientationPlotFlattened)), os.path.join(Folder, f"Flattened Neurite Orientations.tiff"))

        if ( self.RodStainStack is not None ):
            self.RodStainStack.SaveTIFF(Folder)
            if ( self.RodStainMaximumProjection is not None ):
                Utils.WriteImage(Utils.ConvertTo8Bit(self.RodStainMaximumProjection), os.path.join(Folder, f"Rod Stain Maximum Intensity Projection.tiff"))
            self.FilteredIdentifiedRods.SaveTIFF(Folder)

            self.RodOrientationVisualizationStack.SaveTIFF(Folder)
            Utils.WriteImage(Utils.ConvertTo8Bit(self.RodOrientationVisualizationStack.MaximumIntensityProjection()), os.path.join(Folder, f"Rod Orientation Visualization - Flattened.tiff"))

            self.RodOrientationPlotStack.SaveTIFF(Folder)
            if ( self.RodOrientationPlotFlattened is not None ):
                Utils.WriteImage(Utils.ConvertTo8Bit(Utils.FigureToImage(self.RodOrientationPlotFlattened)), os.path.join(Folder, f"Flattened Rod Orientations.tiff"))

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
        Results.BrightFieldStack = Config.BrightFieldImage.Copy().SetName("Bright Field Stack")
        ChipExclusionMask, ExplantBodyExclusionMask, ExplantCentroid = ProcessBrightField(Config.BrightFieldImage.MinimumIntensityProjection(), Results)
    else:
        LogWriter.Println(f"No bright-field image is provided, chip exclusion mask cannot be identified and will be skipped...")


    if ( Results.ExplantBodyMask is None ):
        #   Next, assuming that we have a nuclear-stained fluorescent image, attempt to extract a mask
        #   corresponding to the explant body from this to further mask and reduce the neurite search space
        if ( Config.NuclearStainedImage is not None ):
            LogWriter.Println(f"Processing nuclear-stained image to identify the explant body mask.")
            Results.NuclearStainStack = Config.NuclearStainedImage.Copy().SetName(f"Nuclear Stained Stack")
            ExplantBodyExclusionMask = ProcessNuclearStain(Config.NuclearStainedImage.MaximumIntensityProjection(), ChipExclusionMask, Results)
        else:
            if ( Config.BrightFieldImage is not None ):
                LogWriter.Println(f"No nuclear-stained image provided to extract the explant body mask. Attempting to extract this from the bright-field image.")
                ExplantBodyExclusionMask = ExtractBrightFieldExplantMask(Config.BrightFieldImage.MinimumIntensityProjection(), ChipExclusionMask, Results)
            else:
                LogWriter.Println(f"No bright-field image is provided, explant body exclusion mask cannot be identified and will be skipped...")

    #   Combine the exclusion masks into one mask for all future uses
    NeuriteExclusionMask: np.ndarray = None
    if ( ChipExclusionMask is not None ):
        NeuriteExclusionMask = ChipExclusionMask.copy()
    if ( ExplantBodyExclusionMask is not None ):
        NeuriteExclusionMask *= ExplantBodyExclusionMask

    #   Now, with the regions known not to correspond to neurites identified, work throughe each layer of the neurite Z-stack
    #   and extract out the set of neurite pixels from each layer.
    if ( Config.NeuriteStainedImage is not None ):
        Results.NeuriteStainStack = Config.NeuriteStainedImage.Copy().SetName(f"Neurite Stained Stack")
        LogWriter.Println(f"Working with neurite-stained image to identify neurites.")
        for LayerIndex, NeuriteLayer in enumerate(Config.NeuriteStainedImage.Layers()):

            LogWriter.Println(f"Identifying neurites in layer [ {LayerIndex+1}/{Config.NeuriteStainedImage.LayerCount()} ]...")
            ProcessNeuriteStain(NeuriteLayer, NeuriteExclusionMask, Results)

    #   The final processing of the stacks involves working with the rods image to identify the rod orientations
    if ( Config.RodsStainedImage is not None ):
        LogWriter.Println(f"Working with rod-stained image to identify rod orientations.")
        Results.RodStainStack = Config.RodsStainedImage.Copy().SetName(f"Rod Stained Stack")
        for LayerIndex, RodLayer in enumerate(Config.RodsStainedImage.Layers()):
            LogWriter.Println(f"Identifying rods in layer [ {LayerIndex+1}/{Config.RodsStainedImage.LayerCount()} ]...")
            ProcessRodStain(RodLayer, NeuriteExclusionMask, Results)

    #   With all of the information extracted from the images acquired, compute the resulting metrics
    #   and report them back out to the user.
    Results.Quantify()

    #   Save out the configuration state for possible later review.
    Config.Save(Text=True, JSON=True)
    Results.Save(Folder=Config.OutputDirectory, DryRun=False)

    return STATUS_SUCCESS

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
    ImageType: str = ".png"

    #   Display the image to the screen
    Utils.DisplayImage(f"{ImageSequenceNumber} - {Description}", Image, DEBUG_DISPLAY_TIMEOUT, True, (not Headless) and DEBUG_DISPLAY_ENABLED)

    #   Save the image to disk.
    if ( not DryRun ):
        if ( Utils.WriteImage(Image, os.path.join(Config.OutputDirectory, f"{ImageSequenceNumber} - {Description}{ImageType}")) ):
            LogWriter.Println(f"Wrote out image [ {ImageSequenceNumber} - {Description}{ImageType} ] to [ {Config.OutputDirectory}/ ]...")
        else:
            LogWriter.Errorln(f"Failed to write out image [ {ImageSequenceNumber} - {Description}{ImageType} ] to [ {Config.OutputDirectory}/ ]...")

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

def ProcessBrightField(Image: np.ndarray, Results: QuantificationResults) -> typing.Tuple[np.ndarray, np.ndarray, typing.Tuple[int, int]]:
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
    ChipExclusionMask, ExplantBodyMask = ExtractBrightFieldMasks(BinarizedImage, Image)

    #   Cache the results generated from this phase for later storage and quantification
    Results.ChipEdgeMask = ChipExclusionMask

    if ( np.any(np.where(ExplantBodyMask == 0)) ):
        Results.ExplantBodyMask = ExplantBodyMask
        Results.ExplantBodyCentroid = DetermineExplantCentroid(Utils.ConvertTo8Bit(~ExplantBodyMask))

    return ChipExclusionMask, Results.ExplantBodyMask, Results.ExplantBodyCentroid

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

def ExtractBrightFieldMasks(Image: np.ndarray, Background: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
    """
    ExtractBrightFieldChipExclusionMask

    This function...

    Image:
        ...

    Return (np.ndarray):
        ...
    """

    MinimumComponentAreaThreshold: int = 0.01 * np.prod(Image.shape[0:2])
    MinimumContourAreaThreshold: int = 0.0005 * np.prod(Image.shape[0:2])

    #   Pre-initialize the exclusion mask to begin by including all pixels of the image, so that
    #   we can affirmatively remove known invalid locations
    WellEdgeExclusionMask: np.ndarray = np.ones_like(Image)
    ExplantBodyExclusionMask: np.ndarray = np.ones_like(Image)

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
            WellEdgeExclusionMask[Labels == ComponentID] = 0
        elif ( ContinuePromptToBoolean(Utils.DisplayImage(f"Include Component: {ComponentID} ({ComponentIndex+1}/{len(SortedComponents)}) as part of the explant body (White Region)? (y/N)", Utils.ConvertTo8Bit(ComponentMask), 0, True, True)) ):
            ExplantBodyExclusionMask[Labels == ComponentID] = 0

    #   ...
    Contours, _ = cv2.findContours(WellEdgeExclusionMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    Contours = list(filter(lambda x: cv2.contourArea(x) > MinimumContourAreaThreshold, sorted(Contours, key=lambda x: cv2.contourArea(x), reverse=True)))
    for ContourIndex, Contour in enumerate(Contours):
        ContourMask = cv2.drawContours(np.zeros_like(WellEdgeExclusionMask), [Contour], 0, 1, -1)
        ContourInverse = cv2.drawContours(np.ones_like(WellEdgeExclusionMask), [Contour], 0, 0, -1)
        if ( ContinuePromptToBoolean(Utils.DisplayImage(f"Include Contour: ({ContourIndex+1}/{len(Contours)}) as part of chip exterior (White Region)? (y/N)", Utils.ConvertTo8Bit(ContourMask), 0, True, True)) ):
            WellEdgeExclusionMask[ContourMask != 0] = 0
        elif ( ContinuePromptToBoolean(Utils.DisplayImage(f"Include Contour: ({ContourIndex+1}/{len(Contours)}) as part of chip exterior (White Region)? (y/N)", Utils.ConvertTo8Bit(ContourInverse), 0, True, True)) ):
            WellEdgeExclusionMask[ContourInverse != 0] = 0

    return WellEdgeExclusionMask, ExplantBodyExclusionMask

def ProcessNuclearStain(Image: np.ndarray, ChipExclusionMask: np.ndarray, Results: QuantificationResults) -> np.ndarray:
    """
    ProcessNuclearStain

    This function...

    Return ():
        ...
    """

    #   Normalize the image to a standard format, and then apply the chip exclusion mask to remove any parts known to be
    #   outside the range of possibility.
    NormalizedImage: np.ndarray = Utils.ConvertTo8Bit(Utils.BGRToGreyscale(Image))
    NormalizedImage = ApplyImageMask(Image, ErodeImageMask(ChipExclusionMask))

    #   The nuclear stain only stains the nucleii of the cortical explant cells, so this is usually only
    #   a bunch of nearby but disconnected dots, rather than a smooth stained region. Start by
    #   1: Blur the image
    #   2: Binarize the image
    #   3: Morphological close
    Blurred: np.ndarray = cv2.GaussianBlur(NormalizedImage, (75,75), 0)
    Binarized: np.ndarray = cv2.threshold(Blurred, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    Closed: np.ndarray = cv2.morphologyEx(Binarized, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (21,21)))

    MinimumComponentAreaThreshold: int = 0.001 * np.prod(NormalizedImage.shape)

    #   Now, identify components and determine their envelopes to find the biggest component
    ComponentCount, Labels, Stats, Centroids = cv2.connectedComponentsWithStats(Closed, connectivity=8)

    SortedComponents: typing.List[typing.Tuple[int, int]] = sorted([
        (x, Stats[x, cv2.CC_STAT_AREA]) for x in range(1, ComponentCount) if Stats[x, cv2.CC_STAT_AREA] > MinimumComponentAreaThreshold
    ], key=lambda x: x[1], reverse=True)

    ExplantCoreMask: np.ndarray = np.ones_like(Image)
    for ComponentIndex, (ComponentID, ComponentArea) in enumerate(SortedComponents):
        ComponentMask: np.ndarray = (Labels == ComponentID).astype(np.uint8)
        ClosedComponentMask: np.ndarray = cv2.morphologyEx(ComponentMask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (51,51)))

        Contours, _ = cv2.findContours(ClosedComponentMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        Contour = sorted([x for x in Contours], key=lambda x: cv2.arcLength(x, closed=True), reverse=True)[0]

        CandidateMask: np.ndarray = Utils.GreyscaleToBGR(Image.copy())
        CandidateMask = cv2.drawContours(CandidateMask, [Contour], 0, (0, 255, 0), -1)

        if ( ContinuePromptToBoolean(Utils.DisplayImage(f"Include the green region as part of the explant core? (y/N)", CandidateMask, 0, True)) ):
            ExplantCoreMask = cv2.drawContours(ExplantCoreMask, [Contour], 0, 0, -1)

    #   Cache the results into the outgoing Results value
    Results.NuclearStainStackStainMaximumProjection = Image
    Results.ExplantBodyMask = ExplantCoreMask
    Results.ExplantBodyCentroid = DetermineExplantCentroid(Utils.ConvertTo8Bit(~ExplantCoreMask))

    return ExplantCoreMask

def ExtractBrightFieldExplantMask(Image: np.ndarray, ChipExclusionMask: np.ndarray, Results: QuantificationResults) -> np.ndarray:
    """
    ExtractBrightFieldExplantMask

    This function...

    Return ():
        ...
    """

    if ( ContinuePromptToBoolean(Utils.DisplayImage(f"Has the explant body already been masked out? (y/N)", ApplyImageMask(Image, ErodeImageMask(ChipExclusionMask)), 0, True, True))):
        #   TODO:
        #   Add in logic to allow selecting the explant body to determine the centroid location.
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

    MaskedImage, InclusionMask = ApplyManualROI(ApplyImageMask(NormalizedImage, ErodeImageMask(ChipExclusionMask)), ApplyImageMask(Image, ErodeImageMask(ChipExclusionMask)), InclusionMask=True)

    #   Binarize the image, to segment the foreground from background
    BinarizedImage, ThresholdLevel = BinarizeBrightField(NormalizedImage)

    #   Apply the chip exclusion mask to remove contours and components we know can't be the explant body.
    BinarizedImage = ApplyImageMask(BinarizedImage, ErodeImageMask(ChipExclusionMask * InclusionMask))

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
        if ( len(SortedComponents) == 1 ) or ( ContinuePromptToBoolean(Utils.DisplayImage(f"Include Component: {ComponentID} ({ComponentIndex+1}/{len(SortedComponents)})? (y/N)", ApplyImageMask(BinarizedImage, ErodeImageMask(ComponentMask)), 0, True, True)) ):
            ExclusionMask[Labels == ComponentID] = 0

    #   ...
    Contours, _ = cv2.findContours(ExclusionMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    Contours = list(filter(lambda x: cv2.contourArea(x) > MinimumContourAreaThreshold and cv2.contourArea(x) <= 0.75 * np.prod(ExclusionMask.shape), sorted(Contours, key=lambda x: cv2.contourArea(x), reverse=True)))
    for ContourIndex, Contour in enumerate(Contours):
        ContourMask = cv2.drawContours(np.zeros_like(ExclusionMask), [Contour], 0, 1, -1)
        if ( len(Contours) == 1 ) or ( ContinuePromptToBoolean(Utils.DisplayImage(f"Include Contour: ({ContourIndex+1}/{len(Contours)})? (y/N)", ApplyImageMask(BinarizedImage, ErodeImageMask(ContourMask)), 0, True, True)) ):
            ExclusionMask[ContourMask != 0] = 0

    #   Cache the results...
    Results.ExplantBodyMask = ExclusionMask
    Results.ExplantBodyCentroid = DetermineExplantCentroid(Utils.ConvertTo8Bit(~ExclusionMask))
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

    SortedComponents: typing.List[int] = list(sorted([
        x for x in range(1, ComponentCount)
    ], key=lambda x: Stats[x, cv2.CC_STAT_AREA], reverse=True))

    return tuple((int(x) for x in Centroids[SortedComponents[0]]))

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

    BinarizedImage = ApplyImageMask(BinarizedImage, ErodeImageMask(ExclusionMask, Size=MaskExpansionSize))

    #   Finally, separate out all of the components of the image and filter them to remove any which do not
    #   satisfy the expectations of neurites
    FilteredNeuriteComponents = FilterNeuriteComponents(BinarizedImage, SpeckleComponentAreaThreshold, NeuriteAspectRatioThreshold, NeuriteInfillFractionThreshold)

    Results.FilteredIdentifiedNeurites.Append(FilteredNeuriteComponents)

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

def PrepareEllipticalKernel(EllipticalFilterKernelSize: float, EllipticalFilterSigma: float, EllipticalFilterMinSigma: float, EllipticalFilterScaleFactor: float) -> np.ndarray:
    """
    PrepareEllipticalKernel

    This function...

    EllipticalFilterKernelSize:
        ...
    EllipticalFilterSigma:
        ...
    EllipticalFilterMinSigma:
        ...
    EllipticalFilterScaleFactor:
        ...

    Return (np.ndarray):
        ...
    """

    #   Prepare the two asymmetric kernels to use to construct a Difference of Gaussians approximation to a Mexican Hat filter
    #   Kernel 2 must have larger sigma than Kernel 1
    Kernel1_X: np.ndarray = cv2.getGaussianKernel(EllipticalFilterKernelSize, EllipticalFilterSigma)
    Kernel1_Y: np.ndarray = cv2.getGaussianKernel(EllipticalFilterKernelSize, EllipticalFilterMinSigma)
    Kernel2_X: np.ndarray = cv2.getGaussianKernel(EllipticalFilterKernelSize, EllipticalFilterScaleFactor*EllipticalFilterSigma)
    Kernel2_Y: np.ndarray = cv2.getGaussianKernel(EllipticalFilterKernelSize, EllipticalFilterScaleFactor*EllipticalFilterMinSigma)

    #   Create a 2nd rank kernel as the outer product of the different X and Y 1st rank tensors
    Kernel1: np.ndarray = Kernel1_X * Kernel1_Y.T
    Kernel2: np.ndarray = Kernel2_X * Kernel2_Y.T

    #   Assert that the scale factor is such that the wider gaussian is subtracted from the narrower gaussian.
    if ( EllipticalFilterScaleFactor < 1 ):
        Kernel1, Kernel2 = Kernel2, Kernel1

    return (Kernel1 - Kernel2)

def ApplyEllipticalConvolution(Image: np.ndarray, DistinctOrientations: int, EllipticalKernel: np.ndarray) -> np.ndarray:
    """
    ApplyEllipticalConvolution

    This function...

    Image:
        ...
    DistinctOrientations:
        ...
    EllipticalKernel:
        ...

    Return (np.ndarray):
        ...
    """

    #   Apply the Mexican hat filter to the image for a set of N different angles,
    #   storing each result as a layer in a new "z-stack".
    AngleStack: np.ndarray = np.zeros((DistinctOrientations,) + Image.shape[0:2])

    #   For each of the orientations of interest, iterate over the half-open range of angles [90,-90)
    for Index, Angle in enumerate(np.linspace(90, -90, DistinctOrientations, endpoint=False)):

        #   Construct the rotated Difference of Gaussian kernel to apply
        K: np.ndarray = Utils.RotateFrame(EllipticalKernel, Theta=Angle)

        #   Apply the kernel over the image
        G: np.ndarray = cv2.filter2D(Image, ddepth=cv2.CV_32F, kernel=K)

        #   Truncate any pixels which end up negative
        G[G < 0] = 0

        #   Store this result in the corresponding slice of the angle-image Z-stack
        AngleStack[Index,:] = G

    #   With the results of the elliptical filter in a "Z-Stack", construct the
    #   resulting "angle image", by taking the maximum intensity pixel (and the
    #   angle of the filter it corresponds to) from the Z-stack.
    Mask: np.ndarray = np.max(AngleStack, axis=0)
    Orientations: np.ndarray = (np.argmax(AngleStack, axis=0).astype(np.float64) * (180.0 / DistinctOrientations)).astype(np.uint8)

    #   Only work with the pixels coming from the actually identified features
    Mask[Image == 0] = 0

    #   Apply a threshold to the maximum intensity pixels across the Z-stack, to
    #   isolate only those regions of the image where the correlation to the
    #   elliptical filter is strongest. Use this to mask away all of the
    #   orientation pixels which don't correspond to rods or neurites.
    _, ValidOrientations = cv2.threshold(Utils.ConvertTo8Bit(Mask), 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)

    #   Set a sentinel value for all of the orientations which are not valid
    Orientations[ValidOrientations == 0] = 255

    # Z: np.ndarray = np.zeros_like(Utils.GreyscaleToBGR(ValidOrientations), dtype=np.uint8)
    # Z[:,:,0] = Orientations
    # Z[:,:,1] = 255
    # Z[:,:,2] = ValidOrientations

    # Utils.DisplayImage(f"Angle Image", Utils.ConvertTo8Bit(cv2.cvtColor(Z, cv2.COLOR_HSV2BGR)), 0, True)

    return Orientations

def ProcessRodStain(Image: np.ndarray, ExclusionMask: np.ndarray, Results: QuantificationResults) -> None:
    """
    ProcessRodStain

    This function...

    Return ():
        ...
    """


    KernelSize: int = 201

    NormalizedImage: np.ndarray = ApplyImageMask(Utils.ConvertTo8Bit(Utils.BGRToGreyscale(Image)), ErodeImageMask(ExclusionMask))

    Background: np.ndarray = cv2.GaussianBlur(NormalizedImage, (KernelSize, KernelSize), 0)
    Foreground: np.ndarray = NormalizedImage.astype(np.int16) - Background.astype(np.int16)
    Foreground[Foreground < 0] = 0
    Foreground = Utils.ConvertTo8Bit(Foreground)

    Binarized: np.ndarray = cv2.threshold(Foreground, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

    Results.FilteredIdentifiedRods.Append(Binarized)

    return None

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

    if ( Mask is not None ):
        Mask = Utils.GammaCorrection(Mask.astype(np.float64), Gamma=1, Minimum=0.0, Maximum=1.0)
        return Utils.ConvertTo8Bit(Image * Mask)
    else:
        return Image

def ErodeImageMask(Mask: np.ndarray, Size: int = 51) -> np.ndarray:
    """
    ErodeImageMask

    This function...

    Mask:
        ...
    Size:
        ...

    Return (np.ndarray):
        ...
    """

    return cv2.erode(Mask, np.ones((Size, Size), dtype=np.uint8))

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

def ComputeNeuriteDistances(IdentifiedPixels: ZStack.ZStack, Origin: typing.Tuple[int, int]) -> typing.Sequence[np.ndarray]:
    """
    ComputeNeuriteDistances

    This function...

    Return ():
        ...
    """

    Distances: typing.List[np.ndarray] = []
    for LayerIndex, Layer in enumerate(IdentifiedPixels.Layers()):
        LogWriter.Println(f"Computing neurite lengths from identified centroid in layer [ {LayerIndex+1}/{IdentifiedPixels.LayerCount()} ]")
        NeuriteCoordinates: np.ndarray = np.argwhere(Layer != 0)
        LayerDistances: np.ndarray = np.hypot(NeuriteCoordinates[:,0] - Origin[1], NeuriteCoordinates[:,1] - Origin[0]).flatten()
        Distances.append(LayerDistances)

    return Distances

def ComputeOrientations(IdentifiedPixels: ZStack.ZStack) -> ZStack.ZStack:
    """
    ComputeOrientations

    This function...

    Return ():
        ...
    """

    FeatureSize, DistinctOrientations = 65, 18

    Orientations: ZStack.ZStack = ZStack.ZStack(LogWriter).InitializePixels(IdentifiedPixels.Pixels.shape)
    for LayerIndex, Layer in enumerate(IdentifiedPixels.Layers()):
        LogWriter.Println(f"Determining feature orientations in layer [ {LayerIndex+1}/{IdentifiedPixels.LayerCount()} ]...")
        LayerOrientations: np.ndarray = ComputeEllipticalOrientation(Layer, FeatureSize, DistinctOrientations)
        Orientations.InsertLayer(LayerOrientations, LayerIndex)

    return Orientations

def ComputeEllipticalOrientation(Image: np.ndarray, FeatureSize: float, DistinctOrientations: int) -> np.ndarray:
    """
    ComputeEllipticalOrientation

    This function...

    Image:
        ...

    Return (np.ndarray):
        ...
    """


    EllipticalFilterKernelSize: int     = Utils.RoundUpKernelToOdd(int(round(FeatureSize * 1.15)))
    EllipticalFilterSigma: float        = EllipticalFilterKernelSize / 2.0
    EllipticalFilterMinSigma: float     = EllipticalFilterSigma / 50.0
    EllipticalFilterScaleFactor: float  = EllipticalFilterKernelSize / 20.0

    EllipticalKernel: np.ndarray = PrepareEllipticalKernel(EllipticalFilterKernelSize, EllipticalFilterSigma, EllipticalFilterMinSigma, EllipticalFilterScaleFactor)

    Orientations: np.ndarray = ApplyEllipticalConvolution(Image, DistinctOrientations, EllipticalKernel)

    return Orientations

def ColourNeuriteDistances(BackgroundImage: ZStack.ZStack, IdentifiedPixels: ZStack.ZStack, Origin: typing.Tuple[int, int], MaximumDistance: float) -> ZStack.ZStack:
    """
    ColourNeuriteDistances

    This function...

    Return ():
        ...
    """

    ColouredNeuriteDistances: ZStack.ZStack = ZStack.ZStack(LogWriter, "Colour-Annotated Neurite Distances").InitializePixels(IdentifiedPixels.Pixels.shape + (3,))

    for LayerIndex, (LayerBackground, Layer) in enumerate(zip(BackgroundImage.Layers(), IdentifiedPixels.Layers())):
        LogWriter.Println(f"Generating colour-annotated neurite distance image for layer [ {LayerIndex+1}/{IdentifiedPixels.LayerCount()} ]...")

        NeuriteCoordinates: np.ndarray = np.argwhere(Layer != 0)
        ColouredLayer: np.ndarray = Utils.GreyscaleToBGR(np.zeros_like(Layer))
        Distances: np.ndarray = np.hypot(NeuriteCoordinates[:,0] - Origin[1], NeuriteCoordinates[:,1] - Origin[0])

        for DistanceIndex, Distance in enumerate(Distances):
            ColouredLayer[NeuriteCoordinates[DistanceIndex, 0], NeuriteCoordinates[DistanceIndex, 1], :] = (180 * (Distance / MaximumDistance), 255, 255)

        Background: np.ndarray = Utils.GammaCorrection(Utils.GreyscaleToBGR(LayerBackground).astype(np.float64), Minimum=0.0, Maximum=1.0)
        Foreground: np.ndarray = Utils.GammaCorrection(cv2.cvtColor(ColouredLayer, cv2.COLOR_HSV2BGR).astype(np.float64), Minimum=0.0, Maximum=1.0)
        Alpha: np.ndarray = Utils.GammaCorrection(Utils.GreyscaleToBGR(Layer).astype(np.float64), Minimum=0.0, Maximum=1.0)

        ColouredLayer = (Foreground * Alpha) + (Background * (1.0 - Alpha))
        ColouredLayer = Utils.ConvertTo8Bit(ColouredLayer)
        ColouredLayer = cv2.circle(ColouredLayer, Origin, 10, (0, 0, 255), -1)

        ColouredNeuriteDistances.InsertLayer(ColouredLayer, LayerIndex)

    return ColouredNeuriteDistances

def ColourOrientations(BackgroundImage: ZStack.ZStack, IdentifiedPixels: ZStack.ZStack, Orientations: ZStack.ZStack) -> ZStack.ZStack:
    """
    ColourOrientations

    This function...

    Return ():
        ...
    """

    ColouredOrientations: ZStack.ZStack = ZStack.ZStack(LogWriter).InitializePixels(IdentifiedPixels.Pixels.shape + (3,))

    for LayerIndex, (Background, Mask, Orientation) in enumerate(zip(BackgroundImage.Layers(), IdentifiedPixels.Layers(), Orientations.Layers())):
        LogWriter.Println(f"Generating colour-annotated orientation image for layer [ {LayerIndex+1}/{IdentifiedPixels.LayerCount()} ]...")

        Foreground: np.ndarray = Utils.GreyscaleToBGR(np.zeros_like(Background))
        Coordinates = np.argwhere(Orientation != 255)
        for Coordinate in Coordinates:
            if ( len(Coordinate) == 2 ):
                Foreground[*Coordinate, :] = (Orientation[*Coordinate], 255, 255)
            else:
                break

        Background = Utils.GammaCorrection(Utils.GreyscaleToBGR(Background).astype(np.float64), Minimum=0.0, Maximum=1.0)
        Foreground = Utils.GammaCorrection(cv2.cvtColor(Foreground, cv2.COLOR_HSV2BGR).astype(np.float64), Minimum=0.0, Maximum=1.0)
        Alpha: np.ndarray = Utils.GammaCorrection(Utils.GreyscaleToBGR(Mask.astype(np.uint8)).astype(np.float64), Minimum=0.0, Maximum=1.0)

        ColouredLayer: np.ndarray = Utils.ConvertTo8Bit((Foreground * Alpha) + (Background * (1.0 - Alpha)))
        ColouredOrientations.InsertLayer(ColouredLayer, LayerIndex)

        # DisplayAndSaveImage(ColouredLayer, f"Coloured Orientations - Layer {LayerIndex}", True, False)

    return ColouredOrientations

def PlotNeuriteDistances(DistancesByLayer: typing.Sequence[np.ndarray], flatten: bool = False) -> ZStack.ZStack | Figure:
    """
    PlotNeuriteDistances

    This function...

    Return ():
        ...
    """

    F: Figure = Utils.PrepareFigure(Interactive=False)
    MaximumDistance: float = max([np.max(x) for x in DistancesByLayer])

    if ( flatten ):
        Distances: np.ndarray = np.concatenate([x for x in DistancesByLayer])

        F = PlotNeuriteDistanceLayer(F, Distances, MaximumDistance)
        return F
    else:
        PlotStack: ZStack.ZStack = ZStack.ZStack(LogWriter)
        for LayerIndex, Layer in enumerate(DistancesByLayer):
            F = PlotNeuriteDistanceLayer(F, Layer.flatten(), MaximumDistance, LayerIndex)
            PlotStack.Append(Utils.FigureToImage(F))
            F.clear()

        return PlotStack

def PlotNeuriteDistanceLayer(F: Figure, Distances: np.ndarray, MaximumDistance: float, LayerIndex: int = -1) -> Figure:
    """
    PlotNeuriteDistanceLayer

    This function...
    F:
        ...
    Distances:
        ...

    Return (Figure):
        ...
    """

    Ax: Axes = None
    if ( len(F.get_axes()) == 0 ):
        Ax = F.add_subplot(111)
    else:
        Ax = F.get_axes()[0]
        Ax.clear()

    BinCount: int = 100
    n, bins = np.histogram(Distances, bins=BinCount, density=True)
    median, mean, stdev = np.median(Distances), np.mean(Distances), np.std(Distances)

    Ax.plot(bins[:-1], n, color='b')
    Ax.set_xlim((0, MaximumDistance))
    Ax.vlines(median, ymin=0, ymax=np.max(n), label=f"Median Length ({median:.0f}px)", color='g')
    Ax.vlines(mean, ymin=0, ymax=np.max(n), label=f"Mean Length ({mean:.0f}px)", color='r')
    Ax.vlines([mean + stdev, mean - stdev], ymin=0, ymax=np.max(n), label=f"1σ Length ({stdev:.0f}px)", color='k')
    Ax.legend()

    return F

def PlotNeuriteOrientations(OrientationsByLayer: ZStack.ZStack, flatten: bool = False) -> ZStack.ZStack | Figure:
    """
    PlotNeuriteOrientations(self.NeuriteOrientationsStack, flatten=False)

    This function...

    Return ():
        ...
    """

    F: Figure = Utils.PrepareFigure(Interactive=False)
    Ax: Axes = F.add_subplot(111)

    if ( flatten ):
        Orientations: np.ndarray = np.concatenate([x[x != 255].flatten() for x in OrientationsByLayer.Layers()]) if OrientationsByLayer.LayerCount() > 0 else []
        if ( len(Orientations) > 0 ):
            F = PlotOrientationsLayer(F, Orientations)
        return F
    else:
        PlotStack: ZStack.ZStack = ZStack.ZStack(LogWriter)
        for LayerIndex, Layer in enumerate(OrientationsByLayer.Layers()):
            F = PlotOrientationsLayer(F, Layer[Layer != 255].flatten(), LayerIndex)
            PlotStack.Append(Utils.FigureToImage(F))
            F.clear()

        return PlotStack

def PlotRodOrientations(OrientationsByLayer: ZStack.ZStack, flatten: bool = False) -> ZStack.ZStack | Figure:
    """
    PlotRodOrientations(self.RodOrientationsStack, flatten=False)

    This function...

    Return ():
        ...
    """

    F: Figure = Utils.PrepareFigure(Interactive=False)
    Ax: Axes = F.add_subplot(111)

    if ( flatten ):
        Orientations: np.ndarray = np.concatenate([x[x != 255].flatten() for x in OrientationsByLayer.Layers()]) if OrientationsByLayer.LayerCount() > 0 else []
        if ( len(Orientations) > 0 ):
            F = PlotOrientationsLayer(F, Orientations)
        return F
    else:
        PlotStack: ZStack.ZStack = ZStack.ZStack(LogWriter)
        for LayerIndex, Layer in enumerate(OrientationsByLayer.Layers()):
            F = PlotOrientationsLayer(F, Layer[Layer != 255].flatten(), LayerIndex)
            PlotStack.Append(Utils.FigureToImage(F))
            F.clear()

        return PlotStack

def PlotOrientationsLayer(F: Figure, Orientations: np.ndarray, LayerIndex: int = -1) -> Figure:
    """
    PlotOrientationsLayer

    This function...

    F:
        ...
    Orientations:
        ...
    LayerIndex:
        ...

    Return (Figure):
        ...
    """

    Ax: Axes = None
    if ( len(F.get_axes()) == 0 ):
        Ax = F.add_subplot(111)
    else:
        Ax = F.get_axes()[0]
        Ax.clear()

    BinCount: int = min(100, len(np.unique(Orientations)))
    n, bins = np.histogram(Orientations, bins=BinCount, density=True)
    mean, stdev = circmean(Orientations, high=180, low=0), circstd(Orientations, high=180, low=0)

    Ax.plot(bins[:-1], n, color='b')
    Ax.set_xlim((0, 180))
    Ax.vlines(mean, ymin=0, ymax=np.max(n), label=f"Mean Orientation ({mean:.0f})", color='r')
    Ax.vlines([mean + stdev, mean - stdev], ymin=0, ymax=np.max(n), label=f"1σ Orientation ({stdev:.0f})", color='k')
    Ax.legend()

    return F

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
    Flags.add_argument("--neurite-stain", dest="NeuriteStain", metavar="file-path", type=str, required=False, default=None, help="The neurite-stained image of the cortical explant, showing the neurite growth throughout the chip.")
    Flags.add_argument("--bright-field", dest="BrightField", metavar="file-path", type=str, required=False, default=None, help="The bright-field/transmitted light image of the cortical explant, showing the bounds of the well or chip housing the explant.")
    Flags.add_argument("--nuclear-stain", dest="NuclearStain", metavar="file-path", type=str, required=False, default=None, help="The nuclear-stained image of the cortical explant, showing the explant body to allow masking and removal of the region.")
    Flags.add_argument("--rod-stain", dest="RodsStain", metavar="file-path", type=str, required=False, default=None, help="The image showing the micro-scale magnetic rods used to align and orient the neurite growth.")

    #   Add in the flag specifying where the results generated by this script should be written out to.
    Flags.add_argument("--results-directory", dest="OutputDirectory", metavar="folder-path", type=str, required=False, default=os.path.dirname(sys.argv[0]), help="The path to the base folder into which results will be written on a per-execution basis.")
    #   ...

    #   Add in the flags and arguments which modify the parameters of the analysis algorithms
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
