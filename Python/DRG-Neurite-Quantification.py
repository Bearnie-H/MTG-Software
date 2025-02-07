#!/usr/bin/env python3

#   Author: Joseph Sadden
#   Date:   24th January, 2025

#   Script Purpose: This script provides an automated, objective analysis pipeline
#                       for analyzing and quantifying the growth length and directionality
#                       of neurites originating from Dorsal Root Ganglia.

#   Import the necessary standard library modules
from __future__ import annotations
import argparse
import os
import sys
import traceback
import typing

#   Import the necessary third-part modules
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.path import Path
from matplotlib.widgets import PolygonSelector
import numpy as np
import cv2
from scipy.signal import correlate

#   Import the desired locally written modules
from MTG_Common import Logger
from MTG_Common import Utils
from MTG_Common import VideoReadWriter as vwr
from Alignment_Analysis import EllipticalFilter_IdentifyOrientations, ComputeAlignmentMetric, AngleTracker

DEBUG_DISPLAY_ENABLED: bool = False
DEBUG_DISPLAY_TIMEOUT: int = 4

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
    BrightFieldImage: np.ndarray
    MIPImageFile: str
    MIPImage: np.ndarray

    ApplyManualROISelection: bool

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
        self.BrightFieldImage = None
        self.MIPImageFile = ""
        self.MIPImage = None

        self.ApplyManualROISelection = False

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
        self.MIPImageFile = Arguments.MIPImage

        self.ApplyManualROISelection = Arguments.ManualROI

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

        if ( not os.path.exists(self.BrightFieldImageFile) ):
            self._LogWriter.Errorln(f"Bright Field image file does not exist!")
            Validated &= False
        else:
            self.BrightFieldImage = cv2.imread(self.BrightFieldImageFile, cv2.IMREAD_GRAYSCALE)
            self._LogWriter.Println(f"Working with bright-field image file [ {self.BrightFieldImageFile} ]...")

        if ( not os.path.exists(self.MIPImageFile) ):
            self._LogWriter.Errorln(f"Maximum Intensity Projection image file does not exist!")
            Validated &= False
        else:
            self.MIPImage = cv2.imread(self.MIPImageFile, cv2.IMREAD_GRAYSCALE)
            self._LogWriter.Println(f"Working with maximum intensity projection image file [ {self.MIPImageFile} ]...")

        if ( self.ApplyManualROISelection ) and ( self.HeadlessMode ):
            self.ApplyManualROISelection = False
            self._LogWriter.Warnln(f"Manual ROI Filtering is not compatible with headless operation... Skipping manual ROI selection and continuing.")

        return (Validated, self.ValidateOnly)

    def Save(self: Configuration, Text: bool, JSON: bool) -> bool:
        """
        Save

        This function...

        Text:
            ...
        JSON:
            ...

        Return (bool):
            ...
        """

        Success: bool = True

        if ( Text ):
            Success &= self._SaveText()
        if ( JSON ):
            Success &= self._SaveJSON()

        return Success

    ### Private Methods
    def _SaveText(self: Configuration) -> bool:
        """
        _SaveText

        This function...

        Return (bool):
            ...
        """
        Success: bool = False

        #   ...

        return Success

    def _SaveJSON(self: Configuration) -> bool:
        """
        _SaveJSON

        This function...

        Return (bool):
            ...
        """
        Success: bool = False

        #   ...

        return Success

#   Define the globals to set by the command-line arguments
LogWriter: Logger.Logger = Logger.Logger(Prefix=os.path.basename(sys.argv[0]))
Config: Configuration = Configuration(LogWriter=LogWriter)
#   ...

#   Main
#       This is the main entry point of the script.
def main() -> int:

    #   This script will take in two images, a MIP from the fluorescence Z-stack and a bright-field image
    #   of the same size, magnification, and ROI.

    #   Take the bright-field image and use this to identify the centroid of the DRG body,
    #   as well as to generate a mask to remove the DRG body from the fluorescent MIP.
    CentroidLocation, DRGBodyMask, WellEdgeMask = ProcessBrightField(Config.BrightFieldImage)

    #   Take the fluorescent image and segment out the neurite growth pixels
    Neurites: np.ndarray = ProcessFluorescent(Config.MIPImage, DRGBodyMask, WellEdgeMask)

    #   If the user has selected they would like to apply manual ROI selection to exclude specific noise regions,
    #   perform this now.
    Neurites, ManualExclusionMask = ApplyManualROI(Neurites, Config.MIPImage)

    #   With the centroid location and neurite pixels now identified, quantify the distribution of lengths of neurites
    Stats: np.ndarray = QuantifyNeuriteLengths(Neurites, CentroidLocation)

    #   TODO: This needs to be derived from command-line settings, and extracted into a dedicated function for orientation details.
    #   Finally, compute the orientation details of the neurites within the image.
    FeatureSizePx: float = 50 / 0.7644
    DistinctOrientations: int           = 90
    BackgroundRemovalKernelSize: int    = Utils.RoundUpKernelToOdd(int(round(FeatureSizePx * 3.0)))
    BackgroundRemovalSigma: float       = BackgroundRemovalKernelSize / 10.0

    ForegroundSmoothingKernelSize: int  = Utils.RoundUpKernelToOdd(int(round(FeatureSizePx / 4.0)))
    ForegroundSmoothingSigma: float     = ForegroundSmoothingKernelSize / 7.5

    EllipticalFilterKernelSize: int     = Utils.RoundUpKernelToOdd(int(round(FeatureSizePx * 1.15)))
    EllipticalFilterSigma: float        = EllipticalFilterKernelSize / 2.0
    EllipticalFilterMinSigma: float     = EllipticalFilterSigma / 50.0
    EllipticalFilterScaleFactor: float  = EllipticalFilterKernelSize / 20.0

    OrientationVisualization, Orientations = EllipticalFilter_IdentifyOrientations(Utils.BGRToGreyscale(Config.BrightFieldImage), Neurites, False, ForegroundSmoothingKernelSize, -ForegroundSmoothingSigma, DistinctOrientations, EllipticalFilterKernelSize, EllipticalFilterMinSigma, EllipticalFilterSigma, EllipticalFilterScaleFactor)
    OrientationVisualization = cv2.circle(OrientationVisualization, CentroidLocation, 10, (0, 0, 255), -1)
    DisplayAndSaveImage(OrientationVisualization, "Colour Annotated Neurite Orientations")

    RodCount, AlignmentFraction, MeanAngle, AngularStDev, Orientations = ComputeAlignmentMetric(Orientations)
    Angles: AngleTracker = AngleTracker(LogWriter, vwr.VideoReadWriter.FromImageSequence([Neurites]), Config.OutputDirectory, Video=False, DryRun=Config.DryRun).SetAngularResolution(180.0 / DistinctOrientations)
    Angles.Update(Orientations, MeanAngle, AngularStDev, RodCount, AlignmentFraction, Config.HeadlessMode)

    return 0

def DisplayAndSaveImage(Image: np.ndarray, Description: str) -> None:
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
    Utils.DisplayImage(f"{ImageSequenceNumber} - {Description}", Image, DEBUG_DISPLAY_TIMEOUT, True, (not Config.HeadlessMode) and DEBUG_DISPLAY_ENABLED)

    #   Save the image to disk.
    if ( not Config.DryRun ):
        if ( Utils.WriteImage(Image, os.path.join(Config.OutputDirectory, f"{ImageSequenceNumber} - {Description}.png")) ):
            LogWriter.Println(f"Wrote out image [ {ImageSequenceNumber} - {Description}.png ] to [ {Config.OutputDirectory}/ ]...")
        else:
            LogWriter.Errorln(f"Failed to write out image [ {ImageSequenceNumber} - {Description}.png ] to [ {Config.OutputDirectory}/ ]...")

    ImageSequenceNumber += 1

    return

def ProcessBrightField(BrightFieldImage: np.ndarray) -> typing.Tuple[typing.Tuple[int, int], np.ndarray, np.ndarray]:
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

    #   First, convert the bright field image to a full-range 8-bit image and assert that it is greyscale.
    Image: np.ndarray = Utils.ConvertTo8Bit(Utils.BGRToGreyscale(BrightFieldImage))
    DisplayAndSaveImage(Image, "Initial Bright-Field Image")

    #   Next, assert that the image is generally a bright background with a dark foreground
    if ( np.median(Image) <= np.mean(Image) ):
        Image = -Image
        DisplayAndSaveImage(Image, "Inverted Bright-Field Image")

    #   Create a copy of the image to binarize in order to perform the search for the centroid of the DRG body
    ThresholdLevel, BinarizedImage = cv2.threshold(Image.copy(), 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
    DisplayAndSaveImage(BinarizedImage, f"Otsu Binarized Image ({ThresholdLevel:.0f})")

    #   Using this binarized image, identify the centroid of the DRG body
    Centroid: typing.Tuple[int, int] = EstimateCentroid(BinarizedImage)
    #   Annotate where the centroid of the DRG body is found to be, and display this to the user...
    CentroidAnnotated: np.ndarray = cv2.circle(Utils.GreyscaleToBGR(Image.copy()), Centroid, 10, (0, 0, 255), -1)
    DisplayAndSaveImage(CentroidAnnotated, "Centroid Annotated DRG Body")

    #   With the centroid identified, review the bright field image and compute a mask which covers the
    #   body of the DRG.
    DRGBodyMask: np.ndarray = ComputeDRGMask(BinarizedImage, Centroid)
    #   Display the DRG Body Mask to the user...
    DisplayAndSaveImage(Utils.ConvertTo8Bit(DRGBodyMask), "DRG Body Mask")

    #   Compute the mask of the well edge within the image, as this is typically the source of more noise signals than anywhere else
    WellEdgeMask: np.ndarray = ComputeWellEdgeMask(BinarizedImage, Centroid)
    DisplayAndSaveImage(Utils.ConvertTo8Bit(WellEdgeMask), "Well Edge Mask")

    return (Centroid, DRGBodyMask, WellEdgeMask)

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

        #   If the centroid does not exist, then the kernel is improperly sized so we exit.
        if ( Centroid_X == 0 ) and ( Centroid_Y == 0 ):
            LogWriter.Warnln(f"Failed to identify centroid location for kernel of size [ {KernelSize} ]...")
            break

        # Utils.DisplayImage(f"Centroid Estimation - (K={KernelSize})", cv2.circle(Utils.GreyscaleToBGR(Utils.ConvertTo8Bit(CorrelationMap)), (Centroid_X, Centroid_Y), 10, (0, 0, 255), -1), DEBUG_DISPLAY_TIMEOUT, True, (not Config.HeadlessMode) or DEBUG_DISPLAY_ENABLED)

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
    a mask for excluding all DRG body pixels from the fluoresdcense

    ThresholdedImage:
        ...
    DRGCentroid:
        ...

    Return (np.ndarray):
        ...
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
    for ComponentID in OrderedComponents[:,0]:
        ComponentID = int(ComponentID)

        #   Check the size of the component, to eliminate small spot-noise which often makes it through the thresholding
        #   This area is generally in units of pixels, so we want to eliminate things which are "small"
        ComponentArea = Stats[ComponentID, cv2.CC_STAT_AREA]
        if ( ComponentArea <= MinimumComponentArea ):
            continue

        Width, Height = Stats[ComponentID, cv2.CC_STAT_WIDTH], Stats[ComponentID, cv2.CC_STAT_HEIGHT]
        if ( Width >= MaximumComponentExtent ) or ( Height >= MaximumComponentExtent ):
            continue

        #   Next, check if this component is "close" to the DRG Centroid identified earlier
        Distance: float = np.linalg.norm(np.array(DRGCentroid) - np.array(Centroids[ComponentID]))
        if ( Distance >= MaximumDistanceThreshold ):
            continue

        DRGMask[Labels == ComponentID] = 0

    #   Finally, clean up the edges of the mask with an Opening morphological transformation
    DRGMask = cv2.morphologyEx(DRGMask, cv2.MORPH_OPEN, kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (MorphologyKernelSize, MorphologyKernelSize)))
    return DRGMask

def ComputeWellEdgeMask(ThresholdedImage: np.ndarray, DRGCentroid: typing.Tuople[int, int]) -> np.ndarray:
    """
    ComputeWellEdgeMask

    This function...

    ThresholdedImage:
        ...
    DRGCentroid:
        ...

    Return (np.ndarray):
        ...
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

def ProcessFluorescent(MaximumIntensityProjection: np.ndarray, DRGBodyMask: np.ndarray, WellEdgeMask: np.ndarray) -> np.ndarray:
    """
    ProcessFluorescent

    This function...

    MaximumIntensityProjection:
        ...
    DRGBodyMask:
        ...
    WellEdgeMask:
        ...

    Return (np.ndarray):
        ...
    """

    #   Define all of the tunable parameters in one place
    AdaptiveKernelSize: int = 45
    AdaptiveOffset: int = -5
    SpeckleComponentAreaThreshold: int = 125
    MinimumAreaPercentile: int = 1

    #   First, convert the image to a full-range 8-bit image and assert that it is greyscale.
    Image: np.ndarray = Utils.ConvertTo8Bit(Utils.BGRToGreyscale(MaximumIntensityProjection))
    DisplayAndSaveImage(Image, "Initial Maximum Intensity Projection Image")

    #   Apply an adaptive local threshold over the image to further select out the neurite pixels
    ThresholdedImage: np.ndarray = cv2.adaptiveThreshold(Image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, AdaptiveKernelSize, AdaptiveOffset)
    DisplayAndSaveImage(ThresholdedImage, f"Locally Thresholded Image (k={AdaptiveKernelSize}, C={AdaptiveOffset})")

    #   Expand the original DRG Body mask again slightly, to not include artefacts from the "edge" introduced
    #   by applying the mask
    MaskExpansionSize: int = int(AdaptiveKernelSize / 4)
    ThresholdedImage = Utils.GammaCorrection(ThresholdedImage * cv2.erode(DRGBodyMask, kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(MaskExpansionSize, MaskExpansionSize))))
    DisplayAndSaveImage(ThresholdedImage, f"DRG Body Mask Edge Removed")

    #   Expand the original Well Edge mask again slightly, to not include artefacts from the "edge" introduced
    #   by applying the mask
    ThresholdedImage = Utils.GammaCorrection(ThresholdedImage * cv2.erode(WellEdgeMask, kernel=cv2.getStructuringElement(cv2.MORPH_ELLIPSE, ksize=(MaskExpansionSize, MaskExpansionSize))))
    DisplayAndSaveImage(ThresholdedImage, f"Well Edge Mask Edge Removed")

    #   Next, we want to removal little speckle noise and other very small regions which make it through the thresholding.
    #   Identify connected components and only accept those larger than some area threshold
    NumberOfComponents, Labels, Stats, Centroids = cv2.connectedComponentsWithStats(ThresholdedImage, connectivity=4)
    #   Determine the distances of each component centroid from the previously identified DRG centroid
    OrderedComponents: np.ndarray = np.array(sorted([
        (x, Stats[x, cv2.CC_STAT_AREA]) for x in range(1, NumberOfComponents) if Stats[x, cv2.CC_STAT_AREA] > SpeckleComponentAreaThreshold
    ], key=lambda x: x[1], reverse=True))
    MinimumAreaThreshold: float = np.percentile(OrderedComponents[:,1], MinimumAreaPercentile)

    #   Re-build the components into a single image to work with, containing only the meaningful components of the image.
    FilteredComponents: np.ndarray = np.zeros_like(ThresholdedImage)
    for ComponentID, ComponentArea in OrderedComponents:
        if ( ComponentArea < MinimumAreaThreshold ):
            continue

        #   Maybe add some filtering based off the circularity of the components?
        #   We'd expect neurites to be either high aspect ratio, or if the bounding box is close to square, the filled area would be low.
        ComponentMask: np.ndarray = (Labels == ComponentID).astype(np.uint8)
        Contours, _ = cv2.findContours(ComponentMask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if ( len(Contours) > 0 ):

            (Centre, (Width, Height), Orientation) = cv2.minAreaRect(Contours[0])

            AspectRatio: float = np.max([Height, Width]) / np.min([Height, Width])
            BoundingArea: int = Height * Width
            if ( AspectRatio <= 1.5 ):
                FilledFraction: float = BoundingArea / ComponentArea
                if ( FilledFraction >= 0.85 ):
                    continue
            #   ...

            FilteredComponents[Labels == ComponentID] = 1

    DisplayAndSaveImage(Utils.ConvertTo8Bit(FilteredComponents), "Filtered Connected Components after Local Thresholding")

    return FilteredComponents

def ApplyManualROI(ImageToFilter: np.ndarray, Background: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
    """
    ApplyManualROI

    This function...

    ImageToFilter:
        ...
    Background:
        ...

    Return (Tuple):
        [0] - np.ndarray:
            ...
        [1] - np.ndarray:
            ...
    """

    PolygonExclusionMask: np.ndarray = np.ones_like(Utils.BGRToGreyscale(ImageToFilter))

    if ( Config.ApplyManualROISelection ):

        def UpdateExclusionMask(Vertices: np.ndarray) -> None:
            nonlocal PolygonExclusionMask
            Vertices = np.array([
                (int(X), int(Y)) for (X, Y) in Vertices
            ])
            PolygonExclusionMask = cv2.drawContours(PolygonExclusionMask, [Vertices], 0, 0, -1)
            Utils.DisplayImage(f"Current Polygon Exclusion Result", Utils.ConvertTo8Bit(PolygonExclusionMask * ImageToFilter), DEBUG_DISPLAY_TIMEOUT, True, True)

        Base: np.ndarray = Utils.GammaCorrection(Utils.GreyscaleToBGR(Background).astype(np.float64), Minimum=0, Maximum=1.0)
        Foreground: np.ndarray = Utils.GammaCorrection(Utils.GreyscaleToBGR(ImageToFilter).astype(np.float64), Minimum=0, Maximum=1.0)
        Foreground[:,:,0] = 0
        Foreground[:,:,2] = 0
        Alpha: np.ndarray = Foreground.copy()

        Result = (Foreground * Alpha) + (Base * (1.0 - Alpha))

        F, Ax = plt.subplots()
        AxIm = Ax.imshow(Result, origin='upper')

        P = PolygonSelector(Ax, onselect=UpdateExclusionMask, props=dict(color='r', linestyle='-', linewidth=2))
        plt.show(block=True)

    DisplayAndSaveImage(Utils.ConvertTo8Bit(PolygonExclusionMask), "Polygon Exclusion Mask")

    PolygonMaskedImage: np.ndarray = Utils.BGRToGreyscale(ImageToFilter) * PolygonExclusionMask
    DisplayAndSaveImage(Utils.ConvertTo8Bit(PolygonMaskedImage), "Polygon Exclusion Masked Image")

    return PolygonMaskedImage, PolygonExclusionMask

def QuantifyNeuriteLengths(NeuritePixels: np.ndarray, Origin: typing.Tuple[int, int]) -> np.ndarray:
    """
    QuantifyNeuriteLengths

    This function...

    NeuritePixels:
        ...
    Origin:
        ...

    Return (np.ndarray):
        ...
    """

    #   Identify the indices of the image corresponding to the neurite pixels
    NeuriteCoordinates = np.argwhere(NeuritePixels != 0)

    #   Compute the L2 norm from the origin point to each identified neurite pixel
    Distances: np.ndarray = np.hypot(NeuriteCoordinates[:,0] - Origin[1], NeuriteCoordinates[:,1] - Origin[0])

    #   Prepare a visualization of the neurite lengths, with increasing length corresponding to varying hue
    NeuriteLengthVisualization: np.ndarray = Utils.GreyscaleToBGR(np.zeros_like(NeuritePixels))

    #   Add in the centroid location
    NeuriteLengthVisualization = cv2.circle(NeuriteLengthVisualization, Origin, 10, (0, 255, 255), -1)
    MaximumLength: float = np.max(Distances)
    for Index, Distance in enumerate(Distances):
        NeuriteLengthVisualization[NeuriteCoordinates[Index, 0], NeuriteCoordinates[Index, 1], :] = (180 * (Distance / MaximumLength), 255, 255)
    DisplayAndSaveImage(cv2.cvtColor(NeuriteLengthVisualization, cv2.COLOR_HSV2BGR), "Neurite Length Visualization")

    Base: np.ndarray = Utils.GammaCorrection(Utils.GreyscaleToBGR(Config.BrightFieldImage.copy()).astype(np.float64), Minimum=0, Maximum=1.0)
    Foreground: np.ndarray = Utils.GammaCorrection(cv2.cvtColor(NeuriteLengthVisualization, cv2.COLOR_HSV2BGR).astype(np.float64), Minimum=0, Maximum=1.0)
    Alpha: np.ndarray = Utils.GammaCorrection(Utils.GreyscaleToBGR(NeuritePixels).astype(np.float64), Minimum=0, Maximum=1.0)

    Result = (Foreground * Alpha) + (Base * (1.0 - Alpha))
    Result = cv2.circle(Result, Origin, 10, (0, 0, 1.0), -1)
    DisplayAndSaveImage(Utils.ConvertTo8Bit(Result), "Colour Annotated Identified Neurites")

    CreateQuantificationFigures(Distances)

    return Distances

def CreateQuantificationFigures(NeuriteLengths: np.ndarray) -> None:
    """
    CreateQuantificationFigures

    This function...

    NeuriteLengths:
        ...

    Return (None):
        ...
    """

    BinCount: int = 100
    n, bins = np.histogram(NeuriteLengths, bins=BinCount, density=True)
    median, mean, stdev = np.median(NeuriteLengths), np.mean(NeuriteLengths), np.std(NeuriteLengths)

    F: Figure = Utils.PrepareFigure(Interactive=(not Config.HeadlessMode))
    A: Axes = F.add_subplot(111)

    F.suptitle(f"Dorsal Root Ganglion in Ultimatrix")
    A.set_title(f"Neurite Length Quantification - Total Pixel Count {len(NeuriteLengths)}")
    A.set_xlabel(f"Neurite Length (px)")
    A.set_ylabel(f"Normalized Pixel Count")

    A.plot(bins[:-1], n, color='b')
    A.vlines(median, ymin=0, ymax=np.max(n), label=f"Median Length ({median:.0f}px)", color='g')
    A.vlines(mean, ymin=0, ymax=np.max(n), label=f"Mean Length ({mean:.0f}px)", color='r')
    A.vlines([mean + stdev, mean - stdev], ymin=0, ymax=np.max(n), label=f"1σ Length ({stdev:.0f}px)", color='k')
    A.legend()

    DisplayAndSaveImage(Utils.FigureToImage(F), "Neurite Length Distribution")

    return

def HandleArguments() -> bool:
    """
    HandleArguments

    This function...

    Return (bool):
        ...
    """

    #   Prepare the argument parser
    Flags: argparse.ArgumentParser = argparse.ArgumentParser(description="wlkhg")

    #   Add in the command-line flags to accept
    #   Add in the flags for the bright-field and fluorescent MIP images to work from.
    Flags.add_argument("--bf-image",  dest="BrightField", metavar="file-path", type=str, required=True, help="The file path to the bright-field image file to work with.")
    Flags.add_argument("--mip-image", dest="MIPImage",    metavar="file-path", type=str, required=True, help="The file path to the maximum intensity image computed from the fluorescent Z-stack to work with.")

    Flags.add_argument("--manual-roi", dest="ManualROI", action="store_true", required=False, default=False, help="...")

    #   Add in the flag specifying where the results generated by this script should be written out to.
    Flags.add_argument("--results-directory", dest="OutputDirectory", metavar="folder-path", type=str, required=False, default=os.getcwd(), help="The path to the base folder into which results will be written on a per-execution basis.")
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
