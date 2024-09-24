#!/usr/bin/env python3

#   Author: Joseph Sadden
#   Date:   29th May, 2024

#   Script Purpose: This script provides a single consistent tool for processing
#                       and interpreting the high-speed videos of rod alignment
#                       in order to gain quantifiable information about how the
#                       alignment process occurs.

#   Import the necessary standard library modules
from __future__ import annotations

import argparse
import datetime
import math
import os
import typing
#   ...

#   Import the necessary third-part modules
import cv2
import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import scipy.stats

#   ...

#   Import the desired locally written modules
from MTG_Common.Logger import Logger, Discarder
from MTG_Common import VideoReadWriter as vwr
from MTG_Common import Utils as MyUtils
#   ...

ANALYSIS_METHOD_SOBEL:      int = 1
ANALYSIS_METHOD_COMPONENT:  int = 2
ANALYSIS_METHOD_HOUGH:      int = 3
ANALYSIS_METHOD_ELLIPSE:     int = 4

class Configuration():
    """
    Configuration

    This class...
    """

    SourceFilename: str

    IsVideo: bool
    IsImage: bool
    VideoFrameRate: float

    DryRun: bool
    Headless: bool
    ValidateOnly: bool

    AnalysisMethod: str
    AnalysisType: int
    PlaybackMode: int

    #   Elliptical Filter parameters
    BackgroundRemovalKernelSize: int
    BackgroundRemovalSigma: float
    ForegroundSmoothingKernelSize: int
    ForegroundSmoothingSigma: float
    EllipticalFilterKernelSize: int
    EllipticalFilterMinSigma: float
    EllipticalFilterSigma: float
    EllipticalFilterScaleFactor: float

    _LogWriter: Logger
    _OutputFolder: str

    def __init__(self: Configuration, LogWriter: Logger = Discarder) -> None:
        """
        Constructor

        This function...

        LogWriter:
            ...

        Return (None):
            ...
        """

        self.PlaybackMode = vwr.PlaybackMode_NoDelay

        self.IsVideo         = False
        self.IsImage         = False
        self.DryRun          = False
        self.Headless        = False
        self.ValidateOnly    = False

        self._LogWriter = LogWriter
        self._OutputFolder = ""

        return

    def __str__(self: Configuration) -> str:
        """
        Stringify

        This function...

        Return (str):
            ...
        """

        return '\n'.join([

        ])

    def GetOutputDirectory(self: Configuration) -> str:
        """
        GetOutputFilename

        This function...

        Return (str):
            ...
        """

        if ( self._OutputFolder != "" ):
            return self._OutputFolder

        Timecode: str = datetime.datetime.now().strftime("%Y-%m-%d %H-%M-%S")
        BaseFolder: str = os.path.dirname(self.SourceFilename)
        OutputFolder: str = f"{os.path.splitext(os.path.basename(self.SourceFilename))[0]} - Alignment Quantification ({self.AnalysisMethod.title()}) - {Timecode}"

        self._OutputFolder = os.path.join(BaseFolder, OutputFolder)

        return self._OutputFolder

    def ExtractArguments(self: Configuration, Arguments: argparse.Namespace) -> None:
        """
        ExtractArguments

        This function...

        Arguments:
            ...

        Return (None):
            ...
        """

        self.AnalysisMethod = Arguments.AnalysisMethod
        self.SourceFilename = Arguments.Filename
        self.VideoFrameRate = 25.0
        self.InterFrameDuration = 1.0 / self.VideoFrameRate

        if ( Arguments.IsVideo ):
            self.IsVideo = True
            self.VideoFrameRate = Arguments.FrameRate
            self.InterFrameDuration = 1.0 / self.VideoFrameRate
        elif ( Arguments.IsImage ):
            self.IsImage = True

        #   Extract out the elliptical filtering parameters (which may or may not be used)
        self.BackgroundRemovalKernelSize = Arguments.BackgroundRemovalKernelSize
        self.BackgroundRemovalSigma = Arguments.BackgroundRemovalSigma
        self.ForegroundSmoothingKernelSize = Arguments.ForegroundSmoothingKernelSize
        self.ForegroundSmoothingSigma = Arguments.ForegroundSmoothingSigma
        self.EllipticalFilterKernelSize = Arguments.EllipticalFilterKernelSize
        self.EllipticalFilterMinSigma = Arguments.EllipticalFilterMinSigma
        self.EllipticalFilterSigma = Arguments.EllipticalFilterSigma
        self.EllipticalFilterScaleFactor = Arguments.EllipticalFilterScaleFactor

        self.DryRun = Arguments.DryRun
        self.Validate = Arguments.Validate
        self.Headless = Arguments.Headless

        #   ...

        return

    def ValidateArguments(self: Configuration) -> bool:
        """
        ValidateArguments

        This function...

        Return (bool):
            ...
        """

        Validated: bool = True

        match (self.AnalysisMethod.lower().strip()):
            case "sobel":
                self.AnalysisMethod = "Sobel Edge Detection"
                self.AnalysisType = ANALYSIS_METHOD_SOBEL
            case "component":
                self.AnalysisMethod = "Component Extraction"
                self.AnalysisType = ANALYSIS_METHOD_COMPONENT
            case "hough":
                self.AnalysisMethod = "Hough Line Transform"
                self.AnalysisType = ANALYSIS_METHOD_HOUGH
            case "ellipse":
                self.AnalysisMethod = "Elliptical Filtering"
                self.AnalysisType = ANALYSIS_METHOD_ELLIPSE

                if ( self.BackgroundRemovalKernelSize <= 1 ) or (( self.BackgroundRemovalKernelSize % 2 ) == 0 ):
                    self._LogWriter.Errorln(f"Invalid value for BackgroundRemovalKernelSize: {self.BackgroundRemovalKernelSize}. Must be odd and greater than 1.")
                    Validated = False
                if ( self.ForegroundSmoothingKernelSize <= 1 ) or (( self.ForegroundSmoothingKernelSize % 2 ) == 0 ):
                    self._LogWriter.Errorln(f"Invalid value for ForegroundSmoothingKernelSize: {self.ForegroundSmoothingKernelSize}. Must be odd and greater than 1.")
                    Validated = False
                if ( self.EllipticalFilterKernelSize <= 1 ) or (( self.EllipticalFilterKernelSize % 2 ) == 0 ):
                    self._LogWriter.Errorln(f"Invalid value for EllipticalFilterKernelSize: {self.EllipticalFilterKernelSize}. Must be odd and greater than 1.")
                    Validated = False

                if ( self.BackgroundRemovalSigma <= 0 ):
                    self._LogWriter.Warnln(f"Zero-value identified for BackgroundRemovalKernelSize. This will cause a default to be chosen for you.")
                if ( self.ForegroundSmoothingSigma <= 0 ):
                    self._LogWriter.Warnln(f"Zero-value identified for ForegroundSmoothingKernelSize. This will cause a default to be chosen for you.")
                if ( self.EllipticalFilterSigma <= 0 ):
                    self._LogWriter.Warnln(f"Zero-value identified for EllipticalFilterKernelSize. This will cause a default to be chosen for you.")

                if ( self.EllipticalFilterMinSigma <= 0 ) or ( self.EllipticalFilterMinSigma >= self.EllipticalFilterSigma ):
                    self._LogWriter.Errorln(f"Invalid value for EllipticalFilterMinSigma: {self.EllipticalFilterMinSigma}. Must be at least greater than EllipticalFilterSigma ({self.EllipticalFilterSigma}).")
                    Validated = False

                if ( self.EllipticalFilterScaleFactor <= 0 ):
                    self._LogWriter.Errorln(f"Invalid value for EllipticalFilterScaleFactor: {self.EllipticalFilterScaleFactor}. Must be a positive real number.")
                    Validated = False

            case _:
                self._LogWriter.Errorln(f"Invalid analysis method [ {self.AnalysisMethod} ]. Must be one of 'sobel', 'component', or 'hough'.")
                Validated = False

        if ( self.IsVideo ):
            if ( self.VideoFrameRate < 0 ):
                self._LogWriter.Errorln(f"Invalid frame-rate [ {self.VideoFrameRate}fps ].")
                Validated = False
            if ( self.Headless ):
                self.PlaybackMode = vwr.PlaybackMode_NoDisplay
                self._LogWriter.Println(f"Asserting playback mode [ NoDisplay ] in headless mode.")

        #   ...

        return ( Validated ) or ( not self.ValidateOnly )

class AngleTracker():
    """
    AngleTracker

    This class...
    """

    Times: np.ndarray
    RodCounts: np.ndarray
    AlignmentFractions: np.ndarray
    MeanAngles: np.ndarray
    AngularStDevs: np.ndarray

    IntraframeAlignmentFigure: Figure
    AlignmentFractionFigure: Figure

    RodsVideo: vwr.VideoReadWriter
    IntraframeAlignmentVideo: vwr.VideoReadWriter
    AlignmentFractionVideo: vwr.VideoReadWriter


    def __init__(self: AngleTracker, LogWriter: Logger = Logger(Prefix="AngleTracker"), RodVideo: vwr.VideoReadWriter = None, OutputDirectory: str = ".") -> None:
        """
        Constructor

        This function...
        """

        self.Times              = np.array([])
        self.RodCounts          = np.array([])
        self.AlignmentFractions = np.array([])
        self.MeanAngles         = np.array([])
        self.AngularStDevs      = np.array([])

        self.HistogramBinSize: float = 5.0

        #   Prepare the resources in order to also prepare and display the live videos of how the alignment statistics vary over time.
        self.IntraframeAlignmentFigure = MyUtils.PrepareFigure(Interactive=False)
        self.AlignmentFractionFigure = MyUtils.PrepareFigure(Interactive=False)
        self.AlignmentFractionVideo = None
        self.IntraframeAlignmentVideo = None

        self.RodsVideo = RodVideo
        if ( Config.DryRun ):
            self.IntraframeAlignmentVideo = vwr.VideoReadWriter(readFile=None, writeFile=None, logger=LogWriter, progress=(not LogWriter.WritesToFile()))
            self.AlignmentFractionVideo = vwr.VideoReadWriter(readFile=None, writeFile=None, logger=LogWriter, progress=(not LogWriter.WritesToFile()))
        else:
            self.IntraframeAlignmentVideo = vwr.VideoReadWriter(readFile=None, writeFile=os.path.join(OutputDirectory, "Rod Alignment Histograms.mp4"), logger=LogWriter, progress=(not LogWriter.WritesToFile()))
            self.AlignmentFractionVideo = vwr.VideoReadWriter(readFile=None, writeFile=os.path.join(OutputDirectory, "Rod Alignment over Time.mp4"), logger=LogWriter, progress=(not LogWriter.WritesToFile()))

        return

    def FormatCSV(self: AngleTracker) -> typing.Iterable[str]:
        """
        FormatCSV

        This function formats the rod alignment statistics in a tabular comma-separated value format for display or storage.

        Return (Iterable[str]):
            ...
        """

        Lines: typing.List[str] = []

        Lines.append(f"Time, Mean Angle, Angular StDev, Count, Alignment Fraction\n")
        for Time, Mean, StDev, Count, AlignmentFraction in zip(self.Times, self.MeanAngles, self.AngularStDevs, self.RodCounts, self.AlignmentFractions):
            Lines.append(f"{Time}, {Mean}, {StDev}, {Count}, {AlignmentFraction}\n")

        return Lines

    def Save(self: AngleTracker) -> None:
        """
        Save

        This function...

        Return (None):
            ...
        """

        if ( Config.DryRun ):
            return

        with open(os.path.join(Config.GetOutputDirectory(), "Rod Orientation Data.csv"), "+w") as OutFile:
            OutFile.writelines(self.FormatCSV())

        return

    def Update(self: AngleTracker, Orientations: np.ndarray, AlignmentAngle: float, AngularStDev: float, RodCount: int = None, AlignmentFraction: float = 0, Headless: bool = False) -> None:
        """
        Update

        This function...

        Orientations:
            ...
        AlignmentAngle:
            ...
        AngularStDev:
            ...
        RodCount:
            ...
        AlignmentFraction:
            ...

        Return (None):
            ...
        """

        self.MeanAngles = np.append(self.MeanAngles, AlignmentAngle)
        self.AngularStDevs = np.append(self.AngularStDevs, AngularStDev)
        self.RodCounts = np.append(self.RodCounts, RodCount)
        self.AlignmentFractions = np.append(self.AlignmentFractions, AlignmentFraction)

        self.UpdateFigures(Orientations, Headless)

        PlaybackMode: int = vwr.PlaybackMode_NoDelay
        if ( Headless ):
            PlaybackMode = vwr.PlaybackMode_NoDisplay

        if ( self.IntraframeAlignmentVideo is not None ):
            if ( self.IntraframeAlignmentVideo.OutputHeight == -1 ):
                self.IntraframeAlignmentVideo.PrepareWriter(Resolution=tuple(reversed(MyUtils.FigureToImage(self.IntraframeAlignmentFigure).shape[:2])))
            self.IntraframeAlignmentVideo.WriteFrame(MyUtils.FigureToImage(self.IntraframeAlignmentFigure), PlaybackMode, WindowName="Mean Rod Orientation")

        if ( self.AlignmentFractionVideo is not None ):
            if ( self.AlignmentFractionVideo.OutputHeight == -1 ):
                self.AlignmentFractionVideo.PrepareWriter(Resolution=tuple(reversed(MyUtils.FigureToImage(self.AlignmentFractionFigure).shape[:2])))
            self.AlignmentFractionVideo.WriteFrame(MyUtils.FigureToImage(self.AlignmentFractionFigure), PlaybackMode, WindowName="Rod Alignment over Time")

        return

    def UpdateFigures(self: AngleTracker, Orientations: np.ndarray, Headless: bool = False) -> None:
        """
        UpdateFigures

        This function...

        Orientations:
            ...
        Headless:
            ...

        Return (None):
            ...
        """

        #   Update each of the figures, showing alignment within a single frame, and the alignment fraction as a function of time.
        self._UpdateIntraFrameAlignmentFigure(Orientations, Headless)
        self._UpdateAlignmentFractionFigure(Headless)

        return

    def _UpdateIntraFrameAlignmentFigure(self: AngleTracker, Orientations: np.ndarray, Headless: bool = False) -> None:
        """
        _UpdateIntraFrameAlignmentFigure

        This function...

        Orientations:
            ...
        Headless:
            ...

        Return (None):
            ...
        """

        HistogramBinSizing: int = self.HistogramBinSize #   Degrees per bin

        F: Figure = self.IntraframeAlignmentFigure
        F.clear()

        A: Axes = F.gca()

        UpperStDev, LowerStDev = (self.MeanAngles[-1] + self.AngularStDevs[-1]), (self.MeanAngles[-1] - self.AngularStDevs[-1])
        UpperStDev = UpperStDev if UpperStDev < 90 else UpperStDev - 180
        LowerStDev = LowerStDev if LowerStDev > -90 else LowerStDev + 180

        n = A.hist(Orientations, bins=int(180 / HistogramBinSizing), range=(-90,90), density=True, histtype="step")[0]
        A.vlines(self.MeanAngles[-1], 0, np.max(n), colors='g', label=f"Mean Orientation = {self.MeanAngles[-1]:.3f} degrees")
        A.vlines(LowerStDev, 0, np.max(n), colors='k', label=f"Angular Standard Deviation = {self.AngularStDevs[-1]:.3f} degrees (Left)")
        A.vlines(UpperStDev, 0, np.max(n), colors='b', label=f"Angular Standard Deviation = {self.AngularStDevs[-1]:.3f} degrees (Right)")
        A.set_title(f"Rod Orientation Angular Distribution\nRod Count = {self.RodCounts[-1]}\nAlignment Fraction = {self.AlignmentFractions[-1]:.3f}")
        A.set_xlabel(f"Rod Orientation Angles (degrees)")
        A.set_ylabel(f"Probability Density (n.d.)")
        A.set_xlim([-90, 90])
        A.minorticks_on()
        A.legend()
        F.tight_layout()

        return

    def _UpdateAlignmentFractionFigure(self: AngleTracker, Headless: bool = False) -> None:
        """
        _UpdateAlignmentFractionFigure

        This function...

        Headless:
            ...

        Return (None):
            ...
        """

        F: Figure = self.AlignmentFractionFigure
        F.clear()

        A: Axes = F.gca()

        A.set_title(f"Mean Rod Orientation vs. Time")
        A.errorbar(x=self.Times[:len(self.MeanAngles)], y=self.MeanAngles, yerr=self.AngularStDevs, capsize=2, capthick=1, ecolor='r', elinewidth=0, label="Mean Orientation")
        A.set_xlabel(f"Time (s)")
        A.set_ylabel(f"Mean Rod Orientation (degrees)")

        A.set_ylim([-90, 90])
        A.minorticks_on()
        A.legend()
        F.tight_layout()

        return

#   Define the globals to set by the command-line arguments
LogWriter: Logger = Logger()
Config: Configuration = Configuration(LogWriter=LogWriter)

def ComputeAlignmentMetric(Orientations: np.ndarray) -> typing.Tuple[int, float, float, float, typing.List[float]]:
    """
    ComputeAlignmentMetric

    This function takes the set of orientations of the rods within the image, by whatever
    format this orientation data is presented in, and computes the following values:
        Count - The total number of orientation data points being worked with.
        Mean - The circular mean value of the set of orientations.
        Standard Deviation - The circular standard deviation of the set of orientations
        Alignment Fraction - The fraction of orientations within 1 standard deviation of the mean.
        Orientations - The set of orientations after shifting to a domain of [-90,90).

    Orientations:
        A list of values which encompass the orientation data points to work with for the current image.

    Return (Tuple):
        [0] - int:
            The total number of rods identified in the image
        [1] - float:
            The fraction of rods within 1 standard deviation of the mean rod orientation angle (i.e. Alignment Fraction)
        [2] - float:
            The mean orientation angle of the rods.
        [3] - float:
            The angular standard deviation of the rod angles.
        [4] - List[float]:
            The updated and proper domain of the orientations.
        ...
    """

    Orientations[Orientations > 90] = Orientations[Orientations > 90] - 180

    RodCount: int = len(Orientations)

    AngularMean: float = scipy.stats.circmean(Orientations, high=90, low=-90)
    AngularStDev: float = scipy.stats.circstd(Orientations, high=90, low=-90)

    #   Subtract the mean orientation angle, to get a copy of the orientations with mean angle at 0
    ShiftedOrientations: np.ndarray = Orientations.copy() - AngularMean

    #   Find the count of rods with angles within 1 standard deviation of the mean orientation, and divide by the total number of rods
    #   to get an alignment fraction value
    AlignmentFraction: float = float(len(ShiftedOrientations[abs(ShiftedOrientations) <= AngularStDev]) / RodCount)

    return (RodCount, AlignmentFraction, AngularMean, AngularStDev, Orientations)

def RodsAdaptiveThreshold(Image: np.ndarray) -> np.ndarray:
    """
    RodsAdaptiveThreshold

    This function:
        ...

    Image:
        ...

    Return (np.ndarray):
        ...
    """

    #   Define the parameters of this operation
    AdaptiveThresholdKernelSize: int = 9
    AdaptiveThresholdConstant: int = 5

    #   Assert that the image is in greyscale, single-channel format
    Image = MyUtils.BGRToGreyscale(Image)

    #   Contrast-Enhance the image to always be full-scale
    Image = MyUtils.GammaCorrection(Image, Minimum=0, Maximum=255)

    #   Try using an unsharp mask to reduce blurring, low-frequency components?

    #   Apply an adaptive threshold to the image, to extract the (foreground) rods from the (background) gel.
    Image = cv2.adaptiveThreshold(Image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, AdaptiveThresholdKernelSize, AdaptiveThresholdConstant)

    return Image

def RodSegmentation(Image: np.ndarray) -> typing.Tuple[np.ndarray, typing.List[int]]:
    """
    RodSegmentation

    This function...

    Image:
        ...

    Return (Tuple):
        [1] - np.ndarray:
            ...
        [2] - List[int]:
            ...
    """

    #   Define the parameters of this operation
    ComponentConnectivity: int = 8
    ComponentAreaBounds: typing.Tuple[int, int] = (10, 100)

    #   Actually extract out the set of components from the image we're interested in working with
    NumberOfComponents, LabelledImage, Stats, _ = cv2.connectedComponentsWithStats(Image, connectivity=ComponentConnectivity)

    #   Iterate over the set of components, and remove those which do not satisfy the requirements of being a rod
    FilteredLabels: typing.List[int] = []
    for ComponentId in range(1, NumberOfComponents):

        Area = Stats[ComponentId, cv2.CC_STAT_AREA]

        if not ( ComponentAreaBounds[0] <= Area <= ComponentAreaBounds[1] ):
            #   If the area is too big or too small, ignore this component
            LabelledImage[LabelledImage == ComponentId] = 0
            continue

        FilteredLabels.append(ComponentId)

    return (LabelledImage, FilteredLabels)

def EllipseContourOrientations(OriginalImage: np.ndarray, Components: np.ndarray, Labels: typing.List[int]) -> typing.Tuple[np.ndarray, np.ndarray]:
    """
    EllipseContourOrientations

    This function...

    OriginalImage:
        ...
    Components:
        ...
    Labels:
        ...

    Return (Tuple):
        [0] - np.ndarray:
            Annotated Image
        [1] - np.ndarray:
            Orientations
    """

    EllipseAreaBounds: typing.Tuple[int, int] = (10, 100)
    EllipseAxisLengthBounds: typing.Tuple[int, int] = (1, 50)

    Orientations: np.ndarray = np.array([])

    #   Create the output annotated image to return and display
    AnnotatedImage = MyUtils.GreyscaleToBGR(OriginalImage.copy())

    #   Iterate over all of the filtered components in the image, one by one
    for ComponentID in Labels:

        #   Extract out just the pixels for this one component
        # Component: np.ndarray = np.zeros_like(Components, dtype=np.uint8)
        Components[Components == ComponentID] = 255
    Components = Components.astype(np.uint8)

    #   Find the contours of just this one component, there should only be one.
    Contours, _ = cv2.findContours(Components, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)

    #   There should always and only be one contour found, but look at all of them just in case...
    for Contour in Contours:

        #   Assert that the contour always has at least 5 points
        if ( len(Contour) < 5 ):
            continue

        #   Compute the ellipse which best matches this contour...
        Ellipse = cv2.fitEllipseDirect(Contour)
        (x, y), (Major, Minor), Angle = Ellipse

        EllipseArea: float = (np.pi / 4.0) * Major * Minor  #   Since the Major and Minor values correspond to *diameters* rather than *radii*

        #   Filter out ellipses with bad dimensions
        if (math.isnan(EllipseArea)) or ( not ( EllipseAreaBounds[0] <= EllipseArea <= EllipseAreaBounds[1])):
            continue
        if not ( EllipseAxisLengthBounds[0] <= Major <= EllipseAxisLengthBounds[1] ):
            continue
        if not ( EllipseAxisLengthBounds[0] <= Minor <= EllipseAxisLengthBounds[1] ):
            continue

        Orientations = np.append(Orientations, Angle)

        #   Construct the colour of the ellipse based off the orientation it lies in.
        #   Use this angle as the Hue angle, and set both saturation and brightness to maximum
        EllipseColour: np.ndarray = np.array([[[int(Angle), 255, 255]]], dtype=np.uint8)

        #   Convert from Hue-Saturation-Brightness colour space to Blue-Green-Red colour space.
        EllipseColour = tuple(int(x) for x in cv2.cvtColor(EllipseColour, cv2.COLOR_HSV2BGR).flatten())

        #   Draw the ellipse on the original image with the correct colour.
        AnnotatedImage = cv2.ellipse(AnnotatedImage, Ellipse, EllipseColour, 1)

    return AnnotatedImage, Orientations

def _ComputeAlignmentFraction(Image: np.ndarray, Arguments: typing.List[typing.Any]) -> typing.Tuple[np.ndarray, bool]:

    """
    _ComputeAlignmentFraction

    This function is the top-level per-frame callback function used to compute
    the rod alignment statistics of interest for the videos or images of
    micro-rods.

    Image:
        The current image to be processed.
    Arguments:
        A list of user-defined arguments to be used by the callback.
        Element 0 is always the analysis method used to dispatch which analysis method to work with.
        Element 1 is always an AngleTracker instance used to pass back the orientation information after computing it.

    Return (Tuple[np.ndarray, bool]):
        [0] - np.ndarray:
            ...
        [1] - bool:
            ...
    """

    #   You can apply any transformations or operations on "Image" here 260-283
    #   ...
    AnalysisMethod: int = Arguments[0]

    if ( AnalysisMethod == ANALYSIS_METHOD_SOBEL):
        Image = SobelAlignmentMethod(Image, Arguments)
    elif ( AnalysisMethod == ANALYSIS_METHOD_COMPONENT):
        Image = ComponentAlignmentMethod(Image, Arguments)
    elif ( AnalysisMethod == ANALYSIS_METHOD_HOUGH):
        Image = HoughAlignmentMethod(Image, Arguments)
    elif ( AnalysisMethod == ANALYSIS_METHOD_ELLIPSE ):
        Image = EllipticalFilteringAlignmentMethod(Image, Arguments)

    return Image, True

def EllipticalFilteringAlignmentMethod(Image, Arguments) -> np.ndarray:

    Angles: AngleTracker = Arguments[1]
    BackgroundRemovalKernelSize: int = Config.BackgroundRemovalKernelSize
    BackgroundRemovalSigma: float = Config.BackgroundRemovalSigma
    ForegroundSmoothingKernelSize: int = Config.ForegroundSmoothingKernelSize
    ForegroundSmoothingSigma: float = Config.ForegroundSmoothingSigma
    DistinctOrientations: int = Angles.HistogramBinSize
    EllipticalFilterKernelSize: int = Config.EllipticalFilterKernelSize
    EllipticalFilterMinSigma: float = Config.EllipticalFilterMinSigma
    EllipticalFilterSigma: float = Config.EllipticalFilterSigma
    EllipticalFilterScaleFactor: float = Config.EllipticalFilterScaleFactor

    #   Pre-process the image to get it into a standardized and expected format.
    PreparedImage: np.ndarray = EllipticalFilter_PreprocessImage(Image.copy())

    #   Actually apply the elliptical filtering to identify the orientation information from the image.
    Orientations: np.ndarray = EllipticalFilter_IdentifyOrientations(PreparedImage, BackgroundRemovalKernelSize, BackgroundRemovalSigma, ForegroundSmoothingKernelSize, ForegroundSmoothingSigma, DistinctOrientations, EllipticalFilterKernelSize, EllipticalFilterMinSigma, EllipticalFilterSigma, EllipticalFilterScaleFactor)

    Count, AlignmentFraction, AngularMean, AngularStDev, Orientations = ComputeAlignmentMetric(Orientations=Orientations)

    Angles.Update(Orientations=Orientations, AlignmentAngle=AngularMean, AngularStDev=AngularStDev, RodCount=Count, AlignmentFraction=AlignmentFraction, Headless=Config.Headless)

    return Image

def EllipticalFilter_PreprocessImage(Image: np.ndarray) -> np.ndarray:
    """
    EllipticalFilter_PreprocessImage

    This function...

    Image:
        ...

    Return (np.ndarray):
        ...
    """

    #   Convert to greyscale, as we don't need colour information for this process
    Image = MyUtils.BGRToGreyscale(Image)

    #   Linearly scale the brightness of the image to cover the full 8-bit range
    Image = MyUtils.ConvertTo8Bit(Image)

    #   Check the median pixel of the image to see the foreground is bright or dark
    MedianPixelIntensity: int = np.median(Image)

    #   Apply Otsu to the image to determine the threshold between foreground and background
    Threshold, _ = cv2.threshold(Image, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)

    #   If the median pixel is brighter than the threshold, then this implies the background is bright and foreground is dark.
    if ( MedianPixelIntensity >= Threshold ):
        Image = -Image

    return Image

def EllipticalFilter_IdentifyOrientations(Image: np.ndarray, BackgroundRemovalKernelSize: int, BackgroundRemovalSigma: float, ForegroundSmoothingKernelSize: int, ForegroundSmoothingSigma: float, DistinctOrientations: int, EllipticalFilterKernelSize: int, EllipticalFilterMinSigma: float, EllipticalFilterSigma: float, EllipticalFilterScaleFactor: float) -> np.ndarray:
    """
    EllipticalFilter_IdentifyOrientations

    This function...

    BackgroundRemovalKernelSize:
        ...
    BackgroundRemovalSigma:
        ...
    ForegroundSmoothingKernelSize:
        ...
    ForegroundSmoothingSigma:
        ...
    DistinctOrientations:
        ...
    EllipticalFilterKernelSize:
        ...
    EllipticalFilterMinSigma:
        ...
    EllipticalFilterSigma:
        ...
    EllipticalFilterScaleFactor:
        ...

    Return (np.ndarray):
        ...
    """

    #   Remove background by subtracting a large-window Gaussian blurred image
    Background: np.ndarray = cv2.GaussianBlur(Image, ksize=(BackgroundRemovalKernelSize, BackgroundRemovalKernelSize), sigmaX=BackgroundRemovalSigma)
    Foreground: np.ndarray = Image.astype(np.int16) - Background.astype(np.int16)

    #   Truncate negative pixels to 0
    Foreground[Foreground < 0] = 0

    #   Smooth the image again, using a smaller-window Gaussian blur
    SmoothedForeground: np.ndarray = cv2.GaussianBlur(Foreground, ksize=(ForegroundSmoothingKernelSize, ForegroundSmoothingKernelSize), sigmaX=ForegroundSmoothingSigma)

    #   Linearly rescale the image contrast back to the full 8-bit range
    SmoothedForeground = MyUtils.GammaCorrection(SmoothedForeground, Minimum=0, Maximum=255)

    #   Apply the Mexican hat filter to the image for a set of N different angles,
    #   storing each result as a layer in a new "z-stack".
    AngleStack: np.ndarray = np.zeros((DistinctOrientations, *Image.shape))

    #   Prepare the two asymmetric kernels to use to construct a Difference of Gaussians approximation to a Mexican Hat filter
    #   Kernel 2 must have larger sigma than Kernel 1
    Kernel1_X: np.ndarray = cv2.getGaussianKernel(EllipticalFilterKernelSize, EllipticalFilterSigma)
    Kernel1_Y: np.ndarray = cv2.getGaussianKernel(EllipticalFilterKernelSize, EllipticalFilterMinSigma)
    Kernel1: np.ndarray = Kernel1_X * Kernel1_Y.T

    Kernel2_X: np.ndarray = cv2.getGaussianKernel(EllipticalFilterKernelSize, EllipticalFilterScaleFactor*EllipticalFilterSigma)
    Kernel2_Y: np.ndarray = cv2.getGaussianKernel(EllipticalFilterKernelSize, EllipticalFilterScaleFactor*EllipticalFilterMinSigma)
    Kernel2: np.ndarray = Kernel2_X * Kernel2_Y.T

    if ( EllipticalFilterScaleFactor < 1 ):
        Kernel1, Kernel2 = Kernel2, Kernel1

    #   For each of the orientations of interest, iterate over the half-open range of angles [-90,90)
    for Index, Angle in enumerate(np.linspace(-np.pi/2, np.pi/2, DistinctOrientations)):

        #   Construct the rotated Difference of Gaussian kernel to apply
        K: np.ndarray = MyUtils.RotateFrame(Kernel1 - Kernel2, Theta=np.rad2deg(Angle), Clockwise=True)

        #   Apply the kernel over the image
        G = cv2.filter2D(SmoothedForeground, ddepth=cv2.CV_32F, kernel=K)

        #   Truncate any pixels which end up negative
        G[G <= 0] = 0

        #   Store this result in the corresponding slice of the angle-image Z-stack
        AngleStack[Index,:] = G

    #   With the results of the elliptical filter in a "Z-Stack", construct the
    #   resulting "angle image", by taking the maximum intensity pixel (and the
    #   angle of the filter it corresponds to) from the Z-stack.
    MaximumPixels: np.ndarray = np.max(AngleStack, axis=0)
    MaximumPixelAngles: np.ndarray = np.argmax(AngleStack, axis=0).astype(np.float64)

    #   Apply a threshold to the maximum intensity pixels across the Z-stack, to
    #   isolate only those regions of the image where the correlation to the
    #   elliptical filter is strongest. Use this to mask away all of the
    #   orientation pixels which don't correspond to rods.
    _, MaximumPixels = cv2.threshold(MyUtils.ConvertTo8Bit(MaximumPixels), 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)
    MaximumPixelAngles = MaximumPixelAngles[MaximumPixels != 0].flatten()

    #   Finally, scale the angles from the indices of the angle stack to the actual values (in degrees) each slice corresponds to.
    MaximumPixelAngles -= (DistinctOrientations / 2)
    MaximumPixelAngles *= (180.0 / DistinctOrientations)

    return MaximumPixelAngles

def HoughAlignmentMethod(Image, Arguments) -> np.ndarray:

    raise NotImplementedError(f"HoughAlignmentMethod has not yet been implemented!")

    return Image

def ComponentAlignmentMethod (Image, Arguments):

    #   Extract out the class instance used to track and record the angular information of the rods per frame...
    Angles: AngleTracker = Arguments[1]

    ScaleFactor: float = 1.0
    Image = MyUtils.UniformRescaleImage(Image, ScalingFactor=ScaleFactor)

    #   Apply an adaptive threshold to the image in order to identify the rods versus the background
    ThresholdedImage = RodsAdaptiveThreshold(Image)

    #   Apply component-based segmentation of this binary image, to extract out the individual rods as distinct groups of pixels
    Components, Labels = RodSegmentation(ThresholdedImage)

    #   Now, find the contours of each component within the image and fit an ellipse to these points in order to extract directionality information
    AnnotatedImage, Orientations = EllipseContourOrientations(Image, Components, Labels)

    #   Compute the alignment metrics we actually care about.
    RodCount, AlignmentFraction, MeanAngle, AngularStDev, Orientations = ComputeAlignmentMetric(Orientations)

    #   Record the alignment status data for the current frame.
    Angles.Update(Orientations, MeanAngle, AngularStDev, RodCount, AlignmentFraction, Config.Headless)

    return AnnotatedImage

def SobelAlignmentMethod(Image:  np.ndarray, Arguments: typing.List[typing.Any]) -> np.ndarray:
    """
    SobelAlignmentMethod

    This function....

    Image:
        ...

    Return np.ndarray:
        ...
    """

    Angles: AngleTracker = Arguments[1]
    SobelBlockSize = 3

    ScaleFactor = 1.0
    Image = MyUtils.UniformRescaleImage(Image, ScaleFactor)

    #   Apply adaptive thresholding to identify the rods from the background
    ThresholdedImage: np.ndarray = RodsAdaptiveThreshold(Image)

    #   Apply a morphological transform to dilate the image and get the rods as larger structures.
    DilatedImage: np.ndarray = cv2.morphologyEx(ThresholdedImage, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_RECT, (7,7)))

    #   Compute a foreground mask to extract out the original pixels from the image
    Altered_Image = cv2.bitwise_and(DilatedImage, Image)

    Altered_Image = Altered_Image.astype(np.int16)
    Gx = cv2.Sobel(Altered_Image, cv2.CV_16S, 0, 1, None, ksize=SobelBlockSize)
    Gy = cv2.Sobel(Altered_Image, cv2.CV_16S, 1, 0, None, ksize=SobelBlockSize)

    EdgeDirection = np.arctan2(Gy.astype(np.float64), Gx.astype(np.float64))

    Gradient = np.zeros((Gy.shape[0], Gy.shape[1], 3), dtype=np.uint8)
    Gradient[:,:,0] = MyUtils.ConvertTo8Bit(EdgeDirection)
    Gradient[:,:,1] = 255
    Gradient[:,:,2] = MyUtils.ConvertTo8Bit(np.hypot(Gx, Gy))
    Gradient = cv2.cvtColor(Gradient, cv2.COLOR_HSV2BGR)

    EdgeDirection[Gx == 0 & Gy == 0] = np.NaN

    MeanAngle: float = scipy.stats.circmean(EdgeDirection.flatten())
    AngularStDev: float = scipy.stats.circstd(EdgeDirection.flatten())
    AlignmentFraction: float = np.sum([1 if abs(x) <= AngularStDev else 0 for x in (EdgeDirection.copy().flatten() - MeanAngle) ]) / len(EdgeDirection.flatten())
    Angles.Update(EdgeDirection.flatten(), MeanAngle, AngularStDev, RodCount=0, AlignmentFraction=AlignmentFraction, Headless=Config.Headless)

    Gradient = MyUtils.UniformRescaleImage(Gradient, 1.0 / ScaleFactor)
    return Gradient

#   Main
#       This is the main entry point of the script.
def main() -> None:

    #   Prepare the instance of the AngleTracker to record all of the rod
    #   alignment statistics over the length of the video or image to be
    #   processed.
    Tracker: AngleTracker = AngleTracker(LogWriter=LogWriter, OutputDirectory=Config.GetOutputDirectory())

    #   Open up the source file as a sequence of images to work with...
    #   Respect that in dry-run mode nothing should be written to disk.
    Video: vwr.VideoReadWriter = None
    if ( Config.DryRun ):
        Video = vwr.VideoReadWriter(readFile=Config.SourceFilename, writeFile=None, logger=LogWriter, progress=(not LogWriter.WritesToFile()))
    else:
        Video = vwr.VideoReadWriter(readFile=Config.SourceFilename, writeFile=os.path.join(Config.GetOutputDirectory(), f"{Config.AnalysisMethod.title()} Method.mp4"), logger=LogWriter, progress=(not LogWriter.WritesToFile()))

    #   Prepare the timing information for the angle statistics...
    FrameCount: int = (Video.EndFrameIndex - Video.StartFrameIndex) + 1
    Tracker.Times = np.linspace(Video.StartFrameIndex / Config.VideoFrameRate, (Video.EndFrameIndex-1) / Config.VideoFrameRate, FrameCount)
    Video.PrepareWriter(FrameRate=Config.VideoFrameRate)

    #   Actually go ahead and process the provided file, calling the _ComputeAlignmentFraction callback on each frame.
    Video.ProcessVideo(PlaybackMode=Config.PlaybackMode, Callback=_ComputeAlignmentFraction, CallbackArgs=[Config.AnalysisType, Tracker])

    #   If there were only a few frames to process, print the data to the terminal, otherwise only try printing it to a file.
    if ( len(Tracker.Times) < 10 ) or ( Config.DryRun ):
        LogWriter.Write("".join(Tracker.FormatCSV()))

    #   Save the rod alignment data to a file.
    Tracker.Save()

    return

def HandleArguments() -> bool:
    """
    HandleArguments

    This function...

    Return (bool):
        ...
    """

    #   Create the argument parser...
    Flags: argparse.ArgumentParser = argparse.ArgumentParser()

    #   Add the command-line flags and parameters...
    Flags.add_argument("--filename", dest="Filename", metavar="file-path", type=str, required=True, help="The video or image file showing rod rotation to be processed.")
    Flags.add_argument("--video", dest="IsVideo", action='store_true', default=False, help="")
    Flags.add_argument("--frame-rate", dest="FrameRate", metavar="per-second", type=float, required=False, default=-1, help="The frame-rate of the video file to process. Only checked if --video flag is set.")
    Flags.add_argument("--image", dest="IsImage", action='store_true', default=False, help="")
    Flags.add_argument("--method", dest="AnalysisMethod", metavar="<sobel|component|hough|ellipse>", type=str, required=False, default="component", help="The rod segmentation and identification method to use.")

    #   Add in argument for brightfield versus fluorescent imaging.
    #   Add in handling for Z-stack images.
    #   ...

    #   Add in the arguments for the elliptical filtering analysis method
    Flags.add_argument("--ellipse-background-kernel", dest="BackgroundRemovalKernelSize", metavar="kernel-size", type=int, required=False, default=81, help="The size of the kernel used for background subtraction.")
    Flags.add_argument("--ellipse-background-sigma", dest="BackgroundRemovalSigma", metavar="sigma", type=float, required=False, default=15, help="The standard deviation of the Gaussian blur used for background subtraction.")
    Flags.add_argument("--ellipse-smoothing-kernel", dest="ForegroundSmoothingKernelSize", metavar="kernel-size", type=int, required=False, default=11, help="The size of the kernel used for foreground smoothing.")
    Flags.add_argument("--ellipse-smoothing-sigma", dest="ForegroundSmoothingSigma", metavar="sigma", type=float, required=False, default=2, help="The standard deviation of the Gaussian blur used for foreground smoothing.")
    Flags.add_argument("--ellipse-kernel", dest="EllipticalFilterKernelSize", metavar="kernel-size", type=int, required=False, default=31, help="The size of the kernel used for the Mexican Hat filtering.")
    Flags.add_argument("--ellipse-min-sigma", dest="EllipticalFilterMinSigma", metavar="sigma", type=float, required=False, default=1, help="The standard deviation of the short axis of the Mexican Hat filter.")
    Flags.add_argument("--ellipse-sigma", dest="EllipticalFilterSigma", metavar="sigma", type=float, required=False, default=15, help="The standard deviation of the long axis of the Mexican Hat filter.")
    Flags.add_argument("--ellipse-scale-factor", dest="EllipticalFilterScaleFactor", metavar="s", type=float, required=False, default=4, help="The scale factor for the standard deviations of the Gaussian kernels used to approximate a Mexican Hat by a Difference of Gaussians.")

            #   Add in flags for manipulating the logging functionality of the script.
    Flags.add_argument("--log-file", dest="LogFile", metavar="file-path", type=str, required=False, default="-", help="File path to the file to write all log messages of this program to.")
    Flags.add_argument("--quiet",    dest="Quiet",   action="store_true",  required=False, default=False, help="Enable quiet mode, disabling logging of eveything except fatal errors.")

    #   Finally, add in scripts for modifying the basic environment or end-state of the script.
    Flags.add_argument("--dry-run",  dest="DryRun",   action="store_true", required=False, default=False, help="Enable dry-run mode, where no file-system alterations are performed.")
    Flags.add_argument("--validate", dest="Validate", action="store_true", required=False, default=False, help="Only validate the command-line arguments, do not proceed with the remainder of the program.")
    Flags.add_argument("--headless", dest="Headless", action="store_true", required=False, default=False, help="Run in 'headless' mode, where nothing is displayed to the screen.")

    #   Process the command-line arguments
    Arguments: argparse.Namespace = Flags.parse_args()

    #   Extract out the arguments.
    Config.ExtractArguments(Arguments)

    #   Attempt to validate the arguments.
    return Config.ValidateArguments()

#   Allow this script to be called from the command-line and execute the main function.
#   If anything needs to happen before executing main, add it here before the call.
if __name__ == "__main__":
    if ( HandleArguments() ):
        main()
    else:
        pass
