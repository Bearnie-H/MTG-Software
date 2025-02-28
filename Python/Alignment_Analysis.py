#!/usr/bin/env python3

#   Author: Joseph Sadden
#   Date:   29th May, 2024

#   Script Purpose: This script provides a single consistent tool for processing
#                       and interpreting the high-speed videos of rod or neurite alignment
#                       in order to gain quantifiable information about how the
#                       alignment process occurs.

#   Import the necessary standard library modules
from __future__ import annotations

import argparse
import datetime
import jsonpickle
import math
import os
import typing
#   ...

#   Import the necessary third-part modules
import cv2
import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import scipy.stats

#   ...

#   Import the desired locally written modules
from MTG_Common.Logger import Logger, Discarder
from MTG_Common import VideoReadWriter as vwr
from MTG_Common import Utils as MyUtils
from MTG_Common import ZStack
#   ...

ANALYSIS_METHOD_SOBEL:      int = 1
ANALYSIS_METHOD_COMPONENT:  int = 2
ANALYSIS_METHOD_HOUGH:      int = 3
ANALYSIS_METHOD_ELLIPSE:    int = 4

SHOW_DEBUGGING_TEMPORARIES: bool = False
DEBUGGING_HOLD_TIME: int = 2

class Configuration():
    """
    Configuration

    This class holds the set of configurable parameters and settings for this
    program, and presents them in a single global location.
    """

    SourceFilename: str
    SourceFolder: str

    IsVideo: bool
    IsImage: bool
    IsZStack: bool
    LIFClearingAlgorithm: str

    InvertImage: bool

    DryRun: bool
    Headless: bool
    ValidateOnly: bool

    AnalysisMethod: str
    AnalysisType: int
    SkipPreprocessing: bool
    PlaybackMode: int

    AngularResolution: float
    ImageResolution: float
    FeatureLengthScale: float

    _LogWriter: Logger
    _OutputFolder: str

    def __init__(self: Configuration, LogWriter: Logger = Discarder) -> None:
        """
        Constructor

        This function prepares the Configuration object, and initializes some
        necessary default values.

        LogWriter:
            The Logger to write any and all log messages out to.

        Return (None):
            None, the Configuration instance is initialized and ready to be
            used.
        """

        self.PlaybackMode = vwr.PlaybackMode_NoDelay

        self.SourceFilename = ""
        self.SourceFolder   = ""

        self.IsVideo        = False
        self.IsImage        = False
        self.IsZStack       = False
        self.DryRun         = False
        self.Headless       = False
        self.ValidateOnly   = False

        self.InvertImage        = False
        self.SkipPreprocessing  = True

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

        FilepathToProcess: str = Arguments.Filepath
        self.AnalysisMethod = Arguments.AnalysisMethod
        self.SkipPreprocessing = Arguments.SkipPreprocessing

        self.AngularResolution = Arguments.AngularResolution
        self.FeatureLengthScale= Arguments.LengthScale
        self.ImageResolution   = Arguments.ImageResolution

        self.DryRun = Arguments.DryRun
        self.Validate = Arguments.Validate
        self.Headless = Arguments.Headless

        if ( not os.path.exists(FilepathToProcess) ):
            raise ValueError(f"Filepath [ {FilepathToProcess} ] provided does not exist!")
        elif ( os.path.isdir(FilepathToProcess) ):
            self.SourceFilename = ""
            self.SourceFolder = FilepathToProcess
        elif ( os.path.isfile(FilepathToProcess) ):
            self.SetSourceFilename(FilepathToProcess)
        else:
            raise ValueError(f"Filepath provided is neither a directory nor a regular file! [ {FilepathToProcess } ]")

        return

    def SetSourceFilename(self: Configuration, Filename: str) -> bool:
        """
        SetSourceFilename

        This function...

        Filename:
            ...

        Return (bool):
            ...
        """

        if ( Filename is None ) or ( Filename == "" ):
            self._LogWriter.Warnln(f"No source filename provided!")
            return False

        self.SourceFilename = Filename
        self._OutputFolder = ""
        self._OutputFolder = self.GetOutputDirectory()

        VideoExtensions: typing.List[str] = [".avi", ".mp4", ".mov", ".mkv"]
        ImageExtensions: typing.List[str] = [".png", ".jpg", ".jpeg", ".tif", ".tiff"]
        StackExtensions: typing.List[str] = [".lif", ".czi"]

        CurrentExtension: str = os.path.splitext(self.SourceFilename)[1].lower()
        for Extension in VideoExtensions:
            if ( Extension.lower() == CurrentExtension ):
                self.IsVideo = True
                self.IsImage = False
                self.IsZStack = False
                return True

        for Extension in ImageExtensions:
            if ( Extension.lower() == CurrentExtension ):
                self.IsVideo = False
                self.IsImage = True
                self.IsZStack = False
                return True

        for Extension in StackExtensions:
            if ( Extension.lower() == CurrentExtension ):
                self.IsVideo = False
                self.IsImage = False
                self.IsZStack = True
                return True

        self._LogWriter.Errorln(f"Unknown or unhandled file type: [ {CurrentExtension} ]!")
        return False

    def ValidateArguments(self: Configuration) -> bool:
        """
        ValidateArguments

        This function...

        Return (bool):
            ...
        """

        Validated: bool = True

        match (self.AnalysisMethod.lower().strip()):
            case "sobel" | "Sobel Edge Detection":
                self.AnalysisMethod = "Sobel Edge Detection"
                self.AnalysisType = ANALYSIS_METHOD_SOBEL
            case "component" | "Component Extraction":
                self.AnalysisMethod = "Component Extraction"
                self.AnalysisType = ANALYSIS_METHOD_COMPONENT
            case "hough" | "Hough Line Transform":
                self.AnalysisMethod = "Hough Line Transform"
                self.AnalysisType = ANALYSIS_METHOD_HOUGH
            case "ellipse" | "Elliptical Filtering":
                self.AnalysisMethod = "Elliptical Filtering"
                self.AnalysisType = ANALYSIS_METHOD_ELLIPSE
            case _:
                self._LogWriter.Errorln(f"Invalid analysis method [ {self.AnalysisMethod} ]. Must be one of 'sobel', 'component', or 'hough'.")
                Validated = False

        if ( self.FeatureLengthScale <= 0.0 ):
            self._LogWriter.Errorln(f"The length scale of features of interest must be a positive real number!")
            Validated = False

        if ( self.ImageResolution <= 0.0 ):
            self._LogWriter.Errorln(f"The spatial resolution of the image(s) to process must be a positive real number!")
            Validated = False

        if ( self.IsVideo ):
            self.IsImage = False
            if ( self.Headless ):
                self.PlaybackMode = vwr.PlaybackMode_NoDisplay
                self._LogWriter.Println(f"Asserting playback mode [ NoDisplay ] in headless mode.")
        elif ( self.IsZStack ):
            Stack: ZStack.ZStack = ZStack.ZStack.FromFile(self.SourceFilename)
            NewFilename: str = self.SourceFilename.replace(os.path.splitext(self.SourceFilename)[1], ".avi")
            vwr.VideoReadWriter.FromImageSequence(Stack.Pixels, NewFilename)
            self.SourceFilename = NewFilename
            self.IsVideo = True
            if ( self.Headless ):
                self.PlaybackMode = vwr.PlaybackMode_NoDisplay
                self._LogWriter.Println(f"Asserting playback mode [ NoDisplay ] in headless mode.")

        if ( self.AngularResolution <= 0.0 ) or ( self.AngularResolution >= 90.0 ):
            self._LogWriter.Errorln(f"Invalid angular resolution. Must be a real number between (0, 90) degrees: {self.AngularResolution}")
            Validated = False

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
    AlignmentMetrics: np.ndarray

    IntraframeAlignmentFigure: Figure
    AlignmentFractionFigure: Figure

    IntraframeAlignmentVideo: vwr.VideoReadWriter
    AlignmentFractionVideo: vwr.VideoReadWriter

    IsVideo: bool
    OutputDirectory: str

    DryRun: bool

    def __init__(self: AngleTracker, LogWriter: Logger = Logger(Prefix="AngleTracker"), OutputDirectory: str = ".", Video: bool = True, DryRun: bool = True) -> None:
        """
        Constructor

        This function...
        """

        self.Times              = np.array([])
        self.RodCounts          = np.array([])
        self.AlignmentFractions = np.array([])
        self.MeanAngles         = np.array([])
        self.AngularStDevs      = np.array([])
        self.AlignmentMetrics   = np.array([])

        self.HistogramBinSize: float = 1.0

        self.IsVideo = Video
        self.OutputDirectory = OutputDirectory

        self.DryRun = DryRun

        #   Prepare the resources in order to also prepare and display the live videos of how the alignment statistics vary over time.
        self.IntraframeAlignmentFigure = MyUtils.PrepareFigure(Interactive=False)
        self.IntraframeAlignmentVideo = None

        if ( self.IsVideo ):
            self.AlignmentFractionFigure = MyUtils.PrepareFigure(Interactive=False)
        self.AlignmentFractionVideo = None

        if ( self.DryRun ):
            self.IntraframeAlignmentVideo = vwr.VideoReadWriter(readFile=None, writeFile=None, logger=LogWriter, progress=(not LogWriter.WritesToFile()))
            self.AlignmentFractionVideo = vwr.VideoReadWriter(readFile=None, writeFile=None, logger=LogWriter, progress=(not LogWriter.WritesToFile()))
        else:
            self.IntraframeAlignmentVideo = vwr.VideoReadWriter(readFile=None, writeFile=os.path.join(OutputDirectory, "Alignment Histograms.mp4"), logger=LogWriter, progress=(not LogWriter.WritesToFile()))
            self.AlignmentFractionVideo = vwr.VideoReadWriter(readFile=None, writeFile=os.path.join(OutputDirectory, "Alignment over Time.mp4"), logger=LogWriter, progress=(not LogWriter.WritesToFile()))

        return

    def SetAngularResolution(self: AngleTracker, AngularResolution: float) -> AngleTracker:
        """
        SetAngularResolution

        This function...

        AngularResolution:
            ...

        Return (AngleTracker):
            ...
        """

        self.HistogramBinSize = AngularResolution

        return self

    def FormatCSV(self: AngleTracker) -> typing.Iterable[str]:
        """
        FormatCSV

        This function formats the rod alignment statistics in a tabular comma-separated value format for display or storage.

        Return (Iterable[str]):
            ...
        """

        Lines: typing.List[str] = []

        Lines.append(f"Time,Mean Angle,Angular StDev,Count,Alignment Fraction,Alignment Score\n")
        for Time, Mean, StDev, Count, AlignmentFraction, AlignmentMetric in zip(self.Times, self.MeanAngles, self.AngularStDevs, self.RodCounts, self.AlignmentFractions, self.AlignmentMetrics):
            Lines.append(f"{Time},{Mean},{StDev},{Count},{AlignmentFraction},{AlignmentMetric}\n")

        return Lines

    def Save(self: AngleTracker, OutputDirectory: str = None) -> None:
        """
        Save

        This function...

        Return (None):
            ...
        """

        if ( self.DryRun ):
            return

        if ( OutputDirectory is None ):
            OutputDirectory = self.OutputDirectory

        with open(os.path.join(OutputDirectory, "Orientation Data.csv"), "+w") as OutFile:
            OutFile.writelines(self.FormatCSV())

        return

    def Update(self: AngleTracker, Orientations: np.ndarray, AlignmentAngle: float, AngularStDev: float, RodCount: int = None, AlignmentFraction: float = 0, Headless: bool = False) -> typing.Tuple[Figure, Figure]:
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
        self.AlignmentMetrics = np.append(self.AlignmentMetrics, AlignmentFraction / AngularStDev)

        F1, F2 = self.UpdateFigures(Orientations, Headless)

        PlaybackMode: int = vwr.PlaybackMode_NoDelay
        if ( Headless ):
            PlaybackMode = vwr.PlaybackMode_NoDisplay

        if ( self.IntraframeAlignmentVideo is not None ):
            if ( self.IsVideo ):
                if ( self.IntraframeAlignmentVideo.OutputHeight == -1 ):
                    self.IntraframeAlignmentVideo.PrepareWriter(Resolution=tuple(reversed(MyUtils.FigureToImage(self.IntraframeAlignmentFigure).shape[:2])))
                self.IntraframeAlignmentVideo.WriteFrame(MyUtils.FigureToImage(self.IntraframeAlignmentFigure), PlaybackMode, WindowName="Mean Orientation")
            else:
                if ( not self.DryRun ):
                    cv2.imwrite(os.path.splitext(self.IntraframeAlignmentVideo._OutputFilename)[0] + ".png", MyUtils.FigureToImage(self.IntraframeAlignmentFigure))

        if ( self.IsVideo ):
            if ( self.AlignmentFractionVideo is not None ):
                if ( self.AlignmentFractionVideo.OutputHeight == -1 ):
                    self.AlignmentFractionVideo.PrepareWriter(Resolution=tuple(reversed(MyUtils.FigureToImage(self.AlignmentFractionFigure).shape[:2])))
                self.AlignmentFractionVideo.WriteFrame(MyUtils.FigureToImage(self.AlignmentFractionFigure), PlaybackMode, WindowName="Alignment over Time")

        return (F1, F2)

    def UpdateFigures(self: AngleTracker, Orientations: np.ndarray, Headless: bool = False) -> typing.Tuple[Figure, Figure]:
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

        F1, F2 = None, None

        #   Update each of the figures, showing alignment within a single frame, and the alignment fraction as a function of time.
        F1 = self.UpdateIntraFrameAlignmentFigure(Orientations)
        if ( not Headless ):
            MyUtils.DisplayFigure("Feature Alignment Polar Histogram", F1, DEBUGGING_HOLD_TIME, True, (not Config.Headless and SHOW_DEBUGGING_TEMPORARIES))

        if ( self.IsVideo ):
            F2 = self.UpdateAlignmentFractionFigure()
            if ( not Headless ):
                MyUtils.DisplayFigure("Alignment Fractions", F2, DEBUGGING_HOLD_TIME, True, (not Config.Headless and SHOW_DEBUGGING_TEMPORARIES))

        return (F1, F2)

    def UpdateIntraFrameAlignmentFigure(self: AngleTracker, Orientations: np.ndarray) -> Figure:
        """
        UpdateIntraFrameAlignmentFigure

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

        OrientationPDFAxes: Axes = None
        if ( len(F.axes) == 0 ):
            OrientationPDFAxes = F.add_subplot(111, polar=True)
            OrientationPDFAxes.set_thetamin(-90)
            OrientationPDFAxes.set_thetamax(90)
        else:
            OrientationPDFAxes: Axes = F.axes[0]

        #   Shift so that the mean angle is at 0 degrees for this plot
        NormalizedOrientations: np.ndarray = Orientations - self.MeanAngles[-1]
        NormalizedOrientations[NormalizedOrientations > 90] = NormalizedOrientations[NormalizedOrientations > 90] - 180
        NormalizedOrientations[NormalizedOrientations < -90] = NormalizedOrientations[NormalizedOrientations < -90] + 180
        UpperStDev, LowerStDev = self.AngularStDevs[-1], -self.AngularStDevs[-1]

        #   Plot the PDF of orientations as a polar antenna plot.
        n, bins = np.histogram(np.deg2rad(NormalizedOrientations), bins=int(round(180.0 / HistogramBinSizing)), range=(-np.pi/2, np.pi/2), density=True)
        OrientationPDFAxes.plot(bins[:-1], n)
        OrientationPDFAxes.vlines(np.deg2rad([LowerStDev, UpperStDev]), 0, np.max(n), colors='r', label=f"Angular Standard Deviation = {self.AngularStDevs[-1]:.3f} degrees")
        OrientationPDFAxes.set_title(f"Orientation Angular Distribution\nMeasurement Count = {self.RodCounts[-1]:.0f}\nAlignment Fraction = {self.AlignmentFractions[-1]:.3f}")
        OrientationPDFAxes.set_xlabel(f"Orientation Angles (degrees)")
        OrientationPDFAxes.set_ylabel(f"Probability Density (n.d.)")
        OrientationPDFAxes.minorticks_on()
        OrientationPDFAxes.legend()

        F.tight_layout()

        return F

    def UpdateAlignmentFractionFigure(self: AngleTracker) -> Figure:
        """
        UpdateAlignmentFractionFigure

        This function...

        Headless:
            ...

        Return (None):
            ...
        """

        F: Figure = self.AlignmentFractionFigure
        F.clear()

        A: Axes = F.gca()

        A.set_title(f"Mean Orientation vs. Time")
        A.set_xlabel(f"Time (s)")
        if ( Config.IsZStack ):
            A.set_title(f"Mean Orientation vs. Height")
            A.set_xlabel(f"Z-Stack Layer")
        A.errorbar(x=self.Times[:len(self.MeanAngles)], y=self.MeanAngles, yerr=self.AngularStDevs, capsize=2, capthick=1, ecolor='r', elinewidth=0, label="Mean Orientation")
        A.set_ylabel(f"Mean Orientation (degrees)")

        A.set_ylim([-90, 90])
        A.minorticks_on()
        A.legend()
        F.tight_layout()

        return F

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
            The total number of orientation points identified in the image
        [1] - float:
            The fraction of orientation points within 1 standard deviation of the mean rod orientation angle (i.e. Alignment Fraction)
        [2] - float:
            The mean orientation angle of the orientation points.
        [3] - float:
            The angular standard deviation of the orientation angles.
        [4] - List[float]:
            The updated and proper domain of the orientations.
    """

    Orientations[Orientations > 90] = Orientations[Orientations > 90] - 180

    MeasurementCount: int = len(Orientations)

    AngularMean: float = scipy.stats.circmean(Orientations, high=90, low=-90)
    AngularStDev: float = scipy.stats.circstd(Orientations, high=90, low=-90)

    #   Subtract the mean orientation angle, to get a copy of the orientations with mean angle at 0
    ShiftedOrientations: np.ndarray = Orientations.copy() - AngularMean
    ShiftedOrientations[ShiftedOrientations > 90] = ShiftedOrientations[ShiftedOrientations > 90] - 180
    ShiftedOrientations[ShiftedOrientations < -90] = ShiftedOrientations[ShiftedOrientations < -90] + 180

    #   Find the count of orientations with angles within 1 standard deviation of the mean orientation, and divide by the total number of measurements
    #   to get an alignment fraction value
    AlignmentFraction: float = float(len(ShiftedOrientations[abs(ShiftedOrientations) <= AngularStDev]) / MeasurementCount)

    return (MeasurementCount, AlignmentFraction, AngularMean, AngularStDev, Orientations)

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
    #   TODO: ...
    AdaptiveThresholdKernelSize: int = 9
    AdaptiveThresholdConstant: int = 5

    #   Assert that the image is in greyscale, single-channel format
    Image = MyUtils.BGRToGreyscale(Image)

    #   Contrast-Enhance the image to always be full-scale
    Image = MyUtils.ConvertTo8Bit(Image)

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
    #   TODO: ...
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

    #   TODO: ...
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

def EllipticalFilteringAlignmentMethod(Image: np.ndarray, Arguments: typing.List[typing.Any]) -> np.ndarray:
    """
    EllipticalFilteringAlignmentMethod

    This function...

    Image:
        ...
    Arguments:
        ...

    Return (np.ndarray):
        ...
    """

    Angles: AngleTracker = Arguments[1]

    FeatureSizePx: float = Config.FeatureLengthScale / Config.ImageResolution

    #   Derive the parameters required for this method from the expected or desired feature size
    #   to extract and identify.
    DistinctOrientations: int           = int(round(180.0 / Angles.HistogramBinSize))

    BackgroundRemovalKernelSize: int    = MyUtils.RoundUpKernelToOdd(int(round(FeatureSizePx * 3.0)))
    BackgroundRemovalSigma: float       = BackgroundRemovalKernelSize / 10.0

    ForegroundSmoothingKernelSize: int  = MyUtils.RoundUpKernelToOdd(int(round(FeatureSizePx / 4.0)))
    ForegroundSmoothingSigma: float     = ForegroundSmoothingKernelSize / 7.5

    EllipticalFilterKernelSize: int     = MyUtils.RoundUpKernelToOdd(int(round(FeatureSizePx * 1.15)))
    EllipticalFilterSigma: float        = EllipticalFilterKernelSize / 2.0
    EllipticalFilterMinSigma: float     = EllipticalFilterSigma / 50.0
    EllipticalFilterScaleFactor: float  = EllipticalFilterKernelSize / 20.0

    #   Pre-process the image to get it into a standardized and expected format.
    PreparedImage: np.ndarray = np.array([])
    if ( not Config.SkipPreprocessing ):
        PreparedImage = EllipticalFilter_PreprocessImage(Image.copy(), BackgroundRemovalKernelSize, BackgroundRemovalSigma, ForegroundSmoothingKernelSize, ForegroundSmoothingSigma)
    else:
        PreparedImage = MyUtils.BGRToGreyscale(Image.copy())

    #   Prepare the base elliptical kernel to work with.
    EllipticalKernel: np.ndarray = PrepareEllipticalKernel(EllipticalFilterKernelSize, EllipticalFilterSigma, EllipticalFilterMinSigma, EllipticalFilterScaleFactor)

    #   Apply the elliptical convoluation and identify all of the orientations
    Mask, Orientations = ApplyEllipticalConvolution(PreparedImage, DistinctOrientations, EllipticalKernel)

    OutputImage = CreateOrientationVisualization(Image.copy(), Orientations, Mask)

    #   ...
    Count, AlignmentFraction, AngularMean, AngularStDev, Orientations = ComputeAlignmentMetric(Orientations=Orientations)

    #   ...
    Angles.Update(Orientations=Orientations, AlignmentAngle=AngularMean, AngularStDev=AngularStDev, RodCount=Count, AlignmentFraction=AlignmentFraction, Headless=Config.Headless)

    return OutputImage

def EllipticalFilter_PreprocessImage(Image: np.ndarray, BackgroundRemovalKernelSize: int, BackgroundRemovalSigma: float, ForegroundSmoothingKernelSize: int, ForegroundSmoothingSigma: float) -> np.ndarray:
    """
    EllipticalFilter_PreprocessImage

    This function...

    Image:
        ...

    Return (np.ndarray):
        ...
    """

    #   Convert to greyscale, as we don't need colour information for this
    #   process
    Image = MyUtils.BGRToGreyscale(Image)
    MyUtils.DisplayImage("Greyscale Original Image", MyUtils.ConvertTo8Bit(Image.copy()), HoldTime=DEBUGGING_HOLD_TIME, Topmost=True, ShowOverride=(not Config.Headless and SHOW_DEBUGGING_TEMPORARIES))

    #   Linearly scale the brightness of the image to cover the full 8-bit range
    Image = MyUtils.ConvertTo8Bit(Image)
    MyUtils.DisplayImage("8-Bit Greyscale Image", MyUtils.ConvertTo8Bit(Image.copy()), HoldTime=DEBUGGING_HOLD_TIME, Topmost=True, ShowOverride=(not Config.Headless and SHOW_DEBUGGING_TEMPORARIES))

    #   Check the median pixel of the image to see the foreground is bright or
    #   dark
    MedianPixelIntensity: int = np.median(Image)

    #   Apply Otsu to the image to determine the threshold between foreground
    #   and background
    Threshold, _ = cv2.threshold(Image, 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)

    #   If the median pixel is brighter than the threshold, then this implies
    #   the background is bright and foreground is dark.
    if ( MedianPixelIntensity >= Threshold ):
        Image = -Image
        Config.InvertImage = True
    else:
        Config.InvertImage = False

    MyUtils.DisplayImage("8-Bit Greyscale Image with Dark Background", MyUtils.ConvertTo8Bit(Image.copy()), HoldTime=DEBUGGING_HOLD_TIME, Topmost=True, ShowOverride=(not Config.Headless and SHOW_DEBUGGING_TEMPORARIES))

    #   Remove background by subtracting a large-window Gaussian blurred image
    Background: np.ndarray = cv2.GaussianBlur(Image, ksize=(BackgroundRemovalKernelSize, BackgroundRemovalKernelSize), sigmaX=BackgroundRemovalSigma)
    MyUtils.DisplayImage("Blurred Background Image", MyUtils.ConvertTo8Bit(Background), DEBUGGING_HOLD_TIME, True, (not Config.Headless and SHOW_DEBUGGING_TEMPORARIES))

    Foreground: np.ndarray = Image.astype(np.int16) - Background.astype(np.int16)
    MyUtils.DisplayImage("Foreground Image", MyUtils.ConvertTo8Bit(Foreground), DEBUGGING_HOLD_TIME, True, (not Config.Headless and SHOW_DEBUGGING_TEMPORARIES))

    #   Truncate negative pixels to 0
    Foreground[Foreground < 0] = 0
    MyUtils.DisplayImage("Truncated Foreground Image", MyUtils.ConvertTo8Bit(Foreground), DEBUGGING_HOLD_TIME, True, (not Config.Headless and SHOW_DEBUGGING_TEMPORARIES))

    #   Smooth the image again, using a smaller-window Gaussian blur
    SmoothedForeground: np.ndarray = cv2.GaussianBlur(Foreground, ksize=(ForegroundSmoothingKernelSize, ForegroundSmoothingKernelSize), sigmaX=ForegroundSmoothingSigma)
    MyUtils.DisplayImage("Smoothed Foreground Image", MyUtils.ConvertTo8Bit(SmoothedForeground), DEBUGGING_HOLD_TIME, True, (not Config.Headless and SHOW_DEBUGGING_TEMPORARIES))

    #   Linearly rescale the image contrast back to the full 8-bit range
    SmoothedForeground = MyUtils.GammaCorrection(SmoothedForeground, Minimum=0, Maximum=255)
    MyUtils.DisplayImage("Full-Range Smoothed Foreground Image", MyUtils.ConvertTo8Bit(SmoothedForeground), DEBUGGING_HOLD_TIME, True, (not Config.Headless and SHOW_DEBUGGING_TEMPORARIES))

    return Image

def ApplyEllipticalConvolution(Image: np.ndarray, DistinctOrientations: int, EllipticalKernel: np.ndarray) -> typing.Tuple[np.ndarray, np.ndarray]:
    """
    ApplyEllipticalConvolution

    This function...

    Image:
        ...
    DistinctOrientations:
        ...
    EllipticalKernel:
        ...

    Return (Tuple):
        [0] - np.ndarray:
            ...
        [1] - np.ndarray:
            ...
    """

    #   Apply the Mexican hat filter to the image for a set of N different angles,
    #   storing each result as a layer in a new "z-stack".
    AngleStack: np.ndarray = np.zeros((DistinctOrientations, *MyUtils.BGRToGreyscale(Image).shape))

    #   For each of the orientations of interest, iterate over the half-open range of angles [90,-90)
    for Index, Angle in enumerate(np.linspace(90, -90, DistinctOrientations, endpoint=False)):

        #   Construct the rotated Difference of Gaussian kernel to apply
        K: np.ndarray = MyUtils.RotateFrame(EllipticalKernel, Theta=Angle)

        #   Apply the kernel over the image
        G: np.ndarray = cv2.filter2D(Image, ddepth=cv2.CV_32F, kernel=K)

        #   Truncate any pixels which end up negative
        G[G < 0] = 0

        # MyUtils.DisplayImages(
        #     Images=[
        #         (f"Elliptical Kernel: {np.rad2deg(Angle):.2f}deg", MyUtils.ConvertTo8Bit(K)),
        #         (f"Elliptical Features: {np.rad2deg(Angle):.2f}deg", MyUtils.ConvertTo8Bit(G)),
        #     ],
        #     HoldTime=DEBUGGING_HOLD_TIME,
        #     Topmost=True,
        #     ShowOverride=(not Config.Headless and SHOW_DEBUGGING_TEMPORARIES)
        # )

        #   Store this result in the corresponding slice of the angle-image Z-stack
        AngleStack[Index,:] = G

    #   With the results of the elliptical filter in a "Z-Stack", construct the
    #   resulting "angle image", by taking the maximum intensity pixel (and the
    #   angle of the filter it corresponds to) from the Z-stack.
    Mask: np.ndarray = np.max(AngleStack, axis=0)
    Orientations: np.ndarray = np.argmax(AngleStack, axis=0).astype(np.float64) * (180.0 / DistinctOrientations)

    #   Apply a threshold to the maximum intensity pixels across the Z-stack, to
    #   isolate only those regions of the image where the correlation to the
    #   elliptical filter is strongest. Use this to mask away all of the
    #   orientation pixels which don't correspond to rods or neurites.
    Otsu_Threshold, Mask = cv2.threshold(MyUtils.ConvertTo8Bit(Mask), 0, 255, cv2.THRESH_OTSU | cv2.THRESH_BINARY)

    MyUtils.DisplayImage(f"Segmented Angle Image {Otsu_Threshold}", MyUtils.ConvertTo8Bit(Mask), DEBUGGING_HOLD_TIME, True, (not Config.Headless and SHOW_DEBUGGING_TEMPORARIES))

    return Mask, Orientations

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

def CreateOrientationVisualization(OriginalImage: np.ndarray, Angles: np.ndarray, Mask: np.ndarray) -> np.ndarray:
    """
    CreateOrientationVisualization

    This function...

    OriginalImage:
        ...
    Angles:
        ...
    Mask:
        ...

    Return (np.ndarray):
        ...
    """

    #   Construct a new image, consisting of the pixels composing the foreground features,
    #   where the colour depends on the hue angle.
    OutputImage: np.ndarray = MyUtils.GreyscaleToBGR(np.zeros_like(OriginalImage, dtype=np.uint8))
    OutputImage[:,:,0] = Angles
    OutputImage[:,:,1] = 255
    OutputImage[:,:,2] = Mask
    OutputImage = cv2.cvtColor(OutputImage, cv2.COLOR_HSV2BGR)

    OriginalPixelsMask = MyUtils.GreyscaleToBGR(Mask)
    OutputImage[OriginalPixelsMask == 0] = MyUtils.GreyscaleToBGR(MyUtils.GammaCorrection(OriginalImage))[OriginalPixelsMask == 0]

    return OutputImage

def HoughAlignmentMethod(Image, Arguments) -> np.ndarray:

    raise NotImplementedError(f"HoughAlignmentMethod has not yet been implemented!")

    return Image

def ComponentAlignmentMethod (Image, Arguments):

    #   Extract out the class instance used to track and record the angular information of the rods per frame...
    Angles: AngleTracker = Arguments[1]

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

    # TODO: ...
    Angles: AngleTracker = Arguments[1]
    SobelBlockSize = 3

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

    return Gradient

def ProcessFile() -> None:
    """
    ProcessFile

    This function...

    Return (None):
        ...
    """

    LogWriter.Println(f"Working with input file: [ {Config.SourceFilename} ]...")

    #   Prepare the instance of the AngleTracker to record all of the rod
    #   alignment statistics over the length of the video or image to be
    #   processed.
    Tracker: AngleTracker = AngleTracker(LogWriter=LogWriter, OutputDirectory=Config.GetOutputDirectory(), Video=Config.IsVideo, DryRun=Config.DryRun).SetAngularResolution(Config.AngularResolution)

    #   Open up the source file as a sequence of images to work with...
    #   Respect that in dry-run mode nothing should be written to disk.
    Video: vwr.VideoReadWriter = None
    if ( Config.DryRun ):
        Video = vwr.VideoReadWriter(readFile=Config.SourceFilename, writeFile=None, logger=LogWriter, progress=(not LogWriter.WritesToFile()))
    else:
        Video = vwr.VideoReadWriter(readFile=Config.SourceFilename, writeFile=os.path.join(Config.GetOutputDirectory(), f"{Config.AnalysisMethod.title()} Method.mp4"), logger=LogWriter, progress=(not LogWriter.WritesToFile()))

    #   Prepare the timing information for the angle statistics...
    FrameCount: int = (Video.EndFrameIndex - Video.StartFrameIndex) + 1
    Tracker.Times = np.linspace(Video.StartFrameIndex, (Video.EndFrameIndex-1), FrameCount)
    Video.PrepareWriter(FrameRate=1)

    #   Actually go ahead and process the provided file, calling the _ComputeAlignmentFraction callback on each frame.
    Video.ProcessVideo(PlaybackMode=Config.PlaybackMode, Callback=_ComputeAlignmentFraction, CallbackArgs=[Config.AnalysisType, Tracker])

    #   If there were only a few frames to process, print the data to the terminal, otherwise only try printing it to a file.
    if ( len(Tracker.Times) < 10 ) or ( Config.DryRun ):
        LogWriter.Write("".join(Tracker.FormatCSV()))

    #   Save the rod alignment data to a file.
    Tracker.Save(OutputDirectory=Config.GetOutputDirectory())

    if ( not Config.DryRun ):
        with open(os.path.join(Config.GetOutputDirectory(), "Configuration-Parameters.json"), "w+") as ConfigDump:
            ConfigDump.write(jsonpickle.encode(Config, indent="\t"))

    LogWriter.Println(f"Finished working with input file: [ {Config.SourceFilename} ].")

    return

#   Main
#       This is the main entry point of the script.
def main() -> None:

    if ( Config.SourceFolder is not None ) and ( Config.SourceFolder != "" ):
        LogWriter.Println(f"Working with all files in the directory: [ {Config.SourceFolder} ]...")
        for root, _, files in os.walk(Config.SourceFolder):
            for Filename in files:
                if ( Config.SetSourceFilename(os.path.join(root, Filename)) ) and ( Config.ValidateArguments() ):
                    ProcessFile()
    elif ( Config.SourceFilename is not None ) and ( Config.SourceFilename != "" ):
        ProcessFile()
    else:
        raise RuntimeError(f"Neither source file or source folder has been provided!")

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
    Flags.add_argument("--to-process", dest="Filepath", metavar="<file|folder>", type=str, required=True, help="The path to either the file to process, or the folder of files to batch process.")

    Flags.add_argument("--method", dest="AnalysisMethod", metavar="<sobel|component|hough|ellipse>", type=str, required=False, default="ellipse", help="The rod segmentation and identification method to use.")
    Flags.add_argument("--skip-preprocess", dest="SkipPreprocessing", action="store_true", required=False, default=False, help="...")

    Flags.add_argument("--angular-resolution", dest="AngularResolution", metavar="degrees", type=float, required=False, default=1.0, help="The angular resolution of the resulting histogram of rod orientations, in units of degrees.")

    Flags.add_argument("--length-scale",     dest="LengthScale",     metavar="µm",    type=float, required=True, help="The length scale of the features of which to determine the orientation of.")
    Flags.add_argument("--image-resolution", dest="ImageResolution", metavar="µm/px", type=float, required=True, help="The size of the pixels within the image, in units of µm per pixel.")

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
