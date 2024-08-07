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
import math
import typing
#   ...

#   Import the necessary third-part modules
import cv2
from openpiv import tools, pyprocess, validation, filters
import numpy as np
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import scipy.stats

#   ...

#   Import the desired locally written modules
from MTG_Common.Logger import Logger, Discarder
from MTG_Common.AlignmentResults import AlignmentResult
from MTG_Common import VideoReadWriter as vwr
from MTG_Common import Utils as MyUtils
from MTG_Common import Utils as MyUtils
#   ...

ANALYSIS_METHOD_SOBEL:      int = 1
ANALYSIS_METHOD_COMPONENT:  int = 2
ANALYSIS_METHOD_HOUGH:      int = 3

class Configuration():
    """
    Configuration

    This class...
    """

    VideoFilename: str

    IsVideo: bool
    IsImage: bool
    VideoFilename: str
    VideoFrameRate: float
    ImageFilename: str

    DryRun: bool
    Headless: bool
    ValidateOnly: bool

    AnalysisMethod: str
    AnalysisType: int
    PlaybackMode: int

    _LogWriter: Logger

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

        self._LogWriter = LogWriter

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

        if ( Arguments.IsVideo ):
            self.IsVideo = True
            self.VideoFilename = Arguments.Filename
            self.VideoFrameRate = Arguments.FrameRate
            self.InterFrameDuration = 1.0 / self.VideoFrameRate
        elif ( Arguments.IsImage ):
            self.IsImage = True
            self.ImageFilename = Arguments.Filename

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

        match (self.AnalysisMethod.lower()):
            case "sobel":
                self.AnalysisMethod = "Sobel Edge Detection"
                self.AnalysisType = ANALYSIS_METHOD_SOBEL
            case "component":
                self.AnalysisMethod = "Component Extraction"
                self.AnalysisType = ANALYSIS_METHOD_COMPONENT
            case "hough":
                self.AnalysisMethod = "Hough Line Transform"
                self.AnalysisType = ANALYSIS_METHOD_HOUGH
            case _:
                self._LogWriter.Errorln(f"Invalid analysis method [ {self.AnalysisMethod} ]. Must be one of 'sobel', 'component', or 'hough'.")
                Validated = False

        if ( self.IsVideo ):
            if ( self.VideoFrameRate < 0 ):
                self._LogWriter.Errorln(f"Invalid frame-rate [ {self.VideoFrameRate}fps ].")
                Validated = False
            if ( self.Headless ):
                self.PlaybackMode = vwr.PlaybackMode_NoDisplay

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


    def __init__(self: AngleTracker) -> None:
        """
        Constructor

        This function...
        """

        self.Times              = np.array([])
        self.RodCounts          = np.array([])
        self.AlignmentFractions = np.array([])
        self.MeanAngles         = np.array([])
        self.AngularStDevs      = np.array([])

        self.IntraframeAlignmentFigure = MyUtils.PrepareFigure(Interactive=(not Config.Headless))
        self.AlignmentFractionFigure = MyUtils.PrepareFigure(Interactive=(not Config.Headless))

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

        #   ...

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

        HistogramBinSizing: int = 1 #   Degrees per bin

        F: Figure = self.IntraframeAlignmentFigure
        F.clear()

        A: Axes = F.gca()

        A.hist(Orientations, bins=(180 / HistogramBinSizing), range=(-90,90), density=True, histtype="step")
        A.set_xlabel(f"Rod Orientation Angles (degrees)")
        # A.

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

        return

#   Define the globals to set by the command-line arguments
LogWriter: Logger = Logger()
Config: Configuration = Configuration(LogWriter=LogWriter)

def ComputeAlignmentMetric(Orientatons: np.ndarray) -> typing.Tuple[int, float, float, float]:
    """
    ComputeAlignmentMetric

    This function...

    Orientations:
        ...

    Return (Tuple):
        [0] - int:
            The total number of rods identified in the image
        [1] - float:
            The fraction of rods within 1 standard deviation of the mean rod orientation angle (i.e. Alignment Fraction)
        [2] - float:
            The mean orientation angle of the rods.
                NOTE: An orientation of 0 corresponds to a purely vertical rod.
        [3] - float:
            The angular standard deviation of the rod angles.
        ...
    """

    RodCount: int = len(Orientatons)
    AngularMean: float = scipy.stats.circmean(Orientatons, high=180, low=0)
    AngularStDev: float = scipy.stats.circstd(Orientatons, high=180, low=0)

    #   Subtract the mean orientation angle, to get a copy of the orientations with mean angle at 0
    ShiftedOrientations: np.ndarray = Orientatons.copy() - AngularMean

    #   Wrap the values such that they are centred around 0 on the range (-90,90]
    ShiftedOrientations[ShiftedOrientations > 90] = 180 - ShiftedOrientations[ShiftedOrientations > 90]

    #   Find the count of rods with angles within 1 standard deviation of the mean orientation, and divide by the total number of rods
    #   to get an alignment fraction value
    AlignmentFraction: float = float(len(ShiftedOrientations[abs(ShiftedOrientations) <= AngularStDev]) / RodCount)

    Results = (RodCount, AlignmentFraction, AngularMean, AngularStDev)

    return Results

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
    AdaptiveThresholdConstant: int = 6

    #   Assert that the image is in greyscale, single-channel format
    Image = MyUtils.BGRToGreyscale(Image)

    #   Contrast-Enhance the image to always be full-scale
    Image = MyUtils.GammaCorrection(Image, Minimum=0, Maximum=255)

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
        Component: np.ndarray = Components.copy()
        Component[Component != ComponentID] = 0
        Component[Component == ComponentID] = 255
        Component = Component.astype(np.uint8)

        #   Find the contours of just this one component, there should only be one.
        Contours, _ = cv2.findContours(Component, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_SIMPLE)

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
            EllipseColour: np.ndarray = np.array([[[int(Angle), 255, 255]]], dtype=np.uint8)
            EllipseColour = tuple(int(x) for x in cv2.cvtColor(EllipseColour, cv2.COLOR_HSV2BGR).flatten())
            AnnotatedImage = cv2.ellipse(AnnotatedImage, Ellipse, EllipseColour, 1)

    return AnnotatedImage, Orientations

def _ComputeAlignmentFraction(Image: np.ndarray, Arguments: typing.List[typing.Any]) -> typing.Tuple[np.ndarray, bool]:

    """
    _ComputeAlignmentFraction

    This function...

    Image:
        ...

    Return (Tuple[np.ndarray, bool]):
        [0] - np.ndarray:
            ...
        [1] - bool:
            ...
    """

    #   You can apply any transformations or operations on "Image" here 260-283
    #   ...
    AlignmentType: int = Arguments[0]

    if ( AlignmentType == ANALYSIS_METHOD_SOBEL):
        Image = SobelAlignmentMethod(Image, Arguments)
    elif ( AlignmentType == ANALYSIS_METHOD_COMPONENT):
        Image = ComponentAlignmentMethod(Image, Arguments)
    elif ( AlignmentType == ANALYSIS_METHOD_HOUGH):
        Image = HoughAlignmentMethod(Image, Arguments)

    return Image, True

def HoughAlignmentMethod(Image, Arguments) -> np.ndarray:



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
    RodCount, AlignmentFraction, MeanAngle, AngularStDev = ComputeAlignmentMetric(Orientations)

    #   Record the alignment status data for the current frame.
    Angles.Update(Orientations, MeanAngle, AngularStDev, RodCount, AlignmentFraction)

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
    Angles.Update(EdgeDirection.flatten(), MeanAngle, AngularStDev, RodCount=0, AlignmentFraction=AlignmentFraction)

    Gradient = MyUtils.UniformRescaleImage(Gradient, 1.0 / ScaleFactor)
    return Gradient

#   Main
#       This is the main entry point of the script.
def main() -> None:

    if ( Config.VideoFilename is not None ):

        Video: vwr.VideoReadWriter = vwr.VideoReadWriter(readFile=Config.VideoFilename, writeFile=Config.GetOutputFilename(), logger=LogWriter, progress=(not LogWriter.WritesToFile()))

        FrameCount: int = (Video.EndFrameIndex - Video.StartFrameIndex)
        Tracker: AngleTracker = AngleTracker()
        Tracker.Times = np.linspace(Video.StartFrameIndex / Config.VideoFrameRate, (Video.EndFrameIndex-1) / Config.VideoFrameRate, FrameCount)

        Video.ProcessVideo(PlaybackMode=Config.PlaybackMode, Callback=_ComputeAlignmentFraction, CallbackArgs=[Config.AnalysisType, Tracker])

        plt.close()
        plt.errorbar(x=Tracker.Times, y=Tracker.MeanAngles, yerr=Tracker.AngularStDevs, capsize=2, capthick=1, ecolor='r', elinewidth=0)
        plt.xlabel(f"Time (s)")
        plt.ylabel(f"Mean Rod Orientation Angle (degrees)")
        plt.title(f"Mean Rod Orientation Angle vs. Time")
        plt.suptitle(f"6% GelMA - 0.02vol% PCL Rods - Bright-Field Imaging")
        plt.show(block=True)

        #   ...
    elif ( Config.ImageFilename is not None ):
        pass


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
    Flags.add_argument("--method", dest="AnalysisMethod", metavar="<sobel|component|hough>", type=str, required=False, default="component", help="The rod segmentation and identification method to use.")
    #   ...

            #   Add in flags for manipulating the logging functionality of the script.
    Flags.add_argument("--log-file", dest="LogFile", metavar="file-path", type=str, required=False, default="-", help="File path to the file to write all log messages of this program to.")
    Flags.add_argument("--quiet",    dest="Quiet",   action="store_true",  required=False, default=False, help="Enable quiet mode, disabling logging of eveything except fatal errors.")

    #   Finally, add in scripts for modifying the basic environment or end-state of the script.
    Flags.add_argument("--dry-run",  dest="DryRun",   action="store_true", required=False, default=False, help="Enable dry-run mode, where no file-system alterations or heavy computations are performed.")
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
