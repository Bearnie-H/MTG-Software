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

    WindowSize: int
    WindowOverlap: int
    InterFrameDuration: float
    PixelSize: float
    SNRThreshold: float
    SmoothingStDevThreshold: float
    SmoothingKernelSize: int
    SmoothingIterations: int

    DryRun: bool
    Headless: bool
    ValidateOnly: bool

    AlignmentMethod: str

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

        self._LogWriter = LogWriter
        self.AlignmentMethod = ""

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

        Config.VideoFilename = Arguments.Filename

        if ( Arguments.IsVideo ):
            self.IsVideo = True
            self.VideoFilename = Arguments.Filename
            self.VideoFrameRate = Arguments.FrameRate
            self.InterFrameDuration = 1.0 / self.VideoFrameRate
        elif ( Arguments.IsImage ):
            self.IsImage = True
            self.ImageFilename = Arguments.Filename

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

        #   ...

        return Validated

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

        return

class PIVAnalyzer():
    """
    PIVAnalyzer

    This class...
    """

    WindowSize: int
    WindowOverlap: int
    InterFrameDuration: float
    PixelSize: float
    SNRThreshold: float
    SmoothingStDevThreshold: float
    SmoothingKernelSize: int
    SmoothingIterations: int

    VelocityFields: np.ndarray

    _PreviousFrame: np.ndarray
    _LogWriter: Logger

    def __init__(self: PIVAnalyzer, LogWriter: Logger = Logger(Prefix="PIVAnalyzer")) -> None:
        """
        Constructor

        This function...

        LogWriter:
            ...

        Return (None):
            ...
        """

        self._LogWriter = LogWriter

        #   ...

        return None

    def ConfigurePIVSettings(self: PIVAnalyzer, InterFrameDuration: float, PixelSize: int, WindowSize: int = 32, WindowOverlap: int = 16, SNRThreshold: float = 1.0, SmoothingStDevThreshold: float = 3.0, SmoothingKernelSize: int = 7, SmoothingIterations: int = 3) -> PIVAnalyzer:
        """
        ConfigurePIVSettings

        This function...

        InterFrameDuration:
            ...
        PixelSize:
            ...
        WindowSize:
            ...
        WindowOverlap:
            ...
        SNRThreshold:
            ...
        SmoothingStDevThreshold:
            ...
        SmoothingKernelSize:
            ...
        SmoothingIterations:
            ...

        Return (PIVAnalyzer):
            ...
        """

        self.InterFrameDuration      = InterFrameDuration
        self.PixelSize               = PixelSize
        self.WindowSize              = WindowSize
        self.WindowOverlap           = WindowOverlap
        self.SNRThreshold            = SNRThreshold
        self.SmoothingStDevThreshold = SmoothingStDevThreshold
        self.SmoothingKernelSize     = SmoothingKernelSize
        self.SmoothingIterations     = SmoothingIterations

        return self

    def ComputeVelocityField(self: PIVAnalyzer, CurrentFrame: np.ndarray) -> np.ndarray:
        """
        ComputeVelocityField

        This function...

        CurrentFrame:
            ...

        Return (np.ndarray):
            ...
        """

        #   ...

        return None

    def _DoPIV(self: PIVAnalyzer, PreviousFrame: np.ndarray, CurrentFrame: np.ndarray) -> np.ndarray:
        """
        _DoPIV

        This function...

        PreviousFrame:
            ...
        CurrentFrame:
            ...

        Return (np.ndarray):
            ...
        """

        #   ...

        return None

#   Define the globals to set by the command-line arguments
LogWriter: Logger = Logger()
Config: Configuration = Configuration(LogWriter=LogWriter)
VelocimetryAnalyzer: PIVAnalyzer = PIVAnalyzer(LogWriter=LogWriter)

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
    AlignmentFraction: float = float(len(ShiftedOrientations[abs(ShiftedOrientations) < AngularStDev]) / RodCount)

    Results = (RodCount, AlignmentFraction, AngularMean, AngularStDev)
    LogWriter.Println(f"{RodCount=:}, {AlignmentFraction=:.4f}, {AngularMean=:.4f}, {AngularStDev=:.4f}")

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
    GaussianBlurKernelSize: int = 3
    AdaptiveThresholdKernelSize: int = 9
    AdaptiveThresholdConstant: int = 5

    #   Assert that the image is in greyscale, single-channel format
    Image = MyUtils.BGRToGreyscale(Image)

    #   Apply a Gaussian blur to try to remove the speckling shot-noise from the image
    # Image = cv2.GaussianBlur(Image, (GaussianBlurKernelSize, GaussianBlurKernelSize) ,0)

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
    MinimumComponentArea: int = 10
    MaximumComponentArea: int = 50

    #   Actually extract out the set of components from the image we're interested in working with
    NumberOfComponents, LabelledImage, Stats, _ = cv2.connectedComponentsWithStats(Image, connectivity=ComponentConnectivity)

    #   Iterate over the set of components, and remove those which do not satisfy the requirements of being a rod
    FilteredLabels: typing.List[int] = []
    for ComponentId in range(1, NumberOfComponents):

        Area = Stats[ComponentId, cv2.CC_STAT_AREA]

        if not ( MinimumComponentArea <= Area <= MaximumComponentArea ):
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

    EllipseAreaBounds: typing.Tuple[int, int] = (10, 60)
    EllipseAxisLengthBounds: typing.Tuple[int, int] = (1, 40)

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
    AlignmentMethod = Arguments[0]
    if ( AlignmentMethod == "sobel" ):
        Image = SobelAlignmentMethod(Image, Arguments)
    else:
        Image = ComponentAlignmentMethod(Image, Arguments)
    return Image, True

def ComponentAlignmentMethod(Image, Arguments):

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

    if ( len(Angles.MeanAngles) == 0 ):
        Angles.MeanAngles = np.array([MeanAngle])
        Angles.AngularStDevs = np.array([AngularStDev])
    else:
        Angles.MeanAngles = np.append(Angles.MeanAngles, MeanAngle)
        Angles.AngularStDevs = np.append(Angles.AngularStDevs, AngularStDev)

    plt.clf()
    plt.hist(Orientations, bins=180, density=True)
    plt.show(block=False)
    plt.waitforbuttonpress(0.01)

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

    ScaleFactor = 1.0
    Image = MyUtils.UniformRescaleImage(MyUtils.BGRToGreyscale(Image), ScaleFactor)

    Altered_Image = MyUtils.GammaCorrection(Image, Gamma=1, Minimum=0, Maximum=255)
    Altered_Image = cv2.GaussianBlur(Altered_Image,(3,3),0)
    Altered_Image = cv2.adaptiveThreshold(Altered_Image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,9,5)
    Altered_Image = cv2.morphologyEx(Altered_Image, cv2.MORPH_DILATE, cv2.getStructuringElement(cv2.MORPH_RECT, (7,7)))

    ForegroundMask = MyUtils.GammaCorrection(Altered_Image.copy(), Gamma=1, Minimum=0, Maximum=1)
    Altered_Image = ForegroundMask * Image

    SobelBlockSize = 3
    Altered_Image = Altered_Image.astype(np.int16)
    Gx = cv2.Sobel(Altered_Image, cv2.CV_16S, 0, 1, None, ksize=SobelBlockSize)
    Gy = cv2.Sobel(Altered_Image, cv2.CV_16S, 1, 0, None, ksize=SobelBlockSize)

    EdgeDirection = np.arctan2(Gy.astype(np.float64), Gx.astype(np.float64))

    Gradient = np.zeros((Gy.shape[0], Gy.shape[1], 3), dtype=np.uint8)
    Gradient[:,:,0] = MyUtils.ConvertTo8Bit(EdgeDirection)
    Gradient[:,:,1] = 255
    Gradient[:,:,2] = MyUtils.ConvertTo8Bit(np.hypot(Gx, Gy))
    Gradient = cv2.cvtColor(Gradient, cv2.COLOR_HSV2BGR)

    EdgeDirection[Gx == 0] = np.NaN
    EdgeDirection[Gy == 0] = np.NaN

    ax = plt.subplot(1,1,1)
    ax.hist(EdgeDirection.flatten() * (180 / np.pi), bins=72, density=True)
    plt.waitforbuttonpress(0.01)
    plt.clf()

    Gradient = MyUtils.UniformRescaleImage(Gradient, 1.0 / ScaleFactor)
    return Gradient

#   Main
#       This is the main entry point of the script.
def main() -> None:

    if ( Config.VideoFilename is not None ):

        Video: vwr.VideoReadWriter = vwr.VideoReadWriter(readFile=Config.VideoFilename, logger=LogWriter)

        FrameCount: int = (Video.EndFrameIndex - Video.StartFrameIndex)
        Tracker: AngleTracker = AngleTracker()
        Tracker.Times = np.linspace(Video.StartFrameIndex / Config.VideoFrameRate, (Video.EndFrameIndex-1) / Config.VideoFrameRate, FrameCount)
        # Video.PrepareWriter(None, FrameRate=50, Resolution=(480, 720), TopLeft=(500, 0))
        Video.ProcessVideo(PlaybackMode=vwr.PlaybackMode_NoDelay, Callback=_ComputeAlignmentFraction, CallbackArgs=[Config.AlignmentMethod, Tracker])
        #   ...
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
