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
#   ...

class Configuration():
    """
    Configuration

    This class...
    """

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

        if ( self.IsVideo ):
            if ( self.VideoFrameRate < 0 ):
                self._LogWriter.Errorln(f"Invalid frame-rate [ {self.VideoFrameRate}fps ].")
                Validated = False

        #   ...

        return Validated

class AngleTracker():
    """
    AngleTracker

    This class...
    """

    Times: np.ndarray
    MeanAngles: np.ndarray
    AngularStDevs: np.ndarray

    def __init__(self: AngleTracker) -> None:
        """
        Constructor

        This function...
        """

        self.Times = np.array([])
        self.MeanAngles = np.array([])
        self.AngularStDevs = np.array([])

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

def ComputeAlignmentMetric(Image: np.ndarray) -> float:
    """
    ComputeAlignmentMetric

    This function...

    Image:
        ...

    Return (float):
        ...
    """

    AlignmentScore: float = 0.0

    #   ...

    return AlignmentScore



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

    # AlignmentScore: float = ComputeAlignmentMetric(Image)

    #   You can apply any transformations or operations on "Image" here 260-283
    #   ...
    AlignmentMethod = Arguments[0]
    if ( AlignmentMethod == "sobel" ):
        Image = SobelAlignmentMethod(Image, Arguments)
    else:
        Image = ComponentAlignmentMethod(Image, Arguments)
    return Image, True

def ComponentAlignmentMethod (Image, Arguments):

    Angles: AngleTracker = Arguments[1]

    ScaleFactor: float = 1.0
    Image = MyUtils.UniformRescaleImage(Image, ScalingFactor=ScaleFactor)

    Image=MyUtils.BGRToGreyscale(Image)
    Image = cv2.GaussianBlur(Image,(3,3),0)
    Image = cv2.adaptiveThreshold(Image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,9,5)
    NumberOfComponents, labels, stats, centroids = cv2.connectedComponentsWithStats (Image, connectivity=8)

    FilteredImage = np.zeros_like(Image)
    RodCount = 0
    for Component in range(1, NumberOfComponents):

        X, Y, W, H, Area = stats[Component, cv2.CC_STAT_LEFT], stats[Component, cv2.CC_STAT_TOP], stats[Component, cv2.CC_STAT_WIDTH], stats[Component, cv2.CC_STAT_HEIGHT], stats[Component, cv2.CC_STAT_AREA]
        keepArea = 10 < Area and Area < 100
        if keepArea:
            FilteredImage[labels == Component] = 255
            RodCount = RodCount + 1

    # LogWriter.Println(f"Found a total of {RodCount} rods from {NumberOfComponents} components.")
    Contours, Hierarchy = cv2.findContours (FilteredImage, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    FilteredImage = MyUtils.GreyscaleToBGR(FilteredImage)
    # FilteredImage = cv2.drawContours (FilteredImage, Contours, -1, (255,0,0),1)

    Orientations = []
    for Contour in Contours:
        if ( len(Contour) < 5 ):
            continue
        ellipse = cv2.fitEllipseDirect(Contour)
        #   Major and Minor are the lengths of the major and minor *diameters* of the ellipse,
        #   not the *radii*
        (x, y), (Major, Minor), Angle = ellipse
        EllipseArea = math.pi * Major * Minor * (1/4)
        if not ( 1 < EllipseArea <= 60 ) or ( math.isnan(EllipseArea) ):
            continue
        if not ( 1 < Major <= 40 ):
            continue
        if not ( 1 < Minor <= 40 ):
            continue

        #   Maybe we can filter these ellipses based off their axes lengths and their enclosed area?
        FilteredImage = cv2.ellipse(FilteredImage, ellipse, (Angle, 255, 255), 1)

        Orientations.append(Angle)

    Orientations = np.array(Orientations)
    MeanAngle: float = scipy.stats.circmean(Orientations, high=180, low=0)
    AngularStDev: float = scipy.stats.circstd(Orientations, high=180, low=0)
    StDev: float = np.std(Orientations)

    if ( len(Angles.MeanAngles) == 0 ):
        Angles.MeanAngles = np.array([MeanAngle])
        Angles.AngularStDevs = np.array([StDev])
        pass
    else:
        Angles.MeanAngles = np.append(Angles.MeanAngles, MeanAngle)
        Angles.AngularStDevs = np.append(Angles.AngularStDevs, StDev)

    LogWriter.Println(f"Mean Angle: {MeanAngle:.2f}degrees - Angular StDev: {AngularStDev:.4f}degrees - StDev: {StDev:.4f}")

    plt.clf()
    plt.hist(Orientations, bins=180, density=True)
    plt.show(block=False)
    plt.waitforbuttonpress(0.01)

    FilteredImage = cv2.cvtColor(FilteredImage, cv2.COLOR_HSV2BGR)
    FilteredImage = MyUtils.UniformRescaleImage(FilteredImage, ScalingFactor=(1.0/ScaleFactor))

    return FilteredImage

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

        plt.errorbar(x=Tracker.Times, y=Tracker.MeanAngles, yerr=Tracker.AngularStDevs, capsize=5, capthick=2)
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
