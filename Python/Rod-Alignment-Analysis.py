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
import typing
#   ...

#   Import the necessary third-part modules
import cv2
from openpiv import tools, pyprocess, validation, filters
import numpy as np
import matplotlib.pyplot as plt

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

    VideoFilename: str

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
    Image = MyUtils.BGRToGreyscale(Image)
    _, Mask = cv2.threshold(Image, 50, 255, cv2.THRESH_BINARY)
    Image = Image & Mask

    AlignmentScore: float = ComputeAlignmentMetric(Image)

    #   You can apply any transformations or operations on "Image" here 260-283
    #   ...
    #Altered_Image = cv2.flip(Image, 1)
    #Altered_Image = cv2.threshold(Image, 150, 255, cv2.THRESH_BINARY_INV)
    #Altered_Image = cv2.adaptiveThreshold(Image,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV,9,4)
    Altered_Image = MyUtils.GammaCorrection(Image, Gamma=1, Minimum=0, Maximum=255)
    Altered_Image = cv2.GaussianBlur(Altered_Image,(3,3),0)
    Altered_Image = cv2.adaptiveThreshold(Altered_Image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,9,5)
    # Altered_Image = cv2.fastNlMeansDenoising(Altered_Image, None, 6, 7, 21)

    Altered_Image =cv2.GaussianBlur(Altered_Image, (9, 9), 0)

    Altered_Image = Altered_Image.astype(np.int16)
    Gx = cv2.Sobel(Altered_Image, cv2.CV_16S, 0, 1, None, ksize=3)
    Gy = cv2.Sobel(Altered_Image, cv2.CV_16S, 1, 0, None, ksize=3)

    # EdgeDirection = np.arctan2(Gy.astype(np.float64), Gx.astype(np.float64))
    EdgeDirection = np.arctan(Gy.astype(np.float64) / Gx.astype(np.float64))

    Gradient = np.zeros((Gy.shape[0], Gy.shape[1], 3), dtype=np.uint8)
    Gradient[:,:,0] = MyUtils.ConvertTo8Bit(EdgeDirection)
    Gradient[:,:,1] = 255
    Gradient[:,:,2] = MyUtils.ConvertTo8Bit(np.hypot(Gx, Gy))
    Gradient = cv2.cvtColor(Gradient, cv2.COLOR_HSV2BGR)

    EdgeDirection[Gx == 0] = np.NaN
    EdgeDirection[Gy == 0] = np.NaN

    ax = plt.subplot(1,1,1, projection='polar')
    ax.hist(EdgeDirection.flatten(), bins=36, density=True)
    plt.waitforbuttonpress(0.01)
    plt.clf()

    return Gradient, True

    #   ...


    return Altered_Image, True

#   ...

#   Main
#       This is the main entry point of the script.
def main() -> None:

    #   First, open the video file and prepare to start analyzing the frames to compute
    #       the velocity field
    Video: vwr.VideoReadWriter = vwr.VideoReadWriter(readFile=Config.VideoFilename, logger=LogWriter)

    #   Now, with the velocity field computed for each time-point within the source video,
    #       start processing these to extract out rotation rate information
    #   ...

    #   With the time-series rotation data, compute the alignment time by looking for
    #       key time-points in the "average amount of rotation" seen in the video.
    Video.SourceFrameRate = 50
    Video.PrepareWriter(None, FrameRate=50, Resolution=(480, 720), TopLeft=(500, 0))
    Video.ProcessVideo(PlaybackMode=vwr.PlaybackMode_NormalSpeed, Callback=_ComputeAlignmentFraction)
    #   ...

    #   ...

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
    Flags.add_argument("--filename", dest="Filename", metavar="file-path", type=str, required=True, help="The video file showing rod rotation to be processed.")
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
