#!/usr/bin/env python3

##      Author:     Joseph Sadden
##      Date:       29th November, 2021

#   Import the local modules and classes required.
from __future__ import annotations
import datetime
import random
import tempfile
import time
import traceback

from . import Logger
from . import Utils

#   Import the standard library packages and classes required.
import os
from typing import Any, Iterator, Tuple, List, Callable

#   Import the third-part modules and classes required.
import cv2
import numpy as np

#   Define the constants for the playback mode to use when "writing" a video
#   file out.
_PlaybackMode_START: int = 0
PlaybackMode_SingleStep: int = _PlaybackMode_START
PlaybackMode_NormalSpeed: int = PlaybackMode_SingleStep + 1
PlaybackMode_NoDisplay: int = PlaybackMode_NormalSpeed + 1
PlaybackMode_NoDelay: int = PlaybackMode_NoDisplay + 1
_PlaybackMode_END: int = PlaybackMode_NoDelay

#   Define the error conditions potentially returned by the WriteFrame()
#       function
ErrWriteFrame_None: int = 0
ErrWriteFrame_NoOutputVideo: int = 1
ErrWriteFrame_Quit: int = 2
ErrWriteFrame_SaveFailed: int = 3
ErrWriteFrame_SizeMismatch: int = 4

class VideoReadWriter(Iterator):
    """
        :class:`VideoReadWriter`

        This is a simple encapsulation of an OpenCV2 VideoCapture and
        VideoWriter, such that it more readily exposes the behaviours and
        utilities necessary for the video processing I require for my thesis
        research.
    """

    ##  Public Class Attributes
    SourceFrameRate:    float   #   The frame-rate of the source video file, as defined in the
                                #       video file codec.
    SourceWidth:        int     #   The width, in pixels, of the source video file.
    SourceHeight:       int     #   The height, in pixels, of the source video file.
    SourceNumFrames:    int     #   The total number of video frames in the source video file.
    SourceFrameIndex:   int     #   The index number of the "current" frame being operated on from
                                #       the source video file.

    OutputFrameRate:    float           #   The frame-rate of the output video file.
    OutputWidth:        int             #   The width, in pixels, of the output video file.
    OutputHeight:       int             #   The height, in pixels, of the output video file.
    OutputFourCC:       str             #   The four-character-code defining the output codec to use when writing the
                                        #       Output video file.
    OutputCropTopLeft:  Tuple[int, int] #   A tuple containing the pixel location from the *source* video file
                                        #       to use as the top-left pixel when cropping to form the output video file.
                                        #       Set to (0, 0) to ignore.
    PrintProgress: bool                 #   Print out the frame index for each frame processed during the ProcessVideo() function.
    OutputRotationAngle: float          #   The counter-clockwise rotation angle to apply to each frame of the video being processed,
                                        #       measured in degrees.
    StartFrameIndex: int                #   The first 0-indexed frame to process from the source video
    EndFrameIndex: int                  #   The final 0-indexed frame to process from the source video

    ##  Private Class Attributes

    _LastFrameWrittenTimestamp: float   #   The unix timestamp for when the last frame was written.
                                        #       Used to help ensure the output frame-rate doesn't exceed
                                        #       the user-defined limit (if provided).

    _OutputWindowTopmost: bool          #   Boolean value for whether or not to force any created windows to be presented
                                        #       on top of all other windows or left with the standard window ordering.

    #   Logger, the log writer to use to write any log messages.
    _LogWriter: Logger.Logger

    #   SourceFilename, the raw filename to read video data from, if not None
    _SourceFilename: str

    #   SourceVideo, the cv2.VideoCapture associated with the given
    #   SourceFilename
    _SourceVideo: cv2.VideoCapture

    #   CurrentSourceFrame, the most recently read frame from the source video.
    _CurrentSourceFrame: np.ndarray

    #   OutputFilename, the filename to write video data to, if not None
    _OutputFilename: str

    #   OutputVideo, the cv2.VideoWriter associated with the given
    #   OutputFilename
    _OutputVideo: cv2.VideoWriter

    #   Private members related to printing out the processing rate during ProcessVideo().
    _PreviousFrameTimestamps: List[float]
    _LastCached: float
    _ProcessingRate: float
    _Eta: str

    ##  Magic Methods
    def __init__(self, readFile: str = None, writeFile: str = None, logger: Logger.Logger = Logger.Logger(Prefix="VideoReadWriter"), progress: bool = True) -> None:
        """
        Constructor:

        This function constructs and returns a basic VideoReadWriter class,
        ensuring all of the class variables are set to a usable set of values.

        readFile:
            The filename to the source video file to read from.
        writeFile:
            The filename to write the output video file to.
        logger:
            The Logger() to use for writing out logging information during use.
        progress:
            A boolean indicating whether to print out the processing message to the Logger
            during ProcessVideo(). Set to False when piping the output to a log file generally.
        sequential:
            ...
        """

        self._LogWriter             = logger
        self._SourceFilename        = readFile
        self._SourceVideo           = None
        self._OutputFilename        = writeFile
        self._OutputVideo           = None
        self._OutputWindowTopmost   = False

        self._PreviousFrameTimestamps   = []
        self._LastCached                = 0.0
        self._LastFrameWrittenTimestamp = 0.0

        self.SourceFrameRate        = None
        self.SourceHeight           = None
        self.SourceWidth            = None
        self.SourceNumFrames        = None
        self.SourceFrameIndex       = None

        self.OutputFrameRate        = None
        self.OutputWidth            = None
        self.OutputHeight           = None
        self.OutputFourCC           = None
        self.OutputCropTopLeft      = (0, 0)
        self.OutputRotationAngle    = None

        self.StartFrameIndex        = None
        self.EndFrameIndex          = None

        self.PrintProgress          = progress

        if ( self._SourceFilename is not None ):
            if ( not self._openSourceVideo() ):
                raise AttributeError(f"Failed to initialize VideoReadWriter; could not open source video file [ {readFile} ].")
            self.SetFrameRange()

        if ( self._OutputFilename is not None ):
            self.PrepareWriter(OutputFilename=writeFile)

        return

    def __iter__(self) -> Iterator:
        """
        __iter__

        Return an iterable from the instance of the class. Since this class is
        itself an iterator, this simply initializes the output video and returns
        itself.

        Return (Iterator):
            self
        """

        self.Seek(self.StartFrameIndex)

        if ( self.SourceNumFrames < self.EndFrameIndex ):
            self.EndFrameIndex = self.SourceNumFrames

        self._resetProcessingRate()

        return self.PrepareWriter(OutputFilename=self._OutputFilename, FrameRate=self.OutputFrameRate, FourCC=self.OutputFourCC, Resolution=(self.OutputWidth, self.OutputHeight), TopLeft=self.OutputCropTopLeft)

    def __next__(self) -> np.ndarray:
        """
        __next__

        This function provides the necessary magic method to implement an Iterable
        interface. This returns the 'next' frame read from the source video.

        Return (np.ndarray):
            The 'next' frame from the source video.
            If this VideoReadWriter has been configured with a rotation angle,
            the frame will be rotated as specified.
        """

        if ( self._SourceVideo is None ):
            if ( not self._openSourceVideo() ):
                self._LogWriter.Errorln("Failed to open source video for reading during ReadFrame().")
                self._closeSourceVideo()
                raise StopIteration

        if (( self.SourceFrameIndex >= self.EndFrameIndex ) or ( self.SourceFrameIndex >= self.SourceNumFrames )):
            raise StopIteration

        Success, self._CurrentSourceFrame = self._SourceVideo.read()
        if ( not Success ):
            if ( self.SourceFrameIndex != self.SourceNumFrames ):
                self._LogWriter.Errorln(f"Failed to read next frame from video file [ {self._SourceFilename} ].")
            else:
                self._LogWriter.Println(f"Reached end of video file [ {self._SourceFilename} ].")
            self._closeSourceVideo()
            raise StopIteration

        if ( self.PrintProgress ):
            ProcessingRate, ETA = self._computeProcessingRate()
            if ( not self._LogWriter.WritesToFile() ):
                self._LogWriter.Write(f"Successfully read frame number [ {self.SourceFrameIndex + 1}/{self.EndFrameIndex} ]. ({ProcessingRate:.2f}fps, ETA: {ETA})\r")

        self.SourceFrameIndex += 1
        self._CurrentSourceFrame = self._rotateFrame(self._CurrentSourceFrame)
        return self._CurrentSourceFrame

    def __getitem__(self, Index: int = 0) -> np.ndarray:
        """
        __getitem__

        This returns the frame from the source video at the specified 0-indexed
        frame index. This implements the 'operator[]' to allow retrieving frames from the
        source video using slice notation.

        Index:
            The 0-indexed frame to return. This must be within the range [StartFrame, EndFrame),
            and within the total number of frames of the video.
            This does not support negative indices to refer to reverse indexing.

        Return (np.ndarray):
            The video frame from the source video file at the requested Frame Index.
        """

        if isinstance(Index, slice):
            start, stop, step = Index.start, Index.stop, Index.step
            if start is None:
                start = 0
            if stop is None:
                stop = len(self)

            return [self[ii] for ii in range(start, stop, step or 1)]
        elif isinstance(Index, int):
            Index = int(round(Index))
            self.Seek(Index)

            Success, self._CurrentSourceFrame = self._SourceVideo.read()
            if ( not Success ):
                raise RuntimeError(f"Failed to read frame [ {Index} ] from source video!")

            if ( self.PrintProgress ):
                if ( not self._LogWriter.WritesToFile() ):
                    self._LogWriter.Write(f"Successfully read frame number [ {self.SourceFrameIndex + 1}/{self.EndFrameIndex} ].{' ' * 30}\r")

            self._CurrentSourceFrame = self._rotateFrame(self._CurrentSourceFrame)

            return self._CurrentSourceFrame
        else:
            raise TypeError(f"Invalid argument type for [ Index ].")

    def __len__(self) -> int:
        """
        __len__

        This function...

        Return (int):
            ...
        """

        if ( self.StartFrameIndex is not None ) and ( self.EndFrameIndex is not None):
            return self.EndFrameIndex - self.StartFrameIndex

        if ( self.SourceNumFrames is not None ):
            return self.SourceNumFrames

        return 0

    def __del__(self) -> None:
        """
        Destructor:

        Safely close down the VideoReadWriter, closing the input and output
        videos (if opened).
        """
        try:
            self._closeSourceVideo()
            self._closeOutputVideo()
        except:
            pass

        return

    ##  Private Class Methods
    def _openSourceVideo(self) -> bool:
        """
        openVideo:

        This function will open the given source video file, preparing it to be
        read through.  This will also extract and populate the public class
        members related to the source video file properties.

        Return (bool):
            True if successfully opened and prepared video file for reading,
            False otherwise.
        """

        if ( self._SourceVideo is not None ):
            return True

        #   Make sure there's a proper filename given to try to open
        if ( self._SourceFilename is None )  or ( self._SourceFilename == "" ):
            self._LogWriter.Errorln("Source Filename has not been set, cannot open video file.")
            return False

        #   Make sure the file exists
        if ( not os.path.exists(self._SourceFilename) ):
            self._LogWriter.Errorln(f"Source Filename [ {self._SourceFilename} ] does not exist.")
            return False

        #   Create the video capture wrapper object from OpenCV we actually
        #   operate on.
        self._SourceVideo = cv2.VideoCapture(self._SourceFilename, cv2.CAP_ANY)

        #   Extract a handful of helpful properties from the source video file
        self.SourceFrameRate    =   self._SourceVideo.get(cv2.CAP_PROP_FPS)
        self.SourceNumFrames    =   int(self._SourceVideo.get(cv2.CAP_PROP_FRAME_COUNT))
        self.SourceHeight       =   int(self._SourceVideo.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.SourceWidth        =   int(self._SourceVideo.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.SourceFrameIndex   =   0

        if ( self.SourceNumFrames < 0 ):
            self.SourceNumFrames = 1

        self._LogWriter.Println(f"Successfully opened video file [ {self._SourceFilename} ] for reading.")

        return True

    def _closeSourceVideo(self) -> None:
        """
        closeSourceVideo:

        This function will safely close the source video file, if it has been
        opened. This will leave the VideoReadWriter in a state where it can
        either be destructed safely, or prepared to read a new video file.
        """

        #   If the video capture has been initialized, release it, set the
        #   member value to None, and print a log message.
        if ( self._SourceVideo is not None ):
            self._SourceVideo.release()
            self._SourceVideo = None
            self._LogWriter.Println(f"Closed video file [ {self._SourceFilename} ] for reading.")

        #   Clear the local public parameters about the video file, now that
        #   it's closed.
        self.SourceFrameRate    =   None
        self.SourceWidth        =   None
        self.SourceHeight       =   None
        self.SourceNumFrames    =   None
        self.SourceFrameIndex   =   None

        return

    def _openOutputVideo(self) -> bool:
        """
        _openOutputVideo:

        This function will prepare the output video writer, as well as the
        output video file (if necessary) as requested for this VideoReadWriter.

        Return (bool):
            True if the output video is ready to be used, False otherwise.
        """

        #   Check if we've already opened a video writer
        if ( self._OutputVideo is not None ):
            return True

        #   Make sure there's been a filename given to write to
        if ( self._OutputFilename is None ) or ( self._OutputFilename == "" ):
            self._OutputVideo = None
            return True

        #   Ensure the path to the output file exists, including any
        #   intermediate directories.
        if ( os.path.dirname(self._OutputFilename) != "" ):
            os.makedirs(os.path.dirname(self._OutputFilename), mode=0o755, exist_ok=True)

        #   Create and initialize a new cv2.VideoWriter
        self._OutputVideo = cv2.VideoWriter(self._OutputFilename, cv2.VideoWriter_fourcc(*self.OutputFourCC), self.OutputFrameRate, (int(self.OutputWidth), int(self.OutputHeight)))
        if ( not self._OutputVideo.isOpened() ):
            self._LogWriter.Errorln(f"Failed to open video file [ {self._OutputFilename} ] for writing.")
            return False

        self._OutputVideo.set(cv2.VIDEOWRITER_PROP_QUALITY, 100)

        self._LogWriter.Println(f"Successfully opened video file [ {self._OutputFilename} ] for writing.")

        return True

    def _closeOutputVideo(self) -> None:
        """
        _closeOutputVideo:

        This function will safely close down the output video file, if it has
        been opened, leaving this VideoReadWriter ready to either be destructed
        or prepared to write to a new output file.
        """

        if ( self._OutputVideo is not None ):
            self._OutputVideo.release()
            self._LogWriter.Println(f"Closed video file [ {self._OutputFilename} ] for writing.")
            self._OutputVideo = None

        self.OutputWidth = None
        self.OutputHeight = None
        self.OutputFourCC = None
        self.OutputFrameRate = None
        self.OutputCropTopLeft = (0, 0)

        return

    def _crop(self, Frame: np.ndarray) -> np.ndarray:
        """
        _crop:

        This function will perform a basic cropping of the given video frame,
        based on the cropping dimensions and location given when the
        VideoReadWriter was prepared. This allows for simpler cropping of videos
        while they are being processed.

        Frame:
            This is a 3D list of pixel values, where the first dimension is
            image height, second is image width, and third is pixel depth (RGB
            or similar).

        Return:
            The frame, cropped such that the pixel (0, 0) corresponds to the
            location self._OutputCropTopLeft.
        """

        #   If the argument isn't actually a frame, just ignore it.
        if ( Frame is None ):
            self._LogWriter.Errorln('Cannot crop "None" video frame.')
            return None

        if ( self.OutputCropTopLeft is None ):
            return Frame

        #   If the resolution is the same between source and output videos,
        #   don't do anything to the frame
        if ( self.OutputWidth == self.SourceWidth ) and ( self.OutputHeight == self.SourceHeight ):
            return Frame

        Left = self.OutputCropTopLeft[0]
        Top = self.OutputCropTopLeft[1]

        if ( Left is None ) or ( Top is None ):
            return Frame

        Right = Left + self.OutputWidth
        Bottom = Top + self.OutputHeight

        #   Actually perform the cropping.
        if ( len(Frame.shape) == 2 ):
            return Frame[Top:Bottom, Left:Right]
        elif ( len(Frame.shape) == 3 ):
            return Frame[Top:Bottom, Left:Right, :]
        else:
            raise ValueError(f"Frame was in the wrong or unknown colour-space. Shape was: [ {Frame.shape} ]...")

    def _rotateFrame(self, Frame: np.ndarray) -> np.ndarray:
        """
        _rotateFrame:

        This function will rotate the given video frame by the angle specified by the SetRotationAngle()
        method. This is called during frame-retrieval, prior to being returned to the caller.

        Frame:
            The raw frame as read from the underlying cv2.VideoCapture, prior to being rotated.

        Return (np.ndarray):
            The frame, rotated by the configured angle. Border pixels are handled by
            the option cv2.BORDER_REPLICATE (https://docs.opencv.org/3.4/d2/de8/group__core__array.html#ga209f2f4869e304c82d07739337eae7c5)
        """

        if ( Frame is None ):
            return None

        if ( self.OutputRotationAngle is None ):
            return Frame

        if ( self.OutputRotationAngle == 0.0 ):
            return Frame

        FrameHeight, FrameWidth = Frame.shape[:2]

        #   Compute the centre location of the image frame, as a (width, height) tuple
        #       as half the image size in each dimension.
        Centre = ( FrameWidth / 2.0, FrameHeight / 2.0 )

        #   Compute the rotation matrix to apply to the image to perform the rotation
        #       by the desired angle.
        FrameRotationMatrix = cv2.getRotationMatrix2D(Centre, self.OutputRotationAngle, scale=1.0)

        #   Actually apply the rotation matrix to the frame, returning the resulting rotated image.
        return cv2.warpAffine(Frame, FrameRotationMatrix, (FrameWidth, FrameHeight), borderMode=cv2.BORDER_REPLICATE)

    def _prepareVideoPlayback(self, WindowName: str, QuitKeys: str, PlaybackMode: int) -> Tuple[str, List[int]]:
        """
        _prepareVideoPlayback:

        This function will prepare the necessary additional functionality to
        actually playback a video file.  This will create a window to display
        the frames, set the keys which will allow quitting out of the playback,
        and set the playback mode.

        WindowName:
            The name of the cv2.namedWindow() created window to display the
            video in. Should be unique and meaningful for the video being
            displayed. Typically the filename of the video.
        QuitKeys:
            A string containing each of the letters which, if received, will
            quit out of the video playback.
        PlaybackMode:
            The playback mode to use for the video. One of:
                PlaybackMode_SingleStep: Single-Step through the video,
                requiring a keypress to advance one frame.
                PlaybackMode_NormalSpeed: Play back the video at the frame-rate
                defined in the video codec. No interaction is necessary.
                PlaybackMode_NoDisplay: Iterate over the video frames without
                displaying them, as fast as possible.
                PlaybackMode_NoDelay: Similar to NoDisplay; iterate over the
                frames as fast as possible, while still displaying them. Similar
                to NormalSpeed, but ignores the requirement of matching a
                desired nominal framerate.

        Returns:
            str:
                The name of the window which was actually created, in case it is
                different than the input argument.
            List[int]:
                A list of the key-codes to listen for to quit the video
                playback.
        """

        #   Make sure there's a video to read from
        if ( not self._openSourceVideo() ):
            self._LogWriter.Errorln("Failed to prepare video playback, could not open source video file for reading.")
            return None, None

        #   Set up the key codes to allow safely quitting out of the video
        #   playback
        QuitKeys = [ ord(x) for x in QuitKeys ]

        #   Attempt to open up the output video file to write to.
        #       If no video file is provided, this just configures
        #       the VideoReadWriter to only write to the (possible)
        #       output display and not a file
        if ( self._openOutputVideo() == False ):
            self._LogWriter.Errorln("Failed to prepare video playback, could not open output video file for writing.")
            return None, None

        #   If we're not displaying the video as we process it, just return the set
        #       of keys to stop playback and don't open up a window to display it.
        if ( PlaybackMode == PlaybackMode_NoDisplay ):
            #   Otherwise, still allow for quitting the playback with the
            #   requested keys.
            self._LogWriter.Println("Successfully prepared video playback for NoDisplay mode.")
            return None, QuitKeys

        #   Otherwise, prepare the window to use to show the video frames
        else:
            #   If the window name is not given, derive it from the source
            #   filename of the video being displayed.
            if ( WindowName is None ) or ( WindowName == "" ):
                WindowName = os.path.basename(self._SourceFilename)

            #   Prepare the window, and set it to be the top-most window, if specified.
            cv2.namedWindow(WindowName, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_EXPANDED)
            if ( self._OutputWindowTopmost ):
                cv2.setWindowProperty(WindowName, cv2.WND_PROP_TOPMOST, 1)

            self._LogWriter.Println(f"Successfully prepared video playback for window [ {WindowName} ].")
            return WindowName, QuitKeys

    def _computeProcessingRate(self) -> Tuple[float, str]:
        """
        _computeProcessingRate

        Compute and return the speed (frames/second) at which the current self.ProcessVideo()
        call is occurring at.

        Return (Tuple[float, str]):
            [0] float   - The processing rate, in units of frames per second
            [1] str     - A string containing the estimated completion time.
        """
        AveragingWidth: int = 100
        CurrentTimeStamp = time.time()

        if ( len(self._PreviousFrameTimestamps) <= AveragingWidth ):
            self._PreviousFrameTimestamps.append(time.time())
            if ( len(self._PreviousFrameTimestamps) == 1 ):
                return (0.0, "Unknown")
        else:
            self._PreviousFrameTimestamps.pop(0)
            self._PreviousFrameTimestamps.append(CurrentTimeStamp)

        #   If the time since we last updated the processing rate is longer than 1 second,
        #       recalculate and report the new values, otherwise use the cached ones.
        if ((CurrentTimeStamp - self._LastCached) > 1):
            self._ProcessingRate = np.nanmean(np.diff(self._PreviousFrameTimestamps, 1))
            ETA: datetime.timedelta = datetime.datetime.fromtimestamp(self._PreviousFrameTimestamps[-1] + ((self.EndFrameIndex - self.SourceFrameIndex) * self._ProcessingRate)) - datetime.datetime.fromtimestamp(self._PreviousFrameTimestamps[-1])
            self._Eta = str(ETA - datetime.timedelta(microseconds=ETA.microseconds))
            self._LastCached = CurrentTimeStamp

        return (1.0 / self._ProcessingRate, self._Eta)

    def _resetProcessingRate(self) -> None:
        """
        _resetProcessingRate

        Reset the buffer of previous frame timestamps used in computing the FIR-filtered
        processing rate.

        Return (None):
            None, the internal list is cleared.
        """

        self._PreviousFrameTimestamps   = []
        return

    ##  Static Class Methods
    @staticmethod
    def FromImageSequence(Images: np.ndarray, tempFile: str = None, writeFile: str = None, logger: Logger.Logger = Logger.Logger(Prefix="FromImageSequence"), progress: bool = True) -> VideoReadWriter:
        """
        FromImageSequence

        This function...

        Images:
            ...
        tempFile:
            ...
        writeFile:
            ...
        logger:
            ...
        progress:
            ...

        Return (VideoReadWriter):
            ...

        """

        with tempfile.NamedTemporaryFile() if ( tempFile is None ) else open(tempFile, "+wb") as OutFile:
            WriteFile: str = os.path.splitext(OutFile.name)[0] + ".avi"
            V: VideoReadWriter = VideoReadWriter(None, WriteFile, Logger.Logger(Prefix="Image Sequence to Video Converter"), True)
            V.PrepareWriter(OutputFilename=WriteFile, FourCC='MJPG', Resolution=Images[0].shape)
            for Image in Images:
                V.WriteFrame(Utils.GreyscaleToBGR(Utils.ConvertTo8Bit(Image)))
            V._closeOutputVideo()

            return VideoReadWriter(readFile=WriteFile, writeFile=writeFile, logger=logger, progress=progress)

    ##  Public Class Methods
    def Play(self, WindowName: str = None) -> bool:
        """
        Play

        This function...

        WindowName:
            ...

        Return (bool):
            ...
        """

        return self.ProcessVideo(
            WindowName=WindowName,
            PlaybackMode=PlaybackMode_NormalSpeed,
        )

    def ProcessVideo(self, WindowName: str = None, QuitKeys: str = "qQ", PlaybackMode: int=PlaybackMode_NoDisplay, Callback: Callable[[np.ndarray, List], Tuple[np.ndarray, bool]] = None, CallbackArgs: List = [], StartFrame: int = None, EndFrame: int = None) -> bool:
        """
        ProcessVideo:

        This function will "Process" a given video file. How exactly this
        operates depends on the PlaybackMode and Callback functions, if
        provided.  This will iterate through the frames of the source video
        file, applying the callback function (with possible arguments) to each
        frame, and then "Writing" the resulting frame out as requested.

        WindowName:
            If the video frame is to be displayed, this will be the name of the
            window to create for displaying the frame data.
        QuitKeys:
            The set of keys to watch for while performing the video playback to
            allow quitting of the process partway through in a safe manner.
        PlaybackMode:
            The playback mode to use for the video. One of:
                PlaybackMode_SingleStep: Single-Step through the video,
                requiring a keypress to advance one frame.
                PlaybackMode_NormalSpeed: Play back the video at the frame-rate
                defined in the video codec. No interaction is necessary.
                PlaybackMode_NoDisplay: Iterate over the video frames without
                displaying them, as fast as possible.
                PlaybackMode_NoDelay: Similar to NoDisplay; iterate over the
                frames as fast as possible, while still displaying them. Similar
                to NormalSpeed, but ignores the requirement of matching a
                desired nominal framerate.
        Callback:
            The callback function to apply to each video frame. This function
            takes in as the first argument the video frame, and as the second
            argument a List containing all of the remainder of the required
            arguments. This function returns the resulting frame, as well as a
            bool to determine whether the callback was successful.
        CallbackArgs:
            A List of the additional arguments to pass to the Callback function
            on each frame. This list is not expanded until it enters the
            callback function, where it is the responsibility of the Callback
            function itself to handle these.
        StartFrame:
            The (0-indexed) frame index to start processing the video at. If not given,
            this defaults to frame index 0, or the beginning of the video.
        EndFrame:
            The (0-indexed) frame index to stop processing the video at. If not given,
            this defaults such that the full length of the video will be processed.

        Return (bool):
            True is the video was fully processed successfully, False otherwise.
        """

        self.PrepareWriter(self._OutputFilename, self.OutputFrameRate, self.OutputFourCC, (self.OutputWidth, self.OutputHeight), self.OutputCropTopLeft)

        WindowName, QuitKeys = self._prepareVideoPlayback(WindowName, QuitKeys, PlaybackMode)
        retVal = True

        if ( QuitKeys is None ):
            self._LogWriter.Errorln("Failed to prepare video playback, no QuitKeys defined. Stopping now.")
            return False

        for Frame in self.SetFrameRange(StartFrame, EndFrame):
            #   If a callback was provided, call it with the current frame to
            #   transform it.
            if ( Callback is not None ):
                try:
                    Frame, Ok = Callback(Frame, CallbackArgs)
                    if ( not Ok ):
                        self._LogWriter.Errorln("Callback Failure while processing video. Stopping now.")
                        retVal = False
                        break
                except Exception as e:
                    self._LogWriter.Errorln(f"Exception raised during Callback on source frame [ {self.SourceFrameIndex} ] - [ {e} ]\n\n{''.join(traceback.format_exception(e, value=e, tb=e.__traceback__))}\n")
                    raise Exception(f"Callback exception on source video frame [ {self.SourceFrameIndex} ].") from e

            WriteStatus: int = self.WriteFrame(Frame, PlaybackMode, WindowName, QuitKeys)
            if ( WriteStatus == ErrWriteFrame_NoOutputVideo ):
                raise AttributeError(f"Failed to write frame to requested output file, no video was initialized.")
            elif ( WriteStatus == ErrWriteFrame_Quit ):
                self._LogWriter.Println(f"Abort signalled during writing of frame [ {self.SourceFrameIndex + 1}/{self.EndFrameIndex} ]. Stopping processing now.")
                retVal = False
                break
            elif ( WriteStatus == ErrWriteFrame_SaveFailed ):
                self._LogWriter.Errorln(f"Failed to save video frame during playback. Continuing on...")
            elif ( WriteStatus == ErrWriteFrame_SizeMismatch ):
                self._LogWriter.Warnln(f"Failed to write frame to output video due to size mismatch. Expected ({self.OutputHeight}, {self.OutputWidth}), Got {Frame.shape[0:2]}.")
                retVal = False
                break

        self._resetProcessingRate()
        #   If we opened up a window, close it here.
        if ( WindowName is not None ):
            cv2.destroyWindow(WindowName)

        if ( retVal ):
            self._LogWriter.Println(f"Successfully processed video file [ {self._SourceFilename} ].")

        return retVal

    def SetFrameRange(self, StartFrameIndex: int = None, EndFrameIndex: int = None) -> VideoReadWriter:
        """
        SetFrameRange

        This function specifies the 0-indexed starting and ending frame indices
        to actually process from the source video file.

        StartFrameIndex:
            The 0-indexed frame index to start reading the video file from.
        EndFrameIndex:
            The 0-indexed frame index to end reading the video file before.
            This is one larger than the final frame index which will be read
            and processed.

        Return (VideoReadWriter)
            Self, to allow chaining methods. This VideoReadWriter will be configured
            to retain this frame range.
        """

        if ( self._SourceVideo is None ):
            if ( not self._openSourceVideo() ):
                self._LogWriter.Errorln("Failed to open source video for reading during ReadFrame().")
                self._closeSourceVideo()
                raise RuntimeError(f"Error: Cannot set video iteration range with no SourceVideo specified!")

        if ( StartFrameIndex is not None ):
            self.StartFrameIndex = StartFrameIndex
        elif (self.StartFrameIndex is None ):
            self.StartFrameIndex = 0

        if ( EndFrameIndex is not None ):
            if ( EndFrameIndex >= self.SourceNumFrames ):
                EndFrameIndex = self.SourceNumFrames
            self.EndFrameIndex = EndFrameIndex
        elif ( self.EndFrameIndex is None ):
            self.EndFrameIndex = self.SourceNumFrames

        if ( self.EndFrameIndex < 0 ):
            self.EndFrameIndex = self.SourceNumFrames

        if ( self.StartFrameIndex > self.EndFrameIndex ):
            raise RuntimeError(f"Error: StartFrameIndex must be less than or equal to EndFrameIndex!")

        self.Seek(self.StartFrameIndex)
        return self

    def Reduce(self, ReduceFunc: Callable[[np.ndarray, np.ndarray, List], Tuple[np.ndarray, bool]] = None, ReduceFuncArgs: List[Any] = [], StartFrame: int = None, EndFrame: int = None) -> Tuple[np.ndarray, bool]:
        """
        Reduce

        Apply some function over all of the frames of a video, condensing all of
        the frames into a single output frame. This is a simple generic
        implementation of the Reduce operation from the common map-filter-reduce
        functional programming workflow.

        ReduceFunc:
            The function to call on each frame. This returns some processed frame, and accepts
            the previous processed frame (None for the first), the current frame, and the "ReduceFuncArgs"
            list of additional parameters. This will be called on each and every frame of
            the video, accounting for the potentially configured start and end frame indices.
        ReduceFuncArgs:
            Additional user-defined parameters to pass to the ReduceFunc
        StartFrame:
            The 0-indexed frame to start processing from.
        EndFrame:
            The 0-indexed frame to finish processing before.

        Returns:
            np.ndarray:
                The reduced resultant video frame.
            bool:
                Boolean indicating if processing was successful or not.
        """

        if ( ReduceFunc is None ):
            self._LogWriter.Errorln(f"ReduceFunc must be provided!")
            raise ValueError(f"ReduceFunc must be provided!")

        ReducedFrame = None
        for Frame in self.SetFrameRange(StartFrame, EndFrame):
            try:
                ReducedFrame, Ok = ReduceFunc(ReducedFrame, Frame, ReduceFuncArgs)
                if ( not Ok ):
                    self._LogWriter.Errorln("ReduceFunc Failure while processing video. Stopping now.")
                    return ReducedFrame, False
            except Exception as e:
                self._LogWriter.Errorln(f"Exception raised during ReduceFunc on source frame [ {self.SourceFrameIndex} ] - {e}.")
                raise Exception(f"ReduceFunc exception on source video frame [ {self.SourceFrameIndex} ].") from e

        self._resetProcessingRate()

        self._LogWriter.Println(f"Successfully reached the end of the video.")
        return ReducedFrame, True

    def RemoveBackground(self, WindowName: str = None, QuitKeys: str = "qQ", PlaybackMode: int=PlaybackMode_NoDisplay, MedianFilterWidth: int = 100, BackgroundThreshold: int = None, ThresholdStyle: int = cv2.THRESH_TOZERO, BackgroundFrame: np.ndarray = None, StartFrame: int = None, EndFrame: int = None) -> bool:
        """
        RemoveBackground

        This function will compute the 'background' of the video file the
        VideoReadWriter was initialized with, using a median filter of a
        selectable width. This will then process the video file, removing the

        WindowName:
            If the video frame is to be displayed, this will be the name of the
            window to create for displaying the frame data.
        QuitKeys:
            The set of keys to watch for while performing the video playback to
            allow quitting of the process partway through in a safe manner.
        PlaybackMode:
            The playback mode to use for the video. One of:
                PlaybackMode_SingleStep: Single-Step through the video,
                requiring a keypress to advance one frame.
                PlaybackMode_NormalSpeed: Play back the video at the frame-rate
                defined in the video codec. No interaction is necessary.
                PlaybackMode_NoDisplay: Iterate over the video frames without
                displaying them, as fast as possible.
                PlaybackMode_NoDelay: Similar to NoDisplay; iterate over the
                frames as fast as possible, while still displaying them. Similar
                to NormalSpeed, but ignores the requirement of matching a
                desired nominal framerate.
        MedianFilterWidth:
            Integer containing the number of frames to use in computing the
            median frame of the resulting video.
        BackgroundThreshold:
            The threshold pixel value to use when differentiating between
            foreground and background.
        ThresholdStyle:
            The style of threshold to apply. See cv2.threshold() for the
            available options.
        (optional) BackgroundFrame:
            A pre-computed background frame to use, so that this doesn't
            explicitly recompute a background frame as a median-filtered frame.
            If not None, this will be used.
        StartFrame:
            The 0-indexed frame to start processing from.
        EndFrame:
            The 0-indexed frame to finish processing before.

        Returns:
            bool:
                Boolean value indicating the overall video processing was successful.
        """

        self.SetFrameRange(StartFrame, EndFrame)

        if ( BackgroundFrame is None ):
            self._LogWriter.Println(f"No frame provided as background to remove, computing and using Median-filtered frame as background.")
            BackgroundFrame = self.GetMedianFrame(MedianFilterWidth, True)

        if ( BackgroundThreshold is None ) or ( BackgroundThreshold <= 0 ):
            self._LogWriter.Println(f"No background threshold provided, determining optimal threshold using Otsu's algorithm.")
            ThresholdStyle |= cv2.THRESH_OTSU

        CallbackParams = [BackgroundFrame, BackgroundThreshold, ThresholdStyle]

        return self.ProcessVideo(WindowName, QuitKeys, PlaybackMode, _removeBackgroundCallback, CallbackParams)

    def ComputeMeanFrame(self, WindowName: str = None, QuitKeys: str = "qQ", PlaybackMode: int=PlaybackMode_NoDisplay, StartFrame: int = None, EndFrame: int = None) -> np.ndarray:
        """
        ComputeMeanFrame

        This function will compute the Mean frame of the given video, such
        performing a pixel-wise averaging of each and every frame of the video.
        If PlaybackMode is set to NoDisplay, a corresponding video will be
        generated in addition to the resulting mean frame showing the
        progression to this final Mean frame.

        WindowName:
            The name of the cv2.namedWindow() to create if the PlaybackMode
            implies displaying to the screen.
        QuitKeys:
            The set of keys to signal to interrupt and abort processing of the
            video.
        PlaybackMode:
            The playback mode to use for the video. One of:
                PlaybackMode_SingleStep: Single-Step through the video,
                requiring a keypress to advance one frame.
                PlaybackMode_NormalSpeed: Play back the video at the frame-rate
                defined in the video codec. No interaction is necessary.
                PlaybackMode_NoDisplay: Iterate over the video frames without
                displaying them, as fast as possible.
                PlaybackMode_NoDelay: Similar to NoDisplay; iterate over the
                frames as fast as possible, while still displaying them. Similar
                to NormalSpeed, but ignores the requirement of matching a
                desired nominal framerate.

                Note: If PlaybackMode is set to NoDisplay, this will also
                generate a video file showing the progression of the Mean frame
                as it is built up over the course of the video.
        StartFrame:
            The 0-indexed frame to start processing from.
        EndFrame:
            The 0-indexed frame to finish processing before.

        Return (np.ndarray):
            The resulting pixel-wise Mean frame of the source video file.
        """

        RawWindowName = None
        MeanWindowName = None

        if ( WindowName is None ) or ( WindowName == "" ):
            WindowName = os.path.basename(self._SourceFilename)

        if ( PlaybackMode != PlaybackMode_NoDisplay ):
            RawWindowName, RawQuitKeys = self._prepareVideoPlayback(f"{WindowName} - Raw", QuitKeys, PlaybackMode)
            if ( RawQuitKeys is None ):
                self._LogWriter.Errorln("Failed to prepare video playback of raw video frames, no QuitKeys defined. Stopping now.")
                return None

        MeanWindowName, MeanQuitKeys = self._prepareVideoPlayback(f"{WindowName} - Mean", QuitKeys, PlaybackMode)
        if ( MeanQuitKeys is None ):
            self._LogWriter.Errorln("Failed to prepare video playback of mean video frames, no QuitKeys defined. Stopping now.")
            return None

        MeanFrame = None

        self.SetFrameRange(StartFrameIndex=StartFrame, EndFrameIndex=EndFrame)
        WeightingFactor = 1.0 / (self.EndFrameIndex - self.StartFrameIndex)

        for Frame in self:

            #   Compute the initial "mean" frame so far.
            if ( MeanFrame is None ):
                cv2.moveWindow(MeanWindowName, 0, self.OutputHeight)
                MeanFrame = Frame * WeightingFactor
            else:
                MeanFrame += Frame * WeightingFactor

            if ( PlaybackMode != PlaybackMode_NoDisplay ):

                WriteStatus: int = self.WriteFrame(Frame, PlaybackMode, WindowName, QuitKeys)
                if ( WriteStatus == ErrWriteFrame_NoOutputVideo ):
                    raise AttributeError(f"Failed to write frame to requested output file, no video was initialized.")
                elif ( WriteStatus == ErrWriteFrame_Quit ):
                    self._LogWriter.Println(f"Abort signalled during writing of frame [ {self.SourceFrameIndex + 1}/{self.EndFrameIndex} ]. Stopping processing now.")
                    return None
                elif ( WriteStatus == ErrWriteFrame_SaveFailed ):
                    self._LogWriter.Errorln(f"Failed to save video frame during playback. Continuing on...")
                elif ( WriteStatus == ErrWriteFrame_SizeMismatch ):
                    self._LogWriter.Warnln(f"Failed to write frame to output video due to size mismatch. Expected ({self.OutputHeight}, {self.OutputWidth}), Got {Frame.shape[0:2]}.")
                    return None

            DisplayFrame = GammaCorrection(MeanFrame, Minimum=0, Maximum=255).astype(np.uint8)
            WriteStatus: int = self.WriteFrame(DisplayFrame, PlaybackMode, WindowName, QuitKeys)
            if ( WriteStatus == ErrWriteFrame_NoOutputVideo ):
                raise AttributeError(f"Failed to write frame to requested output file, no video was initialized.")
            elif ( WriteStatus == ErrWriteFrame_Quit ):
                self._LogWriter.Println(f"Abort signalled during writing of frame [ {self.SourceFrameIndex + 1}/{self.EndFrameIndex} ]. Stopping processing now.")
                return None
            elif ( WriteStatus == ErrWriteFrame_SaveFailed ):
                self._LogWriter.Errorln(f"Failed to save video frame during playback. Continuing on...")
            elif ( WriteStatus == ErrWriteFrame_SizeMismatch ):
                self._LogWriter.Warnln(f"Failed to write frame to output video due to size mismatch. Expected ({self.OutputHeight}, {self.OutputWidth}), Got {Frame.shape[0:2]}.")
                return None

        self._resetProcessingRate()

        #   If we opened up a window, close it here.
        if ( RawWindowName is not None ):
            cv2.destroyWindow(RawWindowName)

        if (MeanWindowName is not None ):
            cv2.destroyWindow(MeanWindowName)

        self._LogWriter.Println(f"Successfully computed mean frame from source video file [ {self._SourceFilename} ].")
        return MeanFrame.astype(np.uint8)

    def ReadFrame(self) -> np.ndarray:
        """
        ReadFrame:

        This function wraps the __next__() method, returning the next frame from the video.

        Returns (np.ndarray):
            This value is the returned frame. This consists of a 3D List of
            pixel data for the frame. This pixel data is organized as
            [Height, Width, Channel].
        """
        return self.__next__()

    def ReadFramePair(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        ReadFramePair:

        This function is a further wrapper of the self.__next__() method,
        extended to read two consecutive frames in a single operation. Operates
        in the same manner, as ReadFrame(), only twice.

        Returns (Tuple[np.ndarray, np.ndarray]):
            [0] np.ndarray:
                This value is the returned first frame. This consists of a 3D
                List of pixel data for the frame. This pixel data is organized
                as [Height, Width, Channel].
            [1] np.ndarray:
                This value is the returned second frame. This consists of a 3D
                List of pixel data for the frame. This pixel data is organized
                as [Height, Width, Channel].
        """

        return (self.__next__(), self.__next__())

    def GetFramesByNumber(self, FrameNumbers: List[int]) -> List[np.ndarray]:
        """
        GetFramesByNumber:

        This function will read a sequence of frames, based on the provided list
        of numbers.  These numbers correspond to the frame-index, or the
        sequence number of the frames to extract from the source video file.
        This is equivalent to a Seek() followed by a ReadFrame() for the
        requested frames.

        FrameNumbers:
            A list of integers, corresponding to the sequence number of each of
            the frames requested to be read from the video file. Any values
            outside the range of the video will simply be skipped.

        Returns (List[np.ndarray]):
            List of Frames, which are themselves 3D lists of pixels.
        """
        return [self[round(i)] for i in FrameNumbers]

    def SaveFrames(self, FrameIndices: List[int]) -> bool:
        """
        SaveFrames:

        This function...

        FrameIndices:
            ...

        Return (bool):
            ...
        """

        Success: bool = True
        SaveFolder: str = "./" if self._OutputFilename is None else os.path.dirname(self._OutputFilename)
        for FrameIndex in sorted(FrameIndices):
            Filename: str = f"{os.path.basename(self._SourceFilename)} - Frame {FrameIndex}.png"
            if cv2.imwrite(os.path.join(SaveFolder, Filename), self[FrameIndex]):
                self._LogWriter.Println(f"Successfully saved frame [ {FrameIndex} ] to file [ {Filename } ]!")
            else:
                self._LogWriter.Errorln(f"Failed to save frame [ {FrameIndex} ] to file [ {Filename } ]!")
                Success = False

        return Success


    def Seek(self, FrameNumber: int) -> None:
        """
        Seek:

        This function will adjust the current "active" frame of the video file
        to the requested frame number. This will move the video file in either
        direction to ensure the requested frame will be the next to be read.

        FrameNumber:
            Integer consisting of the sequence number of the frame to seek to.

        Return (None):
            None, the VideoReadWriter internals are set to ensure the next
            ReadFrame() or __next__() call return the Seek()'d to index.
        """

        #   Make sure the video is open
        if ( not self._openSourceVideo() ):
            self._LogWriter.Errorln(f"Failed to open source video file [ {self._SourceFilename} ] during Seek().")
            raise RuntimeError(f"Cannot retrieve frame, source video is not opened or accessible!")

        FrameNumber = int(FrameNumber)

        #   Make sure the requested frame number is in range.
        if ( 0 > FrameNumber ) or ( FrameNumber > self.SourceNumFrames ):
            self._LogWriter.Errorln(f"Requested frame index [ {FrameNumber} ] is out of range of source video file: [ 0, {self.SourceNumFrames - 1} ].")
            raise RuntimeError(f"Requested index [ {FrameNumber} ] is out of range!")

        #   Seek to the requested frame number, and update the public class variable.
        self._SourceVideo.set(cv2.CAP_PROP_POS_FRAMES, FrameNumber)
        self.SourceFrameIndex = FrameNumber

        return True

    def DisplayFrame(self, FrameIndex: int = -1) -> bool:
        """
        DisplayFrame

        This function is a simple helper function for Seek()-ing to and displaying
        a desired specific frame from a given video file.

        FrameIndex:
            The 0-indexed integer index number for the frame to read and display.

        Return (bool):
            Boolean indicating whether or not the operation succeeded.
        """


        #   Make sure this VideoReadWriter is initialized with a source video file to read from.
        if ( self._SourceVideo is None ):
            self._LogWriter.Errorln(f"Cannot DisplayFrame() without an initialized SourceFile.")
            return False

        OriginalIndex: int = self.SourceFrameIndex
        FrameToDisplay: np.ndarray = self[FrameIndex]
        self.Seek(OriginalIndex)

        DisplayFrameWindowName = f"Frame {FrameIndex} - {os.path.basename(self._SourceFilename)}"
        DisplayFrameWindowName, _ = self._prepareVideoPlayback(WindowName=DisplayFrameWindowName, QuitKeys='qQ', PlaybackMode=PlaybackMode_SingleStep)

        WriteStatus: int = self.WriteFrame(FrameToDisplay, PlaybackMode_SingleStep, DisplayFrameWindowName)
        if ( WriteStatus == ErrWriteFrame_NoOutputVideo ):
            raise AttributeError(f"Failed to write frame to requested output file, no video was initialized.")
        elif ( WriteStatus == ErrWriteFrame_Quit ):
            self._LogWriter.Println(f"Abort signalled during writing of frame [ {self.SourceFrameIndex + 1}/{self.EndFrameIndex} ]. Stopping processing now.")
            return False
        elif ( WriteStatus == ErrWriteFrame_SaveFailed ):
            self._LogWriter.Errorln(f"Failed to save video frame during playback. Continuing on...")
        elif ( WriteStatus == ErrWriteFrame_SizeMismatch ):
            self._LogWriter.Warnln(f"Failed to write frame to output video due to size mismatch. Expected ({self.OutputHeight}, {self.OutputWidth}), Got {FrameToDisplay.shape[0:2]}.")
            return False

        return True

    def GetMedianFrame(self, MedianFilterWidth: int = None, Greyscale: bool = True, StartFrame: int = None, EndFrame: int = None) -> np.ndarray:
        """
        GetMedianFrame:

        This function will read a video file, and compute the median frame of
        the video. The number of frames used in computing the median frame is
        provided, and this allows for some tasks like background extraction.

        MedianFilterWidth:
            Integer containing the number of frames to use in computing the
            median frame of the resulting video.
        Greyscale:
            Boolean value for whether or not the returned median frame should be
            converted to greyscale or left in the original colour palette.
        StartFrame:
            The 0-indexed frame to start processing from.
        EndFrame:
            The 0-indexed frame to finish processing before.

        Return (np.ndarray):
            The resulting median frame, consisting of a 3D list of pixels
            (Height, Width, Channel). None if an error occurred during median
            filtering.
        """

        if ( not self._openSourceVideo() ):
            self._LogWriter.Errorln(f"Failed to open source video file [ {self._SourceFilename} ] during GetMedianFrame().")
            return None

        self.SetFrameRange(StartFrame, EndFrame)
        NumFrames: int = (self.EndFrameIndex - self.StartFrameIndex)

        if ( MedianFilterWidth is None ):
            DefaultMedianFilterWidth = 0.025
            MedianFilterWidth = max(round(NumFrames * DefaultMedianFilterWidth), 3)
            self._LogWriter.Warnln(f"Argument [ MedianFilterWidth ] not given, defaulting to use {DefaultMedianFilterWidth * 100:.1f}% of the total number of frames: [ {MedianFilterWidth} ].")


        if not ( 0 < MedianFilterWidth < NumFrames ):
            self._LogWriter.Errorln(f"Median filter width [ {MedianFilterWidth} ] larger than the number of usable frames in the video: [ {NumFrames} ].")
            MedianFilterWidth = NumFrames

        #   Generate a random set of frame sequence numbers, equal in size to
        #   the desired number for the median filter.
        FrameIDs = list(sorted(set(random.sample(range(self.StartFrameIndex, self.EndFrameIndex), k=MedianFilterWidth))))
        Frames = [self[x] for x in FrameIDs]

        if ( len(Frames) != MedianFilterWidth ):
            self._LogWriter.Warnln(f"Requested [ {MedianFilterWidth} ] frames for computing median frame, only read [ {len(Frames)} ] frames.")

        MedianFrame = np.median(Frames, axis=0).astype(dtype=np.uint8)
        if ( Greyscale ):
            MedianFrame = cv2.cvtColor(MedianFrame, cv2.COLOR_BGR2GRAY)

        self._LogWriter.Println(f"Successfully computed median frame for video file [ {self._SourceFilename} ].")
        return MedianFrame

    ## VideoWriter Methods
    def EnableTopmost(self) -> VideoReadWriter:
        """
        EnableTopmost

        This function sets the internal flag to ensure that windows opened by this
        VideoReadWriter will be opened with the ENABLE_TOPMOST flag set, so they
        appear above all other windows opened.

        Return (VideoReadWriter):
            The same VideoReadWriter this function is called on, to allow function
            chaining.
        """
        self._OutputWindowTopmost = True
        return self

    def DisableTopmost(self) -> VideoReadWriter:
        """
        DisableTopmost

        This function sets to false the internal flag to ensure that windows opened by this
        VideoReadWriter will be opened with the ENABLE_TOPMOST flag set, so they
        appear above all other windows opened.

        Return (VideoReadWriter):
            The same VideoReadWriter this function is called on, to allow function
            chaining.
        """
        self._OutputWindowTopmost = False
        return self

    def SetVideoRotationAngle(self, RotationAngle: float) -> VideoReadWriter:
        """
        SetVideoRotationAngle

        This function sets the internal frame-rotation correction angle. This will
        allow this VideoReadWriter to transparently rotate frames prior to returning
        them to the caller for any __next__(), ReadFrame(), or related calls. This allows
        for processing a video file which must be rotated by simply defining this angle
        once prior to a ProcessRead(), and then proceeding as normal.

        RotationAngle:
            The angle, in degrees, to rotate each video frame counter-clockwise.

        Return (VideoReadWriter):
            The same VideoReadWriter this function is called on, to allow for
            function chaining.
        """

        if ( RotationAngle is None ):
            self.OutputRotationAngle = None
        else:
            self.OutputRotationAngle = (RotationAngle % 360)

        return self

    def PrepareWriter(self, OutputFilename: str = None, FrameRate: float = 30, FourCC: str = "mp4v", Resolution: Tuple[int, int] = (-1, -1), TopLeft: Tuple[int, int] = (0,0)) -> VideoReadWriter:
        """
        PrepareWriter:

        This function will prepare the output video writer within the
        VideoReadWriter to write out to a specified file, with the given
        frame-rate, dimensions, and video codec. This returns the resulting
        modified VideoReadWriter to allow chaining operations on itself.

        OutputFilename:
            The full filepath to the output video to write frame data out to.
        FrameRate:
            The desired frame-rate the output video file should play at.
        FourCC:
            The FourCC (four character code) indicating the video codec to use
            when writing the output video file. Defaults to mp4v, or Motion JPEG
            encoding.
        Resolution:
            The dimensions (in pixels) of the output video file. A tuple of
            integers such that it represents (width, height).
        TopLeft:
            If the output video should be a cropped version of the input video,
            this defines the pixel location in the source video to set as the
            top-left pixel (0, 0) in the output video.

        Return (VideoReadWriter):
            The modified VideoReadWriter class this method operated on, modified
            such that the Writer component is prepared to write to an output
            video file or display window as required.

        Note:
            If any of FrameRate, Resolution, or TopLeft are to be inherited from
            the Source video file, pass the values as -1.
        """

        self._closeOutputVideo()

        if ( OutputFilename is not None ):
            self._OutputFilename = OutputFilename

        if ( FrameRate is None ):
            FrameRate = self.SourceFrameRate

        if ( FourCC is None ):
            FourCC = "mp4v"

        if ( Resolution[0] is None ):
            Resolution = (self.SourceWidth, Resolution[1])

        if ( Resolution[1] is None ):
            Resolution = (Resolution[0], self.SourceHeight)

        if ( TopLeft is None ):
            TopLeft = (0, 0)

        self.OutputFrameRate = FrameRate
        self.OutputFourCC = FourCC
        self.OutputWidth = int(Resolution[0])
        self.OutputHeight = int(Resolution[1])
        self.OutputCropTopLeft = TopLeft

        #   Check if there's an open source video file that we can inherit video
        #   parameters from. A value of -1 indicates to inherit from the source
        #   video.
        if ( self._SourceVideo is not None ):

            if (( self.OutputFrameRate == -1 ) or ( self.OutputFrameRate is None )):
                self.OutputFrameRate = self.SourceFrameRate

            if (( self.OutputWidth == -1 ) or ( self.OutputWidth is None )):
                self.OutputWidth = self.SourceWidth

            if (( self.OutputHeight == -1 ) or ( self.OutputHeight is None )):
                self.OutputHeight = self.SourceHeight

        return self

    def WriteFrame(self, Frame: np.ndarray, PlaybackMode: int = PlaybackMode_NoDisplay, WindowName: str = "", QuitKeys: List[int] = [ord('q'), ord('Q')], SaveKeys: List[int] = [ord('s'), ord('S')]) -> int:
        """
        WriteFrame:

        This function will write the given frame "out", either to the configured
        output file, or to the output playback window.

        Frame:
            The raw frame-data containing the pixel values of the frame to be
            written out.
        PlaybackMode:
            The playback mode for writing, determining whether it's to a file or
            a display window.
        WindowName:
            If writing the frame to a display window, this is the name of the
            display window to write it to.
        QuitKeys:
            A list of the raw key codes to signal to quit processing after the frame.
        SaveKeys:
            A list of the raw key codes to signal to save the given frame as a
            still image, in addition to whatever the PlaybackMode indicates.

        Return (int):
            An integer indicating whether or not the frame was written successfully.
            0 indicates no errors, while a non-zero code may or may not be an error.
            See the ErrWriteFrame_* constants for definitions and explanations.
        """

        retVal: int = ErrWriteFrame_None

        if ( Frame is None ):
            self._LogWriter.Warnln(f"[ None ] Frame provided to WriteFrame(), skipping frame and moving on.")
            return ErrWriteFrame_None

        #   Make sure the VideoReadWriter is set up to properly handle writing
        #    (or not) to an output video file
        if ( self._openOutputVideo() == False ):
            self._LogWriter.Errorln("Failed to write frame, could not open output video file for writing.")
            return ErrWriteFrame_NoOutputVideo

        #   If the requested resolution of the output video file is different
        #   than the source file, automatically crop the frame here before
        #   writing it out.
        Frame = self._crop(Frame)

        #   Define the amount of time to wait between frames during playback.
        #   Default to waiting 1ms, to allow for key-presses to interrupt and
        #   safely shut down the processing.
        WaitTime = 1
        if ( PlaybackMode == PlaybackMode_NormalSpeed ):
            #   When playing back at normal speed, set the wait time to be equal
            #   to the required inter-frame timing to achieve the requested
            #   frame-rate.
            Now = time.time()
            WaitTime = round(1000 * ((1.0 / self.OutputFrameRate) - (Now - self._LastFrameWrittenTimestamp)))
            if ( WaitTime <= 0 ):
                WaitTime = 1
            elif ( WaitTime >= round(1000.0 / self.OutputFrameRate) ):
                WaitTime = round(1000.0 / self.OutputFrameRate)
        elif ( PlaybackMode == PlaybackMode_SingleStep ):
            #   When single-stepping through a video, wait until a key-press
            #   without allowing for a timeout to automatically move to the next
            #   frame.
            WaitTime = 0

        #   Write to the output video file, if it exists
        if ( self._OutputVideo is not None ):
            #   If the size of the frame to be written does not match the size of the video file
            #   this writer is configured to write to, return an error indicating this.
            if (self.OutputWidth != Frame.shape[1] or self.OutputHeight != Frame.shape[0]):
                return ErrWriteFrame_SizeMismatch

            self._OutputVideo.write(Frame)

        #   If the playback mode also indicates to display the frame, do so here
        if ( PlaybackMode != PlaybackMode_NoDisplay ):

            #   Show the image on screen in the defined window.
            cv2.imshow(WindowName, Frame)

            #   Wait until timeout for a key-press.
            key = cv2.waitKey(WaitTime)

            #   If the key-press indicates to save the current frame as a still
            #   image, do so.
            if ( key in SaveKeys ):
                self.Seek(self.SourceFrameIndex - 1)
                StillImageFilename: str = ""
                StillImageDirectory: str = ""
                if ( self._OutputFilename is None ) or ( self._OutputFilename == "" ):
                    StillImageDirectory = os.path.dirname(self._SourceFilename)
                    StillImageFilename = os.path.splitext(os.path.basename(self._SourceFilename))[0]
                else:
                    StillImageDirectory = os.path.dirname(self._OutputFilename)
                    if ( not os.path.exists(StillImageDirectory) ):
                        StillImageDirectory = os.path.dirname(self._SourceFilename)
                    StillImageFilename = os.path.splitext(os.path.basename(self._OutputFilename))[0]

                StillImageFilename = os.path.join(StillImageDirectory, f"{StillImageFilename} - Frame {self.SourceFrameIndex}.png")
                self._LogWriter.Println(f"Save key [ {chr(key)} ] pressed, attempting to save frame [ {self.SourceFrameIndex} ] to still image file [ {StillImageFilename} ].")
                if ( cv2.imwrite(StillImageFilename, Frame) ):
                    self._LogWriter.Println(f"Successfully saved frame [ {self.SourceFrameIndex} ] to still image file [ {StillImageFilename} ].")
                    retVal = ErrWriteFrame_None
                else:
                    self._LogWriter.Errorln(f"Failed to save frame [ {self.SourceFrameIndex} ] to still image file [ {StillImageFilename} ].")
                    retVal = ErrWriteFrame_SaveFailed

            #   If the key-press indicates to quit processing, do so.
            if ( key in QuitKeys ):
                self._LogWriter.Println(f"Quit key [ {chr(key)} ] pressed, stopping playing video file [ {self._SourceFilename} ].")
                retVal = ErrWriteFrame_Quit

        #   Otherwise, if the key-press is something else, or it timed out, move to the next frame.
        self._LastFrameWrittenTimestamp = time.time()
        return retVal

def ParsePlaybackMode(PlaybackMode: str = None) -> int:

    if ( PlaybackMode is None ) or ( PlaybackMode == "" ):
        return PlaybackMode_NoDisplay

    PlaybackMode = PlaybackMode.lower()

    if ( 'step' in PlaybackMode.lower() ):
        return PlaybackMode_SingleStep
    elif ( 'normal' in PlaybackMode.lower() ):
        return PlaybackMode_NormalSpeed
    elif ( 'fast' in PlaybackMode.lower() ):
        return PlaybackMode_NoDelay
    elif ( 'hidden' in PlaybackMode.lower() ):
        return PlaybackMode_NoDisplay

    return PlaybackMode_NoDisplay

def PlaybackModeToString(PlaybackMode: int) -> str:
    """
    PlaybackModeToString

    This function...

    PlaybackMode:
        ...

    Return (str):
        ...
    """

    if ( PlaybackMode == PlaybackMode_NoDelay ):
        return "NoDelay (fast)"
    elif ( PlaybackMode == PlaybackMode_NoDisplay ):
        return "NoDisplay (hidden)"
    elif ( PlaybackMode == PlaybackMode_NormalSpeed ):
        return "NormalSpeed"
    elif ( PlaybackMode == PlaybackMode_SingleStep ):
        return "Single-Step"
    else:
        return f"Unknown: {PlaybackMode}"

def _removeBackgroundCallback(CurrentFrame: np.ndarray, Parameters: List) -> Tuple[np.ndarray, bool]:
    """
    _removeBackgroundCallback

    This function is the callback function to pass during ProcessVideo() to remove the 'background'
    of the given frame.

    CurrentFrame:
        The current frame of the video being processed, as passed by ProcessVideo().
    Parameters:
        [0] (np.ndarray): The background frame, as a grayscale image to be subtracted from the current frame.
        [1] (int): The threshold between the background and foreground to use when performing the
        binary threshold.
        [2] (int): The threshold style to apply. See cv2.threshold() for options.

    Returns:
        np.ndarray:
            The resulting 'foreground-only' frame, resulting from (CurrentFrame - BackgroundFrame).
        bool:
            Boolean value indicating if the callback was successful.
    """

    BackgroundFrame = Parameters[0]
    Threshold = Parameters[1]
    ThresholdStyle = Parameters[2]

    CurrentFrame = cv2.cvtColor(CurrentFrame, cv2.COLOR_BGR2GRAY)
    CurrentFrame = cv2.absdiff(CurrentFrame, BackgroundFrame)

    if ( ThresholdStyle > 0 ):
        _, CurrentFrame = cv2.threshold(CurrentFrame, Threshold, 255, ThresholdStyle)

    CurrentFrame = cv2.cvtColor(CurrentFrame, cv2.COLOR_GRAY2BGR)

    return CurrentFrame, True

def GammaCorrection(Image: np.ndarray = None, Gamma: float = 1.0, Minimum: int = None, Maximum: int = None) -> np.ndarray:
    """
    GammaCorrection

    This function performs a basic Gamma correction. This applies an exponential scaling
    with the Gamma factor, followed by a linear rescaling to the given Minimum and Maximum.
    See (https://en.wikipedia.org/wiki/Gamma_correction) for details.

    Image:
        The raw image to process.
    Gamma:
        The exponential scaling factor to apply to the frame.
    Minimum:
        The final minimum pixel brightness value to rescale to.
    Maximum:
        The final maximum pixel brightness value to rescale to.

    Return (np.ndarray):
        The final, brightness scaled image.
    """

    if ( Image is None ):
        raise ValueError(f"Image must be provided.")

    if ( 0 >= Gamma ):
        raise ValueError(f"Gamma must be provided and be a positive real number.")

    OriginalType: np.dtype = Image.dtype
    Limits = None
    if ( np.issubdtype(OriginalType, np.integer)):
        Limits = np.iinfo(OriginalType)
    elif ( np.issubdtype(OriginalType, np.float)):
        Limits = np.finfo(OriginalType)
    else:
        raise TypeError(f"Numpy NDArray has non-integral and non-floating point dtype!")

    if ( Minimum is None ):
        Minimum = Limits.min

    if ( Maximum is None ):
        Maximum = Limits.max

    #   Perform the non-linear exponentiation operation
    #       allowing a fast-path for the no-op of exponentiation by 1.
    Scaled = Image.astype(np.float64)
    if ( Gamma != 1.0 ):
        Scaled = Scaled ** Gamma

    #   Linearly re-scale the resulting image to the desired min/max range provided.
    Offset = np.min(Scaled) - Minimum
    if ( Offset != 0.0 ):
        Scaled = Scaled - (np.min(Scaled) - Minimum)

    if ( np.max(Scaled) != 0 ):
        ScaleFactor = Maximum / np.max(Scaled)
        if ( ScaleFactor != 1.0 ):
            Scaled *= ScaleFactor

    return Scaled.astype(OriginalType)
