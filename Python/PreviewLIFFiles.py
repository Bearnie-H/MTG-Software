#!/usr/bin/env python3

#   Author: Joseph Sadden
#   Date:   31st July, 2025

#   Script Purpose: ...
#                       ...

#   Import the necessary standard library modules
from __future__ import annotations
import typing
from typing import TextIO

#   3rd Party Imports
from datetime import datetime
import glob
import math
import os
import sys
import traceback
#   ...

#   Import the necessary third-party modules
import cv2
import numpy as np
import readlif
from readlif.reader import LifFile, LifImage
import czifile
#   ...

#   Import the desired locally written modules
#   ...

#   Locally Required Class Definitions
class Logger():
    """
        Logger:

        This is an embeddable logging API, suitable for using as the generic logger
        for scripts and command-line tools. This allows for raw writing, formatted printing,
        flushing of the output stream, and toggle-able between interactive and machine-modes.
    """

    ##  Public Class Members
    #   ...

    ##  Private Class Members

    #   The TextIO stream to write the log messages to.
    _OutputStream: TextIO

    #   The filename, if any has been provided, to which the logger writes to.
    _Filename: str

    #   A possible prefix to add to the beginning of any message.
    _MessagePrefix: str

    #   Should the logger prefix messages with a time-stamp?
    _TimeStamp: bool

    #   The width of the columns to allow printing.
    _Columns: int

    #   Boolean indicating whether the last output operation was a raw Write or
    #   a formatted write operation. Used in ensuring messages are not
    #   overwritten when interleaving raw and formatted writes.
    _LastWriteRaw: bool

    ##  Magic Methods
    def __init__(self: Logger, OutputStream: TextIO = sys.stdout, Prefix: str = None, IncludeTime: bool = True, Columns: int = -1, AlwaysFlush: bool = False) -> None:
        """
        Constructor:

        This function prepares a Logger, ensuring it is ready to be used.

        OutputStream:
            The output TextIO stream to which the log messages will be written.
            Can be an opened file, network connection, or anything which can be
            operated on as a TextIO.
        Prefix:
            A prefix to add to the beginning of all log messages. Will be
            formatted as
                [ <Prefix> ] <Message>
        IncludeTime:
            Boolean to indicate whether a timestamp should be added to the log
            messages.  If so, this will add an HH:MM:SS time code to the
            beginning of all messages, before even the prefix (if given):
                [ <HH:MM:SS> ] <[ <Prefix> ]> <Message>
        Columns:
            The number of columns to wrap the log messages at. Set negative to
            not hard-wrap the messages. If wrapping is enabled, this will add
            whitespace under the wrapped lines to skip the prefix and time-code,
            if enabled.
        AlwaysFlush:
            Boolean flag to assert that the output stream is flushed after every
            single write operation. Useful for cases where you may have long
            periods of time between potential flushes.
        """

        if ( OutputStream is None ):
            OutputStream = sys.stdout

        self._OutputStream = OutputStream
        self._Filename = None
        self._MessagePrefix = Prefix
        self._TimeStamp = IncludeTime
        self._Columns = Columns
        self._LastWriteRaw = False
        self._AlwaysFlush = AlwaysFlush

        return

    def __del__(self: Logger) -> None:
        """
        Destructor

        This will safely close down a Logger, ensuring the output is flushed
        before closing the output stream.
        """
        if ( not self._OutputStream.closed ):
            if ( self._LastWriteRaw ):
                self._OutputStream.write('\n')
            self._OutputStream.flush()

            if ( self.WritesToFile() ):
                self._OutputStream.close()

        return

    ##  Private Class Methods
    def _write(self: Logger, Message: str, Flush: bool = False) -> int:
        """
        _write:

        This is the bottom-level function which actually pushes the log message
        out to the output stream. This will add the prefix (if set), and compute
        the timestamp (if requested) to the beginning of the messages, as well
        as chunking the message if hard-wrapping was requested.

        Message:
            The full, formatted string of the message to be written.
        Flush:
            Boolean indicating whether or not the output stream should be flushed
            forcefully after the write() call.

        Return (int):
            The total number of characters actually written to the output stream.
        """

        if ( self._OutputStream.closed ):
            return -1

        Prefix = ""
        Timestamp = ""
        LeadingWhitespace = ""
        nWritten = 0

        #   If the last write was a raw write, force going to a new line to ensure we don't overwrite
        #       whatever the last write may have been.
        if ( self._LastWriteRaw ):
            self._OutputStream.write('\n')
            self._OutputStream.flush()

        #   If there's nothing to write, return right away.
        if ( Message is None ) or ( Message == "" ):
            return nWritten

        #   If the prefix is requested, configure this string for inclusion.
        if ( self._MessagePrefix is not None ) and ( self._MessagePrefix != "" ):
           Prefix = f"[ {self._MessagePrefix} ] "

        #   If the timestamp is requested, prepare the timestamp for this message.
        if ( self._TimeStamp is True ):
            Timestamp = f"[ {datetime.now().strftime('%H:%M:%S')} ] "

        #   If hard-wrapping is enabled, perform some extra work to format the
        #   wrapped messages properly.
        if ( self._Columns > 0 ):
            #   Compute how much whitespace needs to be added at the beginning
            #   of each subsequent message.
            LeadingWhitespace = " " * (len(Prefix) + len(Timestamp))

            #   Compute how much of the message can be printed on each line,
            #   making sure that it's never less than 10 characters.
            ChunkLength = self._Columns - (len(Prefix) + len(Timestamp))
            if ( ChunkLength <= 0 ):
                ChunkLength = 10

            #   Split up the message into chunks of the required size, and
            #   iterate over the chunks.
            MessageChunks = [Message[i:i+ChunkLength] for i in range(0, len(Message), ChunkLength)]
            for Index, CurrentChunk in enumerate(MessageChunks):

                #   For the first chunk, print it with the prefix and timestamp
                #   (if required.)
                if ( Index == 0 ):

                    Chunk = f"{Timestamp}{Prefix}{CurrentChunk}\n"
                    self._OutputStream.write(Chunk)
                    nWritten += len(Chunk)
                else:
                    #   Otherwise, pad with leading whitespace and then the
                    #   message.
                    self._OutputStream.write(f"{LeadingWhitespace}{CurrentChunk}\n")
                    nWritten += len(f"{LeadingWhitespace}{CurrentChunk}")
        else:
            #   If no hard-wrapping is enabled, just print the message.
            self._OutputStream.write(f"{Timestamp}{Prefix}{Message}\n")
            nWritten = len(f"{Timestamp}{Prefix}{Message}\n")

        #   Flush the output stream if requested.
        if ( Flush or self._AlwaysFlush ):
            self._OutputStream.flush()

        #   Return the number of characters written to the stream.
        return nWritten

    ##  Public Class Methods
    def Copy(self: Logger) -> Logger:
        """
        Copy

        This function...

        Return (Logger):
            ...
        """

        New: Logger = Logger(self.RawStream(), self._MessagePrefix, self._TimeStamp, self._Columns, self._AlwaysFlush)
        New._Filename = self._Filename

        return New

    def RawStream(self: Logger) -> TextIO:
        return self._OutputStream

    def SetOutputStream(self: Logger, Stream: TextIO) -> Logger:
        """
        SetOutputStream

        This function...

        Stream:
            ...

        Return (Logger):
            ...
        """

        if ( Stream is None ):
            self.Warnln(f"No new \"Stream\" provided. Changing nothing...")
            return self

        if ( Stream.closed ):
            self.Errorln(f"Provided \"Stream\" is already closed. Changing nothing...")
            return self

        self._Filename = None
        self._OutputStream = Stream
        return Stream

    def SetOutputFilename(self: Logger, Filename: str = None) -> Logger:

        if ( Filename is None ) or ( Filename == "" ):
            self.Warnln("None or empty filename provided to set Logger output to, changing nothing...")
            return self

        if ( Filename == "-" ):
            self.Println("Setting Logger output to stdout.")
            self._Filename = None
            self._OutputStream = sys.stdout
            return

        self._Filename = Filename
        LogDirectory: str = os.path.dirname(Filename)
        if ( LogDirectory is None ) or ( LogDirectory == "" ):
            LogDirectory = "./"

        if ( not os.path.exists(LogDirectory) ):
            self.Println("Directory for provided new log filename does not exist, creating it now...")
            os.makedirs(LogDirectory, mode=0o755, exist_ok=True)

        self.Println(f"Opening and changing Logger output stream to [ {Filename} ].")
        self._OutputStream = open(Filename, mode="+wt")

        return self

    def SetPrefix(self: Logger, Prefix: str) -> Logger:
        """
        SetPrefix

        This function...

        Prefix:
            ...

        Return (self):
            ...
        """

        self._MessagePrefix = Prefix
        return self

    def AppendToPrefix(self: Logger, ToAppend: str) -> Logger:
        """
        AppendToPrefix

        This function...

        ToAppend:
            ...

        Return (self):
            ...
        """

        if ( self._MessagePrefix is None ):
            self._MessagePrefix = ToAppend
        else:
            self._MessagePrefix += f" {ToAppend}"

        return self

    def GetOutputFilename(self: Logger) -> str:
        """
        GetOutputFilename:

        This function...

        Return (str):
            ...
        """
        return self._Filename

    def WritesToFile(self: Logger) -> bool:
        """
        WritesToFile

        This function tells whether or not this logger is writing to stdout or stderr, or to a file.
        This can be used to skip user-interactive messages when writing to a log file.

        Return (bool):
            True if the underlying output stream is either stdout or stderr.
        """
        return self._Filename is not None

    def Println(self: Logger, Message: str = None) -> int:
        """
        Println:

        This function will format the message as a standard log message,
        ensuring a newline at the end of the message.

        Message:
            The message to print to the log.
        """

        nWritten = self._write(Message)
        self._LastWriteRaw = False
        return nWritten

    def Warnln(self: Logger, Message: str = None) -> int:
        """
        Warnln:

        This function will print a warning message to the log, prepending it
        with "Warning:" and ensuring a newline at the end of the message.

        Message:
            The message to print to the log.
        """

        nWritten = self._write(f"Warning: {Message}", True)
        self._LastWriteRaw = False
        return nWritten

    def Errorln(self: Logger, Message: str) -> int:
        """
        Errorln:

        This function will print an error message to the log, prepending it with
        "Error:" and ensuring a newline at the end of the message.

        Message:
            The message to print to the log.
        """

        nWritten = self._write(f"Error: {Message}", True)
        self._LastWriteRaw = False
        return nWritten

    def Fatalln(self: Logger, Message: str) -> None:
        """
        Fatalln:

        This function will print a fatal error message to the log, prepending it
        with "FATAL ERROR:" and ensuring a newline at the end of the message.
        This will also forcefully exit the program.

        Message:
            The message to print to the log.
        """

        self._write(f"FATAL ERROR: {Message}", True)
        self._OutputStream.flush()

        sys.stderr.write(f"FATAL ERROR: {Message}\n")
        sys.stderr.flush()

        self._LastWriteRaw = False
        sys.exit(1)

    def Write(self: Logger, Message: str = None, Flush: bool = True) -> int:
        """
        Write:

        This function will perform a raw write() call on the underlying output
        stream. Used for unformatted writing to the stream, where all logic of the Logger
        is skipped.

        Message:
            The message to write to the output stream.
        Flush:
            Boolean indicating whether the stream should be flushed after the write.
        """

        nWritten = self._OutputStream.write(Message)
        if ( Flush == True ):
            self._OutputStream.flush()

        self._LastWriteRaw = True
        return nWritten

Discarder: Logger = Logger(OutputStream=open(os.devnull, "w"))
Discarder._write = lambda *x, **y: None

class ZStack():
    """
    ZStack

    This class represents a single Z-Stack image as a rectangular grid of pixels
    in the X, Y, Z axes. This supports arbitrary bit-depth and colour channels
    per pixel, and provides a standard interface for operating over a Z-Stack
    similar to standard 2D images via either OpenCV or NumPy.
    """

    ##  Public Member Variables
    Name: str
    Pixels: np.ndarray
    #   ...

    ##  Private Member Variables
    _LogWriter: Logger
    #   ...

    ### Magic Methods
    def __init__(self: ZStack, LogWriter: Logger = Discarder, Name: str = None) -> None:
        """
        Constructor

        This function...

        Return (None):
            ...
        """

        self.Name = Name

        self.Pixels = None
        self._LogWriter = LogWriter

        return

    ### Static Class Methods
    @staticmethod
    def FromFile(Filename: str, *args) -> ZStack:
        """
        FromFile

        This function...

        Filename:
            ...
        args:
            ...

        Return (ZStack):
            ...
        """

        if ( not os.path.exists(Filename) ):
            raise ValueError(f"File [ {Filename} ] cannot be opened as the file does not exist.")

        match os.path.splitext(Filename)[1].lower():
            case ".lif":
                return ZStack.FromLIF(Filename, *args)
            case ".tif" | ".tiff":
                return ZStack.FromTIF(Filename)
            case ".czi":
                return ZStack.FromCZI(Filename)
            case _:
                raise NotImplementedError(f"Z-Stacks from [ {os.path.splitext(Filename)[1].lower()} ] files is not yet supported!")

    @staticmethod
    def FromLIF(Filename: str, *, SeriesName: str = "", SeriesIndex: int = -1, ChannelIndex: int = -1) -> ZStack:
        """
        FromLIF

        This function:
            ...

        Filename:
            ...
        SeriesName:
            ...
        SeriesIndex:
            ...
        ChannelIndex:
            ...

        Return (ZStack):
            ...
        """

        Stack: ZStack = ZStack()

        Success: bool = Stack.OpenLIFFile(Filename, SeriesName=SeriesName, SeriesIndex=SeriesIndex, ChannelIndex=ChannelIndex)
        if ( Success ):
            return Stack

        return None

    @staticmethod
    def FromTIF(Filename: str) -> ZStack:
        """
        FromTIF

        This function...

        Filename:
            ...

        Return (ZStack):
            ...
        """

        Stack: ZStack = ZStack()

        Success: bool = Stack.OpenTIFFile(Filename)
        if ( Success ):
            return Stack

        return None

    #   ...

    @staticmethod
    def FromCZI(Filename: str) -> ZStack:
        """
        FromCZI

        This function...

        Filename:
            ...

        Return (ZStack):
            ...
        """

        Stack: ZStack = ZStack()

        Success: bool = Stack.OpenCZIFile(Filename)
        if ( Success ):
            return Stack

        return None

    @staticmethod
    def FromImage(Image: np.ndarray) -> ZStack:
        """
        FromImage

        This function...

        Image:
            ...

        Return (ZStack):
            ...
        """

        Stack: ZStack = ZStack()
        Stack = Stack.InitializePixels((1,) + Image.shape)
        Stack.Pixels[0,:,:] = Image

        return Stack

    ### Public Methods
    def Copy(self: ZStack) -> ZStack:
        """
        Copy

        This function...

        Return (ZStack):
            ...
        """

        New: ZStack = ZStack(self._LogWriter, self.Name)
        New.Pixels = self.Pixels.copy()

        return New

    def InitializePixels(self: ZStack, Shape: typing.Tuple[int, int, int]) -> ZStack:
        """
        InitializePixels

        This function

        Shape:
            ...

        Return (ZStack):
            ...
        """

        self.Pixels = np.zeros(Shape, np.uint8)

        return self

    def Append(self: ZStack, ToAppend: np.ndarray) -> ZStack:
        """
        Append

        This function...

        ToAppend:
            ...

        Return (ZStack):
            ...
        """

        if ( self.Pixels is None ):
            self.Pixels = ToAppend.copy().reshape((1,) + ToAppend.shape)
        else:
            self.Pixels = np.append(self.Pixels, ToAppend.copy().reshape((1,) + ToAppend.shape), axis=0)

        return self

    def InsertLayer(self: ZStack, ToInsert: np.ndarray, LayerIndex: int) -> ZStack:
        """
        InsertLayer

        This function...

        ToInsert:
            ...
        LayerIndex:
            ...

        Return (ZStack):
            ...
        """

        if ( self.Pixels is None ) or ( self.Pixels.shape[0] < LayerIndex ):
            self._LogWriter.Errorln(f"Pixels is not initialized or large enough!")
            return self

        self.Pixels[LayerIndex,:,:] = ToInsert

        return self

    def Layers(self: ZStack) -> typing.Sequence[np.ndarray]:
        """
        Layers

        This function...

        Return (Sequence(np.ndarray)):
            ...
        """

        if ( self.Pixels is None ):
            return []

        if ( not self.IsZStack() ):
            return self.Pixels.reshape((1,) + self.Pixels.shape)

        # return [x for x in self.Pixels[0:5,:]]
        return self.Pixels

    def LayerCount(self: ZStack) -> int:
        """
        LayerCount

        This function...

        Return (int):
            ...
        """

        if ( self.Pixels is None ):
            return 0

        return self.Pixels.shape[0] if self.IsZStack() else 1

    def IsZStack(self: ZStack) -> bool:
        """
        IsZStack

        This function...

        Return (bool):
            ...
        """

        if ( self.Pixels is None ):
            return False

        return len(self.Pixels.shape) >= 3

    def SplitTimeSeries(self: ZStack) -> typing.Sequence[ZStack]:
        """
        SplitTimeSeries

        This function...

        Return (Sequence[ZStack]):
            ...
        """

        TimePoints: typing.List[ZStack] = []

        #   Z,Y,X,T,C
        if ( len(self.Pixels.shape) == 5 ):
            for T in range(self.Pixels.shape[3]):
                t: ZStack = ZStack(LogWriter=self._LogWriter, Name=f"{self.Name} - {T=:}")
                t.Pixels = self.Pixels[:,:,:,T,:].copy()
                if ( t.Pixels.shape[-1] == 1 ):
                    t.Pixels = np.squeeze(t.Pixels, axis=-1)
                TimePoints.append(t)
        else:
            TimePoints = [self]

        return TimePoints

    def SplitChannels(self: ZStack) -> typing.Sequence[ZStack]:
        """
        SplitChannels

        This function...

        Return (Sequence[ZStack]):
            ...
        """

        Channels: typing.List[ZStack] = []

        #   Z,Y,X,T,C
        if ( len(self.Pixels.shape) == 5 ):
            for C in range(self.Pixels.shape[4]):
                c: ZStack = ZStack(LogWriter=self._LogWriter, Name=f"{self.Name} - {C=:}")
                c.Pixels = self.Pixels[:,:,:,:,C].copy()
                if ( c.Pixels.shape[-1] == 1 ):
                    c.Pixels = np.squeeze(c.Pixels, axis=-1)
                Channels.append(c)
        else:
            Channels = [self]

        return Channels

    def Display(self: ZStack) -> None:
        """
        Display

        This function...

        Return (None):
            ...
        """

        if ( self.Pixels is None ):
            return 0

        CurrentLayer: int = 0
        Key: int = 0
        while ( Key not in [ord(x) for x in "qQ"] ):

            Key = DisplayImage(
                Description=f"Z-Stack {self.Name} - Layer {CurrentLayer+1}/{self.Pixels.shape[0]}",
                Image=ConvertTo8Bit(self.Pixels[CurrentLayer,:]),
                HoldTime=0
            )

            if ( Key in [ord(x) for x in 'uU'] ):
                CurrentLayer += 1
                if ( CurrentLayer >= self.Pixels.shape[0] ):
                    CurrentLayer = self.Pixels.shape[0] - 1
            elif ( Key in [ord(x) for x in 'dD'] ):
                CurrentLayer -= 1
                if ( CurrentLayer < 0 ):
                    CurrentLayer = 0

        return

    def OpenLIFFile(self: ZStack, Filename: str, *, SeriesName: str = "", SeriesIndex: int = -1, ChannelIndex: int = -1) -> bool:
        """
        OpenLIFFile

        This function...

        Filename:
            ...
        SeriesName:
            ...
        SeriesIndex:
            ...
        ChannelIndex:
            ...

        Return (bool):
            ...
        """

        #   Assert the arguments are provided...
        if ( Filename is None ) or ( Filename == "" ):
            self._LogWriter.Errorln(f"Failed to open Z-Stack from LIF file, no filename provided.")
            return False

        if ( not os.path.exists(Filename) ):
            raise ValueError(f"File [ {Filename} ] cannot be opened as the file does not exist")

        #   Assert the arguments are provided...
        if (( SeriesName is None ) or ( SeriesName == "" )) and ( SeriesIndex < 0 ):
            self._LogWriter.Errorln(f"Failed to open Z-Stack from LIF file, no image series provided.")
            return False

        #   Try to parse the file as a LIF File, using the 3rd party "readlif" library
        try:
            #   Open and parse the file into a LifFile instance...
            LifStack: LifFile = LifFile(Filename)

            #   Define the variable to hold the actual stack extracted from the file.
            Stack: LifImage = None

            if ( SeriesIndex >= 0 ):
                Stack = LifStack.get_image(SeriesIndex)
                if ( self.Name == "" ):
                    self.SetName(Stack.name)

            elif ( SeriesName != "" ):
                for Index, Series in enumerate(LifStack.image_list):
                    if ( Series['name'].lower() == SeriesName.lower() ):
                        Stack = LifStack.get_image(Index)
                        if ( self.Name == "" ):
                            self.SetName(SeriesName)
                        break
                else:
                    self._LogWriter.Errorln(f"No series was found by the name [ {SeriesName} ]...")
                    return False

            #   Get the dimensions of the resulting series.
            #   These correspond to the (x, y, z) size of the images, as well as the
            #   possible time and colour-channel sequences.
            X, Y, Z, T, C = Stack.dims.x, Stack.dims.y, Stack.dims.z, Stack.dims.t, Stack.channels

            #   Identify the bit depth of the stack...
            BitDepth: int = Stack.bit_depth[0]

            #   Ensure the pixel array is created with the correct size and bit depth to support the image data...
            BitDepth = int(math.ceil(BitDepth / 8.0) * 8)
            self.Pixels: np.ndarray = None
            if ( ChannelIndex >= 0 ):
                #   For specifically requested channel, just allocate for one.
                self.Pixels = np.zeros(shape=(Z, Y, X, T, 1), dtype=f"uint{BitDepth}")
            else:
                self.Pixels = np.zeros(shape=(Z, Y, X, T, C), dtype=f"uint{BitDepth}")

            for t in range(T):
                if ( ChannelIndex >= 0 ):
                    for z, Layer in enumerate(Stack.get_iter_z(t=t, c=ChannelIndex)):
                        self.Pixels[z,:,:,t,0] = Layer
                else:
                    for c in range(C):
                        for z, Layer in enumerate(Stack.get_iter_z(t=t, c=c)):
                            self.Pixels[z,:,:,t,c] = Layer

        except:
            return False

        #   If there is no depth in the "Time" or "Channel" dimensions, then
        #   squeeze them down to a basic Z stack.
        if ( self.Pixels.shape[3] == 1 ) and ( self.Pixels.shape[4] == 1 ):
            self.Pixels = np.squeeze(self.Pixels, (3,4))

        return True

    def OpenTIFFile(self: ZStack, Filename: str) -> bool:
        """
        OpenTIFFile

        This function...

        Filename:
            ...

        Return (bool):
            ...
        """

        if ( Filename is None ) or ( Filename == "" ):
            self._LogWriter.Errorln(f"Failed to open Z-Stack from TIF file, no filename provided.")
            return False

        if ( not os.path.exists(Filename) ):
            raise ValueError(f"File [ {Filename} ] cannot be opened as the file does not exist")

        try:
            Success, ImageStack = cv2.imreadmulti(Filename, [], cv2.IMREAD_ANYDEPTH)
            if not ( Success ):
                raise ValueError(f"Image file [ {Filename} ] cannot be parsed by cv2.imreadmulti().")

            self.Pixels = np.array(ImageStack)
        except:
            return False

        return True

    def OpenCZIFile(self: ZStack, Filename: str) -> bool:
        """
        OpenCZIFile

        This function...

        Filename:
            ...

        Return (bool):
            ...
        """

        if ( Filename is None ) or ( Filename == "" ):
            self._LogWriter.Errorln(f"Failed to open Z-Stack from CZI file, no filename provided.")
            return False

        if ( not os.path.exists(Filename) ):
            raise ValueError(f"File [ {Filename} ] cannot be opened as the file does not exist")

        try:
            TIFFilename: str = Filename.replace(".czi", ".tif")
            self._LogWriter.Println(f"Converting from *.czi file to *.tif file...")
            czifile.czi2tif(Filename, TIFFilename)
            return self.OpenTIFFile(TIFFilename)
        except Exception as e:
            self._LogWriter.Errorln(f"Exception raised while attempting to open CZI Z-Stack: [ {e} ]\n\n{''.join(traceback.format_exception(e, value=e, tb=e.__traceback__))}\n")
            return False

    def SetName(self: ZStack, Name: str) -> ZStack:
        """
        SetName

        This function...

        Name:
            ...

        return (self):
            ...
        """

        self.Name = Name
        return self

    def MaximumIntensityProjection(self: ZStack, Axis: str = 'z') -> np.ndarray:
        """
        MaximumIntensityProjection

        This function computes and prepares the Maximum Intensity Projection from
        the Z-Stack, returning a single 2D image consiting of the collection of the
        brightest pixel values from any slice through the stack. This is a commonly
        used projection method for operating with Z-Stack images as a 'smaller' 2D
        image, ideally without losing too much information.

        Z_Stack:
            The current open Z-stack to compute the projection of.
        Axis:
            Which axis of the Z-Stack should be collapsed in the MIP.

        Return (np.ndarray):
            The resulting 2D NumPy array of the MIP image. The pixel values are
            scaled to the full range of the bit depth of the image.
        """

        if ( not self.IsZStack() ):
            return self.Pixels

        axis: int = 0
        if ( Axis.lower() == 'z' ):
            axis = 0
        elif ( Axis.lower() == 'y' ):
            axis = 1
        elif ( Axis.lower() == 'x' ):
            axis = 2
        else:
            raise ValueError(f"Projection Axis must be one of [ 'x', 'y', 'z' ]. Got [ '{Axis}' ]")

        #   Given that the Z_Stack has the 0th axis corresponding to each Z-Slice through the stack,
        #   the maximum intensity projection (MIP) is computed as the maximum pixel value over the
        #   0th axis of the 3D array.
        Projection: np.ndarray = np.max(self.Pixels, axis=axis)

        #   Return the projection to the user.
        return Projection

    def AverageIntensityProjection(self: ZStack, Axis: str = 'z') -> np.ndarray:
        """
        AverageIntensityProjection

        This function computes and prepares the Average Intensity Projection from
        the Z-Stack, returning a single 2D image consiting of the collection of the
        brightest pixel values from any slice through the stack. This is a commonly
        used projection method for operating with Z-Stack images as a 'smaller' 2D
        image, ideally without losing too much information.

        Z_Stack:
            The current open Z-stack to compute the projection of.
        Axis:
            Which axis of the Z-Stack should be collapsed in the projection.

        Return (np.ndarray):
            The resulting 2D NumPy array of the MIP image. The pixel values are
            scaled to the full range of the bit depth of the image.
        """

        if ( not self.IsZStack() ):
            return self.Pixels

        axis: int = 0
        if ( Axis.lower() == 'z' ):
            axis = 0
        elif ( Axis.lower() == 'y' ):
            axis = 1
        elif ( Axis.lower() == 'x' ):
            axis = 2
        else:
            raise ValueError(f"Projection Axis must be one of [ 'x', 'y', 'z' ]. Got [ '{Axis}' ]")

        #   Given that the Z_Stack has the 0th axis corresponding to each Z-Slice through the stack,
        #   the maximum intensity projection (MIP) is computed as the maximum pixel value over the
        #   0th axis of the 3D array.
        Projection: np.ndarray = np.mean(self.Pixels, axis=axis)

        #   Return the projection to the user.
        return Projection

    def MinimumIntensityProjection(self: ZStack, Axis: str = 'z') -> np.ndarray:
        """
        MinimumIntensityProjection

        This function computes and prepares the Minimum Intensity Projection from
        the Z-Stack, returning a single 2D image consiting of the collection of the
        dimmest pixel values from any slice through the stack. This is a commonly
        used projection method for operating with Z-Stack images as a 'smaller' 2D
        image, ideally without losing too much information.

        Z_Stack:
            The current open Z-stack to compute the projection of.
        Axis:
            Which axis of the Z-Stack should be collapsed in the MIP.

        Return (np.ndarray):
            The resulting 2D NumPy array of the MIP image. The pixel values are
            scaled to the full range of the bit depth of the image.
        """

        if ( not self.IsZStack() ):
            return self.Pixels

        axis: int = 0
        if ( Axis.lower() == 'z' ):
            axis = 0
        elif ( Axis.lower() == 'y' ):
            axis = 1
        elif ( Axis.lower() == 'x' ):
            axis = 2
        else:
            raise ValueError(f"Projection Axis must be one of [ 'x', 'y', 'z' ]. Got [ '{Axis}' ]")

        #   Given that the Z_Stack has the 0th axis corresponding to each Z-Slice through the stack,
        #   the minimum intensity projection (MIP) is computed as the minimum pixel value over the
        #   0th axis of the 3D array.
        Projection: np.ndarray = np.min(self.Pixels, axis=axis)

        #   Return the projection to the user.
        return Projection

    def SaveTIFF(self: ZStack, Folder: str) -> bool:
        """
        SaveTIFF

        This function...

        Folder:
            ...

        Return (bool):
            ...
        """

        if ( self.Name is None ) or ( self.Name == "" ):
            raise ValueError(f"Z Stack Name must be set!")

        if ( Folder is None ) or ( Folder == "" ):
            raise ValueError(f"Output Folder must be set!")

        if ( not os.path.exists(Folder) ):
            self._LogWriter.Println(f"Folder [ {Folder} ] does not exist. Creating it now...")
            os.makedirs(Folder, 0o755, exist_ok=True)

        if ( self.Pixels is None ):
            self._LogWriter.Warnln(f"Z Stack [ {self.Name} ] contains no pixel data...")
            return False

        SeriesName: str = self.Name.replace("/", "-")
        SeriesName: str = SeriesName.replace("\\", "-")

        self._LogWriter.Println(f"Writing out Z-Stack as file [ {Folder}/{SeriesName}.tif ]...")
        return cv2.imwritemulti(os.path.join(Folder, f"{SeriesName}.tif"), [ConvertTo8Bit(x) for x in self.Pixels])

    #   ...

    ### Private Methods
    #   ...


#   Define the globals to set by the command-line arguments
LogWriter: Logger = Logger(Prefix="LIF-Previewer.py")
DefaultHoldTime: int = 3
#   ...

def DisplayImage(Description: str = "", Image: np.ndarray = None, HoldTime: int = DefaultHoldTime, Topmost: bool = False, ShowOverride: bool = True) -> int:
    """
    DisplayImage

    This function displays a given image to the screen in an OpenCV NamedWindow, allowing viewing of the image.

    Description:
        The title of the window to display the image in.
    Image:
        The image to be displayed.
    HoldTime:
        The duration (in seconds) to display the image for. 0 indicates to
        display indefinitely.
    Topmost:
        Boolean flag for whether the window should be displayed in front of all
        other windows.
    ShowOverride:
        An override to disable showing images when set to False. Allows simpler
        enabling/disabling during logging or development.

        The key-code which was pressed during display of the image, if any.
    """
    return DisplayImages(Images=[(Description, Image)], HoldTime=HoldTime, Topmost=Topmost, ShowOverride=ShowOverride)

def DisplayImages(Images: typing.List[typing.Tuple[str, np.ndarray]] = ["", None], HoldTime: int = DefaultHoldTime, Topmost: bool = False, ShowOverride: bool = True) -> int:
    """
    DisplayImages

    Like DisplayImage(), but for more than one image.

    Multiple images will be displayed in distinct Windows, and can be rearranged as desired.

    Images:
        A list of tuples, containing the window name and image pixels to display
    HoldTime:
        The duration (in seconds) to display the image for. 0 indicates to
        display indefinitely.
    Topmost:
        Boolean flag for whether the window should be displayed in front of all
        other windows.
    ShowOverride:
        An override to disable showing images when set to False. Allows simpler
        enabling/disabling during logging or development.

    Return (int):
        The key-code which was pressed during display of the image, if any.
    """

    if ( not ShowOverride ):
        return

    if ( HoldTime <= 0 ):
        HoldTime = 0

    WindowFlags: int = cv2.WINDOW_NORMAL

    #   Keep a record of the descriptions generated for the un-described images to display,
    #       to be able to destroy their display windows later.
    UnmanagedDescriptions: typing.List[str] = []
    ImagesActive: bool = False

    XPos: int = 0
    YPos: int = 0
    for Index, (Description, Image) in enumerate(Images):

        if ( Image is None ):
            LogWriter.Warnln(f"Provided \"Image\" {Index} is None, nothing to display.")
            continue

        ImagesActive = True
        if ( Description is None ) or ( Description == "" ):
            LogWriter.Warnln(f"No \"Description\" provided for Image {Index}...")
            Description = f"Display Image {Index} - {Image.shape[1]}x{Image.shape[0]}"
            if ( len(Image.shape) == 3 ):
                Description += " (RGB)"
            elif ( len(Image.shape) == 4):
                Description += " (RGBA)"
            UnmanagedDescriptions.append(Description)

        if ( XPos != 0 ) and ( XPos + Image.shape[1] >= 1440 ):
            YPos += Images[Index-1][1].shape[0]
            XPos = 0

        cv2.namedWindow(Description, WindowFlags)
        if ( Topmost ):
            cv2.setWindowProperty(Description, cv2.WND_PROP_TOPMOST, 1)
        cv2.moveWindow(Description, XPos, YPos)
        cv2.imshow(Description, Image)

        XPos += Image.shape[1]
        if ( XPos >= 1440 ):
            YPos += Image.shape[0]
            XPos = 0

    Key: int = 0
    if ( ImagesActive ):
        Key = cv2.waitKeyEx(round(HoldTime * 1000))

        #   Allow pressing the "P" key to pause, overriding the HoldTime setting until another key is pressed.
        while ( Key in [ord(x) for x in 'pP']):
            LogWriter.Println(f"[ P ] key pressed, pausing display until another key is pressed...")
            Key = cv2.waitKeyEx(0)

        #   Allow for the "S" key to save all currently displayed images to disk in the local directory.
        if ( Key in [ord(x) for x in 'sS' ]):
            [cv2.imwrite(os.path.join(os.getcwd(), os.path.splitext(Description)[0] + '.png'), Image) for (Description, Image) in Images]

        [cv2.destroyWindow(Description) for (Description, _) in Images]
        [cv2.destroyWindow(Description) for Description in UnmanagedDescriptions]
        _ = cv2.waitKeyEx(1)

    return Key

def GammaCorrection(Image: np.ndarray = None, Gamma: float = 1.0, Minimum: int = None, Maximum: int = None) -> np.ndarray:
    """
    GammaCorrection

    This function applies the standard gamma-based rescaling algorithm of image
    brightness values.  First, the image values are exponentiated to the
    Gamma'th power, then linearly rescaled to the range defined by Minimum and
    Maximum.

    A Gamma value of 1 indicates a linear contrast stretch to fill out the full
    range [Minimum, Maximium].

    Image:
        The image to have the brightness values rescaled for.
    Gamma:
        The exponent to raise the pixel values by during the rescaling.
    Minimum:
        The final minimum brightness value to rescale to.
    Maximum:
        The final maximum brightness value to rescale to.

    Return (np.ndarray):
        A new np.ndarray instance containing the brightness-scaled original image.
    """

    if ( Image is None ) or ( len(Image) == 0 ):
        raise ValueError(f"Image must be provided.")

    if ( 0 >= Gamma ):
        raise ValueError(f"Gamma must be provided and be a positive real number.")

    OriginalDtype: np.dtype = Image.dtype
    Limits = None
    if ( np.issubdtype(OriginalDtype, np.integer)):
        Limits = np.iinfo(OriginalDtype)
        if ( Minimum is None ):
            Minimum = Limits.min
        if ( Maximum is None ):
            Maximum = Limits.max
    elif ( np.issubdtype(OriginalDtype, np.floating)):
        Limits = np.finfo(OriginalDtype)
        #
        if ( Minimum is None ):
            Minimum = 0.0
        if ( Maximum is None ):
            Maximum = 1.0
    else:
        raise TypeError(f"Numpy NDArray has non-integral and non-floating point dtype!")

    #   Perform the non-linear exponentiation operation
    #       allowing short-cutting for the no-op of exponentiation by 1.
    Scaled = Image.astype(np.float64)
    if ( Gamma != 1.0 ):
        Scaled = Scaled ** Gamma

    #   Linearly re-scale the resulting image to the desired min/max range provided.
    Offset = np.min(Scaled) - Minimum
    if ( Offset != 0.0 ):
        Scaled -= Offset

    if ( np.max(Scaled) != 0 ):
        ScaleFactor = Maximum / np.max(Scaled)
        if ( ScaleFactor != 1.0 ):
            Scaled *= ScaleFactor

    return Scaled.astype(OriginalDtype)

def ConvertTo8Bit(Image: np.ndarray) -> np.ndarray:
    """
    ConvertTo8Bit

    This function will take the current image and both convert it to 8-bit, while
    also scaling the brightness linearly to fill the full range [0, 255].

    Image:
        The original image, to convert and linearly rescale brightness values for.

    Return (np.ndarray):
        A new np.ndarray instance, containing the converted and brightness-scaled
        pixel values.
    """

    if ( Image is None ):
        raise ValueError(f"Image must not be None")

    return GammaCorrection(Image=Image.copy(), Gamma=1, Minimum=0, Maximum=255).astype(np.uint8)

#   Main
#       This is the main entry point of the script.
def main() -> int:

    SourceDirectory: str = os.getcwd()
    if ( len(sys.argv) >= 2 ):
        SourceDirectory = sys.argv[1]

    LogWriter.Println(f"Searching for *.LIF files starting from [ {SourceDirectory} ]...")
    for ImageFile in glob.glob(pathname="*.lif", root_dir=SourceDirectory, recursive=True):
        LogWriter.Println(f"Found *.LIF file [ {ImageFile} ]...")

        FullPath: str = os.path.join(SourceDirectory, ImageFile)
        LIFFile: ZStack.LifFile = ZStack.LifFile(FullPath)

        LogWriter.Println(f"This file contains a total of [ {LIFFile.num_images} ] image series:")
        for SeriesIndex in range(LIFFile.num_images):

            CurrentImage: ZStack.LifImage = LIFFile.image_list[SeriesIndex]
            LogWriter.Println(f"{SeriesIndex+1}/{LIFFile.num_images} - Series Name: {CurrentImage['name']} - Dimensions: {CurrentImage['dims']}")

            if ( CurrentImage['dims'].m > 1 ):
                LogWriter.Println(f"This series corresponds to the unstitched tiles of the next series. This will not be displayed.")
                continue

            LogWriter.Println(f"This series contains [ {CurrentImage['channels']} ] colour channels.")
            for ChannelIndex in range(CurrentImage['channels']):
                LogWriter.Println(f"Preparing preview of Series [ {CurrentImage['name']} ], Channel [ {ChannelIndex+1} ]...")
                Stack: ZStack.ZStack = ZStack.ZStack.FromLIF(FullPath, SeriesIndex=SeriesIndex, ChannelIndex=ChannelIndex)

                MinProjection, MaxProjection = Stack.MinimumIntensityProjection(), Stack.MaximumIntensityProjection()

                DisplayImages([
                        (f"Maximum Intensity Projection - Series {CurrentImage['name']} - Channel [ {ChannelIndex+1} ]", MaxProjection),
                        (f"Minimum Intensity Projection - Series {CurrentImage['name']} - Channel [ {ChannelIndex+1} ]", MinProjection),
                    ],
                    HoldTime=0,
                    Topmost=True
                )

        LogWriter.Println(f"Finished previewing *.LIF file [ {ImageFile} ].")
    else:
        LogWriter.Println(f"No *.LIF files found starting from [ {SourceDirectory} ]!")

    return 0

#   Allow this script to be called from the command-line and execute the main function.
#   If anything needs to happen before executing main, add it here before the call.
if __name__ == "__main__":
    main()
