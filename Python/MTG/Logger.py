#!/usr/bin/env python3

##      Author:     Joseph Sadden
##      Date:       29th November, 2021

from __future__ import annotations

#   Import the standard library modules required.
from datetime import datetime
import os
import sys
from typing import TextIO

#   Import the local modules and classes required.
#   ...

#   Import the third-part modules and classes required.
#   ...

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
