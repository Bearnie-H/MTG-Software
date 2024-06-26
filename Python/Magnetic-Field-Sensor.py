#!/usr/bin/env python3

#   Author: Joseph Sadden
#   Date:   18th December, 2023

#   Script Purpose: This script runs the download side of the magnetic sensor array
#                       device, reading the measurements and device identifier details
#                       and transforming the raw readings from the array into a set of
#                       human-interpretable and meaningful output formats.

#   Import the necessary standard library modules
from __future__ import annotations

import argparse
from datetime import datetime
import itertools
import math
import os
import serial
import serial.tools.list_ports
import serial.tools.list_ports_common
import signal
import sys
import threading
import time
import traceback
import typing
#   ...

#   Import the necessary third-part modules
import numpy as np
import matplotlib
from matplotlib.axes import Axes
from matplotlib import cm
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
#   ...

#   Import the desired locally written modules
from MTG_Common import Logger
#   ...

#   Define the globals to set by the command-line arguments
LogWriter: Logger.Logger = Logger.Logger(Prefix=os.path.basename(sys.argv[0]))
#   ...

#   Define the global constants of the program
MagneticFieldOffset: int = 0        #   LSB - Corresponds to 0 Ga
MagneticFieldSensitivity: int = 1   #   LSB/gauss

TemperatureOffset: int = 1708       #   LSB - Corresponds to 0 deg C
TemperatureSensitivity: float = (302.0 / 4096.0) #  LSB/deg C

#   Position Look-Up tables for the 4 different indexing corners.
PositionLookup_A: typing.Dict[int, typing.Tuple[float, float]] = {
    0: (-5.964,  6.342 ),
    1: (-5.964,  12.842),
    2: (-5.964,  19.342),
    3: (-12.714, 9.592 ),
    4: (-12.714, 16.092),
    5: (-19.464, 6.342),
    6: (-19.464, 12.842),
    7: (-19.464, 19.342),
}

PositionLookup_B: typing.Dict[int, typing.Tuple[float, float]] = {
    0: (-5.964,  -14.994),
    1: (-5.964,  -8.494 ),
    2: (-5.964,  -1.994 ),
    3: (-12.714, -11.744),
    4: (-12.714, -5.244 ),
    5: (-19.464, -14.994),
    6: (-19.464, -8.494 ),
    7: (-19.464, -1.994 ),
}

PositionLookup_C: typing.Dict[int, typing.Tuple[float, float]] = {
    0: (18.420, -14.994),
    1: (18.420, -8.494 ),
    2: (18.420, -1.994 ),
    3: (11.670, -11.744),
    4: (11.670, -5.244 ),
    5: (4.920,  -14.994),
    6: (4.920,  -8.494 ),
    7: (4.920,  -1.994 ),
}

PositionLookup_D: typing.Dict[int, typing.Tuple[float, float]] = {
    0: (18.420, 6.342 ),
    1: (18.420, 12.842),
    2: (18.420, 19.342),
    3: (11.670, 9.592 ),
    4: (11.670, 16.092),
    5: (4.920,  6.342 ),
    6: (4.920,  12.842),
    7: (4.920,  19.342),
}

SerialMessageLength: int = 16   #   bytes, including start and stop bytes.

GaussToMilliTesla: float = 1.0 / 10.0   #   Conversion factor from Ga to mT

class Configuration():
    """
    Configuration

    This class represents the set of top-level application configuration
    settings available to be modified.
    """

    def __init__(self: Configuration, LogWriter: Logger.Logger = Logger.Discarder) -> None:
        """
        Constructor

        This function prepares the application configuration into a known initial state.

        LogWriter:
            The logger to use for writing all messages out to the user.

        Return (None):
            None, the configuration instance is initialized and ready to be used.
        """

        self._LogWriter: Logger.Logger = LogWriter

        self.StreamType: str = None

        self.SerialPort: serial.Serial = None
        self.SerialDevice: str = None
        self.BaudRate: int = 0
        self.MeasurementRate: float = None

        self.MeasurementsFilename: str = None
        self.OutputDirectory: str = "."

        self.PositionReference: str = None
        self.SensorOffset_Z: float = 1.6 + 0.65    #   mm
        self.SensorPitch_Z: float = 0.0

        self.EnableDryRun: bool = False

        return

class RawSensorReading():
    """
    RawSensorReading

    This class respresents a single raw reading from one of the N sensing
    elements connected within the magnetic sensor array. This is a simple record
    type to capture and format the raw bytes transmitted over the serial
    interface from the micro-controller, with absolutely no formatting or data
    manipulation applied.
    """

    #   The I2C address of the device this record came from
    I2C_Address: int

    #   The layer and intra-layer position information for the particular device.
    DeviceIndex: int
    LayerIndex: int

    #   The raw sensor values for the X, Y, Z, and T measurements as seen by this sensor.
    Field_X: int
    Field_Y: int
    Field_Z: int
    Temperature: int

    #   The timestamp of this sensor reading, as prepared by the micro-controller.
    Timestamp: int

    def Format(self: RawSensorReading) -> FormattedSensorReading:
        """
        Format

        This function converts the raw sensor reading into the corresponding set
        of dimensional values.  This is used to transform from raw bits to
        dimensonal values to be used for proper analysis and interpretation.

        Return (FormattedSensorReading):
            A new instance of a FormattedSensorReading, containing the dimensional
            values this reading corresponds to.
        """

        #   Convert the raw field values to dimensional values
        FieldX: float = ((self.Field_X - MagneticFieldOffset) / float(MagneticFieldSensitivity)) * GaussToMilliTesla
        FieldY: float = ((self.Field_Y - MagneticFieldOffset) / float(MagneticFieldSensitivity)) * GaussToMilliTesla
        FieldZ: float = ((self.Field_Z - MagneticFieldOffset) / float(MagneticFieldSensitivity)) * GaussToMilliTesla

        #   Convert the temperature from bits to degrees Celcius
        Temperature: float = ComputeFormattedTemperature(self.Temperature)

        #   Determine the X, Y, Z position of this sensor, relative to the particular corner
        #   specified as the reference origin.
        X, Y, Z = PositionLookup(self.LayerIndex, self.DeviceIndex, Config.PositionReference)

        #   Compute the index into the numpy arrays used for plotting, where this reading should be plotted.
        Index = self.LayerIndex * 2**3 + self.DeviceIndex

        #   Format
        return FormattedSensorReading(np.array([FieldX, FieldY, FieldZ]), np.array([X, Y, Z]), Temperature, self.Timestamp / 1e6, Index)

    def Valid(self: RawSensorReading) -> bool:
        """
        Valid

        This function performs a basic check of the raw values, to check if
        there was corruption or bit flips on the Serial line. This simply checks
        that the values of this reading fall within the known bounds for each
        value.

        Return (bool):
            True if no value is obviously wrong, False if at least one is.
        """

        #   LayerIndex is limited to the range [0, 7]
        if not ( 0 <= self.LayerIndex < 8 ):
            LogWriter.Warnln(f"Invalid RawSensorReading: LayerIndex out of range ({self.LayerIndex})")
            return False

        #   DeviceIndex is limited to the range [0, 7]
        if not ( 0 <= self.DeviceIndex < 8 ):
            LogWriter.Warnln(f"Invalid RawSensorReading: DeviceIndex out of range ({self.DeviceIndex})")
            return False

        #   The I2C Address must be derived from the LayerIndex and DeviceIndex according to this equation.
        if ( self.I2C_Address != (self.LayerIndex * 16 + self.DeviceIndex + 8 )):
            LogWriter.Warnln(f"Invalid RawSensorReading: Invalid I2C Address ({self.I2C_Address} vs. {(self.LayerIndex * 16 + self.DeviceIndex + 8 )})")
            return False

        #   The X, Y, Z magnetic field values must be no more than signed 12-bit values.
        for Field, Axis in zip([self.Field_X, self.Field_Y, self.Field_Z], ["X", "Y", "Z"]):
            if not ( -2048 < Field <= 2047 ):
                LogWriter.Warnln(f"Invalid RawSensorReading: Field {Axis} out of range ({Field})")
                return False

        #   The temperature field value must be an unsigned 12-bit value
        if not ( 0 <= self.Temperature < 4096 ):
            LogWriter.Warnln(f"Invalid RawSensorReading: Temperature out of range ({self.Temperature})")
            return False

        return True

    def CSVHeader(self: RawSensorReading) -> str:
        """
        CSVHeader

        This function provides the column headers associated with the ToString()
        method to describe the column ordering when writing to a CSV file for
        data recording.

        Return (str):
            A comma-separated set of column identifiers, matching the order of
            the data values provided by the ToString() method.
        """

        return f"I2C Address,Layer Index,Device Index,Field X, Field Y, Field Z,Temperature,Timestamp\r\n"

    def ToString(self: RawSensorReading) -> str:
        """
        ToString

        This function produces a standard stringified interpretation of the
        values of this record, in a standard format, resolution, and ordering.
        This is suitable for recording the raw sensor values in a text-type file
        for later review.

        Return (str):
            A comma-separated set of data values, in a standard order.
        """

        return f"{self.I2C_Address:3d},{self.LayerIndex},{self.DeviceIndex},{self.Field_X:+4d},{self.Field_Y:+4d},{self.Field_Z:+4d},{self.Temperature:4d},{self.Timestamp}"

class FormattedSensorReading():
    """
    FormattedSensorReading

    This class represents a dimensional interpretation of a RawSensorReading() instance.
    """

    #   3D position, relative to one out of a set of user-selectable reference points
    Position: np.ndarray

    #   3D vector, in units of milliTesla
    MagneticField: np.ndarray

    #   Temperature, in units of degrees Celcius
    Temperature: float

    #   A timestamp, in units of seconds
    Timestamp: float

    #   The index into the "global" numpy arrays used for the live plotting of the temperature and magnetic fields.
    Index: int

    def __init__(self: FormattedSensorReading, MagneticField: np.ndarray = np.array([0, 0, 0]), Position: np.ndarray = np.array([0, 0, 0]), Temperature: float = 0.0, Timestamp: int = 0, Index: int = 0) -> None:
        """
        Constructor

        This function constructs a FormattedSensorReading from the values computed from a RawSensorReading.

        MagneticField:
            The magnetic field, as an [X, Y, Z] vector in units of mT.
        Position:
            The [X, Y, Z] position vector associated with the sensor which produced this reading, in units of mm.
        Temperature:
            The temperature, in units of degrees Celcius
        Timestamp:
            The timestamp provided by the micro-controller associated with this reading.

        Return (None):
            None, the instance is initialized with the provided field values
        """

        self.MagneticField = MagneticField
        self.Position = Position
        self.Temperature = Temperature
        self.Timestamp = Timestamp
        self.Index = Index

        return

    def CSVHeader(self: FormattedSensorReading) -> str:
        """
        CSVHeader

        This function provides the column headers associated with the ToString()
        method to describe the column ordering when writing to a CSV file for
        data recording.

        Return (str):
            A comma-separated set of column identifiers, matching the order of
            the data values provided by the ToString() method.
        """

        return f"X (mm),Y (mm),Z (mm),U (mT),V (mT), W (mT),T (deg C),Timestamp (s)\r\n"

    def ToString(self: FormattedSensorReading) -> str:
        """
        ToString

        This function produces a standard stringified interpretation of the
        values of this record, in a standard format, resolution, and ordering.
        This is suitable for recording the raw sensor values in a text-type file
        for later review.

        Return (str):
            A comma-separated set of data values, in a standard order.
        """

        return f"{self.Position[0]:+2.4f},{self.Position[1]:+2.4f},{self.Position[2]:+2.4f},{self.MagneticField[0]:+4.1f},{self.MagneticField[1]:+4.1f},{self.MagneticField[2]:+4.1f},{self.Temperature:+3.2f},{self.Timestamp * 1e-6:.6f}"

class MeasurementStream():
    """
    MeasurementStream

    This class represents an abstract "stream" of sensor readings, which can be
    read and processed sequentially.
    """

    LogWriter: Logger.Logger
    ArduinoLogWriter: Logger.Logger
    SerialPort: serial.Serial
    Filename: str

    Measurements: typing.List[RawSensorReading]

    _MonitorThread: threading.Thread
    _MonitorActive: bool

    def __init__(self: MeasurementStream, LogWriter: Logger.Logger = Logger.Discarder, SerialPort: serial.Serial = None, Filename: str = None) -> None:
        """
        Constructor

        This function initializes the MeasurementStream, providing a Logger for
        it to write messages out to the user, and either a Serial Port to read
        from, or a filename to read measurements from.

        LogWriter:
            The logger to use for any messages out to the user.
        SerialPort:
            The pre-initialized serial port to read messages from. None if the
            file should be read instead.
        Filename:
            The name of the disk file to open and read measurements from. None
            if the Serial port should be read instead.

        Return (None):
            None, the MeasurementStream is initialized and ready to
            StartReading() in order to start processing sensor readings.
        """

        self.LogWriter = LogWriter
        self.ArduinoLogWriter = Logger.Logger(OutputStream=LogWriter.RawStream(), Prefix="Arduino")
        self.SerialPort: serial.Serial = SerialPort
        self.Filename = Filename

        self._MonitorThread: threading.Thread = None
        self._MonitorActive: bool = False

        self.Measurements: typing.List[RawSensorReading] = []

        return

    def __len__(self: MeasurementStream) -> int:
        """
        __len__

        This function returns the number of unprocessed sensor readings buffered and waiting to be read out.

        Return (int):
            The number of unprocessed sensor readings buffered and waiting to be read out.
        """

        return len(self.Measurements)

    def StartReading(self: MeasurementStream) -> None:
        """
        StartReading

        This function spawns a thread to read the underlying stream for raw
        readings, and present them for consumption by any other thread.

        Return (None):
            None, the new thread is spawned and all book-keeping for managing it
            is handled within the class.
        """

        if ( self.Filename is not None ):
            self.LogWriter.Println(f"Spawning thread to read file: {self.Filename}")
            self._MonitorActive = True
            self._MonitorThread: threading.Thread = threading.Thread(target=self._ReadFile)
            self._MonitorThread.start()
        elif ( self.SerialPort is not None ):
            self.LogWriter.Println(f"Spawning thread to read serial port: {self.SerialPort.name}")
            self._MonitorActive = True
            self._MonitorThread: threading.Thread = threading.Thread(target=self._ReadSerialPort)
            self._MonitorThread.start()
        else:
            self.LogWriter.Errorln(f"Failed to start MeasurementStream, neither Serial Port nor File streams were initialized!")

        return

    def Next(self: MeasurementStream) -> RawSensorReading:
        """
        Next

        This function returns the "next" buffered reading from the stream, if
        one exists.

        Return (RawSensorReading):
            The oldest buffered reading available, if one exists.
        """

        if ( len(self.Measurements) > 0 ):
            return self.Measurements.pop(0)

        if ( not self._MonitorActive ):
            self.LogWriter.Warnln(f"Cannot read next measurement from MeasurementStream, asynchronous thread is not active to read from the underlying stream!")
            return None

        return None

    def Active(self: MeasurementStream) -> bool:
        """
        Active

        This function checks whether this stream is still active. This is True
        if either the asynchronous thread is still active, or there are still
        buffered readings to process.

        Return (bool):
            True if the asynchronous thread is active or there are buffered
            measurements. False otherwise.
        """

        return self._MonitorActive or len(self.Measurements) > 0

    def Halt(self: MeasurementStream) -> None:
        """
        Halt

        This function ends the asynchronous thread, cleaning up the resources
        associated with it and waiting for it to complete before returning.

        Return (None):
            None, the spawned thread is halted and this function blocks until
            the other thread is finalized and fully cleaned up.
        """

        self._MonitorActive = False
        if ( self._MonitorThread is not None ):
            self._MonitorThread.join()

        return

    def _ReadSerialPort(self: MeasurementStream) -> None:
        """
        _ReadSerialPort

        This function is the main function run by the asynchronous thread to
        actually read the raw data being fed to the Serial port, convert this to
        a set of RawSensorReading instances, and present them to be consumed by
        the main thread of the program.

        Return (None):
            None, this function runs until either the Serial port is closed, or
            the MeasurementStream is Halted. Sensor readings are parsed and
            appended to the internal FIFO queue. Log messages from the
            micro-controller are separated and printed to the dedicated
            LogWriter.
        """

        try:

            #   Try to assert the serial port is open and readable.
            if ( not self.SerialPort.is_open ):
                self.SerialPort.open()

            #   We read from the serial port into a buffer, to allow stitching of
            #   sensor readings or log messages across read boundaries
            Buffer: bytearray = bytearray()

            #   Track the rate at which the MeasurementStream is actively receiving
            #   meausrement valeus over the port.
            StartTime: float = time.time()
            EndTime: float = 0.0
            Count: int = 0
            Limit: int = 128

            self.LogWriter.Println(f"Starting to read the Serial Port... Press CTRL+C to halt and exit the program.")

            #   Until either the serial port closes, or the stream is requested to halt...
            while ( self._MonitorActive ) and ( self.SerialPort.is_open ):

                #   Read a block of data from the serial port, accounting for both
                #   timeout in the case of not enough data to satisfy the read(),
                #   and for too much data to fit into the requested read size.
                RawBytes: bytes = self.SerialPort.read(min(Config.BaudRate, 512))

                #   Stitch this together with any existing data in the local buffer...
                Buffer.extend(RawBytes[:])

                #   If there's nothing to process, sleep to yield this thread and
                #   not over-aggressivly read the serial port.
                if ( len(Buffer) == 0 ):
                    self.LogWriter.Write(f"Waiting for serial data...\r")
                    time.sleep(1e-3)
                    continue

                #   While there's data yet to process within the local buffer...
                Done: bool = False
                while ( not Done ):

                    #   Check whether it's a sensor measurement, a log message, or a partial packet.
                    #   If something is successfully parsed, remove it from the buffer, moving
                    #   on to the "next" data to process.
                    Buffer, Measurement, Message, Done = self._ParseSerialBytes(Buffer)

                    #   If a complete meausrement is parsed out, append it to the FIFO queue and
                    #   attempt to update the sampling rate estimate.
                    if ( Measurement is not None ):
                        self.Measurements.append(Measurement)
                        Count += 1
                        if ( Count == Limit ):
                            EndTime = time.time()
                            Config.MeasurementRate = Count / (EndTime - StartTime)
                            Count, StartTime = 0, EndTime

                    #   Otherwise, if it's a message of non-zero length, print it out using the
                    #   dedicated micro-controller Logger
                    elif ( Message is not None ) and ( len(Message) > 0 ):
                        self.ArduinoLogWriter.Println(f'Log Message: {Message}')


            self.LogWriter.Println(f"_ReadSerialPort() exiting...")
            self._MonitorActive = False
            self.SerialPort.close()
        except:
            self.LogWriter.Errorln(f"Exception occured during _ReadSerialPort() - [ {e} ]\n\n{''.join(traceback.format_exception(e, value=e, tb=e.__traceback__))}\n")
            self._MonitorActive = False
            return

        return

    def _ParseSerialBytes(self: MeasurementStream, Buffer: bytearray) -> typing.Tuple[bytearray, RawSensorReading, str, bool]:
        """
        _ParseSerialBytes

        This function is what actually parses out either sensor readings or log
        messages from the local data buffer containing the values as read from
        the serial port.  This buffer contains some set of raw byte sensor
        readings (in a known format), interspersed with ASCII log messages from
        the micro-controller. This function identifies which is at the "front"
        of the buffer, reads the complete data record, and returns at most one
        non-None value to the caller. If only a partial record is present, the
        fourth element of the return tuple is used to indicate that processing
        on this buffer should wait until after the next read() of the serial
        port.

        Buffer:
            The current array of unprocessed bytes as read from the serial port.
            This is ordered such that lower indices correspond to earlier bytes,
            and will be processed to extract the set of sensor readings or log
            messages as they were written to the Serial port.

        Return (Tuple):
            [0] (bytearray):
                The (potentially modified) buffer, after removing the bytes
                successfully processed and parsed.
            [1] (RawSensorReading):
                The RawSensorReading instance corresponding to the raw data
                packet processed. This value is not None only if the first record
                in the buffer is successfully parsed as a sensor reading.
            [2] (str):
                The log message from the micro-controller to be printed out by
                the dedicated logger. This value is not None only if the first
                record in the buffer is successfully parsed as a log message.
                Trailing whitespace is stripped to ensure consistent display and
                printing of the message.
            [3] (bool):
                A boolean value indicating whether or not the buffer is
                exhausted and a new read() of the SerialPort is required.  This
                does not necessarily correspond to len(Buffer) == 0 being True,
                but rather that whatever contents the buffer contains is not a
                complete record.
        """

        #   Wrap this all in a try-catch block, as corrupted bits on the Serial
        #   port or invalid UTF-8 codepoints have a tendency to cause
        #   exceptions. If we get such a sequence, we just ignore it since we
        #   will override any individual reading in short order anyway.
        try:

            Measurement: RawSensorReading = None
            Message: str = None

            #   Nothing to read, nothing to process, and a new read() is required.
            if ( len(Buffer) == 0 ):
                return (Buffer, None, None, True)

            #   Sensor readings ALWAYS start with a NUL byte
            if ( Buffer[0] == 0x00 ):

                #   It's a sensor reading, read a fixed number of bytes and strip the CR/LF bytes.
                #   Make sure we have enough bytes to fill an entire measurement.
                if ( len(Buffer) < SerialMessageLength ):
                    return (Buffer, None, None, True)

                MeasurementBytes: bytearray = Buffer[:SerialMessageLength]

                #   The byte order, format, and endianness is all strictly
                #   defined by the firmware of the micro-controller. We simply
                #   must match the formatting here.
                Measurement = RawSensorReading()
                Measurement.I2C_Address     = MeasurementBytes[1]
                Measurement.LayerIndex      = MeasurementBytes[2]
                Measurement.DeviceIndex     = MeasurementBytes[3]

                #   Arduino is a little-endian architecture, and only the magnetic field values are signed quantities.
                Measurement.Field_X         = int.from_bytes(MeasurementBytes[4:6],   byteorder='little', signed=True)
                Measurement.Field_Y         = int.from_bytes(MeasurementBytes[6:8],   byteorder='little', signed=True)
                Measurement.Field_Z         = int.from_bytes(MeasurementBytes[8:10],  byteorder='little', signed=True)
                Measurement.Temperature     = int.from_bytes(MeasurementBytes[10:12], byteorder='little', signed=False)

                Measurement.Timestamp       = int.from_bytes(MeasurementBytes[12:16], byteorder='little', signed=False)

                #   Perform a very rudimentary check to assert the values of the
                #   record are within the expected bounds based on the known bit
                #   depths or ranges of these quantities.
                if ( not Measurement.Valid() ):
                    Measurement = None

                #   Fast-Forward the buffer to reflect that a reading has been
                #   successfully processed.
                Buffer = Buffer[SerialMessageLength:]

            else:
                #   It's a log message, read to the next newline.
                MessageEnd: int = Buffer.find(ord('\n'))
                if ( MessageEnd == -1 ):
                    #   If no newline is found, then we only have part of the
                    #   message, so we're finished reading.
                    return (Buffer, None, None, True)

                #   Extract out the message, decode it as UTF-8, and remove all
                #   trailing whitespace
                Message = Buffer[:MessageEnd].decode().rstrip()

                #   Fast-Forward the buffer, noting that we want to remove the
                #   newline we indexed as the end of the message.
                Buffer = Buffer[MessageEnd+1:]

            return (Buffer, Measurement, Message, False)

        except:
            #   If an exception occurred, catch it and ignore it.
            if ( len(Buffer) == 0 ):
                #   If the buffer is empty, just skip ahead and try a new read()...
                return (Buffer, None, None, True)
            else:
                #   Otherwise, assume we got a garbage byte and skip ahead one byte.
                return (Buffer[1:], None, None, False)


    def _ReadFile(self: MeasurementStream) -> None:
        """
        _ReadFile

        This function is the main function for reading RawSensorReading() values
        from a CSV file. This parses each line into a RawSensorReading instance,
        and pushes them to the internal FIFO queue.

        Return (None):
            None, the file is read and the internal FIFO queue populated until
            the file contents are exhausted.
        """

        try:

            FirstLine: bool = True
            with open(self.Filename, "r") as Measurements:
                while ( self._MonitorActive ):

                    #   Read a line from the file...
                    Line: str = Measurements.readline()

                    #   If this is the first line of the file, check if it contains
                    #   the column headers or not...
                    if ( FirstLine == True ):
                        if ( Line.split(",")[0].upper().startswith("I2C") ):
                            #   If it does contain the headers, skip ahead one line
                            #   and unset the flag to check for a header.
                            Line = Measurements.readline()
                        FirstLine = False

                    #   If we reach an empty line, that's the end of the file.
                    #   Indicate that this thread is halting and can be join()ed at
                    #   any time.
                    if ( len(Line) == 0 ):
                        self.LogWriter.Println(f"_ReadFile() ending - Empty line encountered.")
                        self._MonitorActive = False
                        return

                    #   Parse the fields from the line, building the
                    #   RawSensorReading() instance from them.
                    Fields: typing.List[str] = Line.split(",")
                    RawMeasurement: RawSensorReading = RawSensorReading()
                    RawMeasurement.I2C_Address = int(Fields[0])
                    RawMeasurement.LayerIndex  = int(Fields[1])
                    RawMeasurement.DeviceIndex = int(Fields[2])
                    RawMeasurement.Field_X     = int(Fields[3])
                    RawMeasurement.Field_Y     = int(Fields[4])
                    RawMeasurement.Field_Z     = int(Fields[5])
                    RawMeasurement.Temperature = int(Fields[6])
                    RawMeasurement.Timestamp   = int(Fields[7])

                    self.Measurements.append(RawMeasurement)

            #   If the file closes somehow, also indicate this thread is ended.
            self.LogWriter.Println(f"_ReadFile() ending - File closed.")
            self._MonitorActive = False
            return

        except Exception as e:
            self.LogWriter.Errorln(f"Exception occured during _ReadFile() - [ {e} ]\n\n{''.join(traceback.format_exception(e, value=e, tb=e.__traceback__))}\n")
            self._MonitorActive = False
            return

Config: Configuration = Configuration(LogWriter=LogWriter)
Measurements: MeasurementStream = None

def SafeShutdown(signal: typing.Any, frame: typing.Any) -> None:
    """
    SafeShutdown

    This function will safely shut down any of the spawned resources, allowing
    the program to close smoothly, safely, and in a timely manner if either a
    fatal exception occurs or a keyboard-interrupt is requested.

    Return (None):
        None, the program will exit once this returns.
    """

    LogWriter.Warnln(f"Shutdown requested... Closing MeasurementStream and open Figures...")
    if ( Measurements is not None ):
        Measurements.Halt()

    plt.close()

    return

#   Attach the SafeShutdown function to the keyboard interrupt signal.
signal.signal(signal.SIGINT, SafeShutdown)

#   Main
#       This is the main entry point of the script.
def main() -> int:

    global Measurements

    #   Initialize a MeasurementStream to asynchronously read individual measurements from the
    #   sensor over the serial port.
    Measurements = MeasurementStream(LogWriter=LogWriter, SerialPort=Config.SerialPort, Filename=Config.MeasurementsFilename)
    Measurements.StartReading()

    #   Initialize the 3 dimensional array of formatted field values with zeroes, and initialize the
    #   vector plot display window to show the field.
    MagneticField: np.ndarray = np.zeros(shape=(6,64), dtype=np.float64)   #   I want to have two 3-d vectors containing (x,y,z) and (u,v,w)
    MagneticFieldNorms: np.ndarray = np.zeros(shape=(4,64), dtype=np.float64)   #   I want 3d a (x,y,z) and a scalar (B)
    TemperatureField: np.ndarray = np.zeros(shape=(4,64), dtype=np.float64)     #   I want a 3d (x,y,z) and a scalar (T)

    #   Create the figure object to work with.
    Fields_Figure: Figure = plt.figure()

    #   Create subplots for the magnetic and temperature fields, as 3D plots.
    MagneticField_Axes: Axes = Fields_Figure.add_subplot(1, 2, 1, projection='3d')
    TemperatureField_Axes: Axes = Fields_Figure.add_subplot(1, 2, 2, projection='3d')

    #   Label the axes of each of the subplots.
    MagneticField_Axes.set_xlabel('X Direction (mm)')
    MagneticField_Axes.set_ylabel('Y Direction (mm)')
    MagneticField_Axes.set_zlabel('Z Direction (mm)')
    TemperatureField_Axes.set_xlabel('X Direction (mm)')
    TemperatureField_Axes.set_ylabel('Y Direction (mm)')
    TemperatureField_Axes.set_zlabel('Z Direction (mm)')

    #   Prepare and initialize the normalization maps and colour maps for
    #   colouring and plotting the data values in each of the subplots
    MagneticFieldMax: int = 2047
    ColourMap = cm.plasma
    MagneticNorm = matplotlib.colors.Normalize(vmin=0, vmax=MagneticFieldMax * math.sqrt(3) * GaussToMilliTesla)
    TemperatureNorm = matplotlib.colors.Normalize(vmin=0, vmax=100)
    MagneticScalarMap = cm.ScalarMappable(norm=MagneticNorm, cmap=ColourMap)
    TemperatureScalarMap = cm.ScalarMappable(norm=TemperatureNorm, cmap=ColourMap)
    MagneticFieldColourBar = Fields_Figure.colorbar(MagneticScalarMap, ax=MagneticField_Axes, label='Magentic Field Strength (mT)', shrink=0.5, pad=.2, aspect=10)
    TemperatureFieldColourBar = Fields_Figure.colorbar(TemperatureScalarMap, ax=TemperatureField_Axes, label=r'Temperature ($\degree$C)', shrink=0.5, pad=.2, aspect=10)

    #   Set the default viewing position of the plots to something where overlap
    #   between layers is relatively minor
    FigureAzimuth: float = -110
    FigureElevation: float = 70
    MagneticField_Axes.azim = FigureAzimuth
    MagneticField_Axes.elev = FigureElevation
    TemperatureField_Axes.azim = FigureAzimuth
    TemperatureField_Axes.elev = FigureElevation

    Fields_Figure.tight_layout()

    #   Initialize the output writer to append each measurement to a text file for later review or
    #   replay of the time-varying field measurements.
    RawOutputStream: typing.TextIO = open(os.devnull, "w+")
    FormattedOutputStream: typing.TextIO = open(os.devnull, "w+")
    if ( not Config.EnableDryRun ):

        MeasurementsFilename: str = f"Sensor-Readings - {datetime.now().strftime('%Y-%m-%d %H-%M-%S')}.csv"
        if ( Config.MeasurementsFilename is not None ):
            MeasurementsFilename = os.path.basename(Config.MeasurementsFilename)

        LogWriter.Println(f"Writing formatted sensor readings to the file [ Formatted {MeasurementsFilename} ]...")
        FormattedOutputStream = open(os.path.join(Config.OutputDirectory, "Formatted " + MeasurementsFilename), "w+")
        FormattedOutputStream.write(FormattedSensorReading().CSVHeader())
        if ( Config.SerialPort is not None ):
            LogWriter.Println(f"Writing raw sensor telemetry to the file [ Raw {MeasurementsFilename} ]...")
            RawOutputStream = open(os.path.join(Config.OutputDirectory, "Raw " + MeasurementsFilename), "w+")
            RawOutputStream.write(RawSensorReading().CSVHeader())

    #   Enter the main loop where we re-fresh the display of the vector plot with the most up-to-date measurement values
    #   as read from the MeasurementStream.
    UniqueSensingElements: typing.Set[int] = set()
    Count: int = 0
    while ( Measurements.Active() ):

        #   Pop the oldest measurement off the queue...
        CurrentMeasurement = Measurements.Next()
        if ( CurrentMeasurement is None ):
            #   If there is none, redraw the figures with potentally new data and try again
            Count = 0
            plt.draw_all(force=True)
            plt.pause(0.1)
            continue

        #   Update the set of unique devices being read, to allow the update
        #   function to know this dynamically.
        UniqueSensingElements.add(CurrentMeasurement.I2C_Address)

        #   Compute the dimensional sensor readings, and write out the raw and
        #   formatted values to the respective output files.
        Formatted: FormattedSensorReading = CurrentMeasurement.Format()
        RawOutputStream.writelines([CurrentMeasurement.ToString(), "\r\n"])
        FormattedOutputStream.writelines([Formatted.ToString(), "\r\n"])

        #   Store the current timestamp value, for reference in the figures.
        CurrentTime: float = Formatted.Timestamp

        #   Push the new data into the pre-allocated plotted data arrays, so the
        #   next re-draw of the figures can include this next reading.
        MagneticField[:,Formatted.Index] = np.concatenate((Formatted.Position, Formatted.MagneticField))
        MagneticFieldNorms[:,Formatted.Index] = np.concatenate((Formatted.Position, np.array([np.linalg.norm(Formatted.MagneticField)])))
        TemperatureField[:,Formatted.Index] = np.concatenate((Formatted.Position, np.array([Formatted.Temperature])))

        Count += 1
        #   For reading from a file, update once we have received as many new
        #   readings as there are sensing elements.
        #   For reading from a serial port, update to keep the backlog of
        #   measurements consistent.
        if (( Config.StreamType.lower() == "file" ) and (( Count >= len(UniqueSensingElements) ) or ( len(Measurements) < len(UniqueSensingElements) ))) \
            or (( Config.StreamType.lower() == "serial" ) and ( Count >= len(Measurements) )):

            #   Clear only the data markers from the plots, without clearing the titles or axis labels.
            for artist in itertools.chain(MagneticField_Axes.lines, MagneticField_Axes.collections, TemperatureField_Axes.lines, TemperatureField_Axes.collections):
                artist.remove()

            #   Compute a mask for the non-zero field values, to skip attempting
            #   to plot the uninitialized elements of the data arrays.
            FieldNonZeroMask: np.ndarray = TemperatureField[3,:] != 0
            MagneticField_NonZeroMask = np.tile(FieldNonZeroMask, 6).reshape(MagneticField.shape)
            TemperatureField_NonZeroMask = np.tile(FieldNonZeroMask, 4).reshape(TemperatureField.shape)

            #   Apply the non-zero mask...
            MagneticField_NonZero = MagneticField[MagneticField_NonZeroMask]
            MagneticField_NonZero = MagneticField_NonZero.reshape((6, int(MagneticField_NonZero.shape[0] / 6)))
            MagneticField_Average = MagneticField_NonZero[3:6,:].copy()
            MagneticField_Average = np.linalg.norm(MagneticField_Average.mean(axis=1))
            MagneticFieldNorm_NonZero = MagneticFieldNorms[TemperatureField_NonZeroMask]
            MagneticFieldNorm_NonZero = MagneticFieldNorm_NonZero.reshape((4, int(MagneticFieldNorm_NonZero.shape[0] / 4)))
            TemperatureField_NonZero = TemperatureField[TemperatureField_NonZeroMask]
            TemperatureField_NonZero = TemperatureField_NonZero.reshape((4, int(TemperatureField_NonZero.shape[0] / 4)))

            #   Create a new array of colours to apply to the quiver-plot arrows
            #   for the magnetic field, to allow both colour-grading and length
            #   scaling in proportion to magnitude.
            C = (MagneticFieldNorm_NonZero[3,:]).copy()
            C = np.concatenate((C, np.repeat(C, 2)))
            C = ColourMap(MagneticNorm(C))

            #   Compute the re-scaling factor to make the Z-axis vectors appear visually consistent with
            #       the X and Y axes...
            Z_VisualScaleFactor: float = 1
            try:
                Z_VisualScaleFactor: float = float(np.diff(MagneticField_Axes.get_zlim())) / float(np.mean([np.diff(MagneticField_Axes.get_xlim()), np.diff(MagneticField_Axes.get_ylim())]))
            except:
                pass


            #   Update the actual data markers in the two plots with the most up-to-date data values.
            MagneticField_Axes.quiver(X=MagneticField[0,:], Y=MagneticField[1,:], Z=MagneticField[2,:], U=MagneticField[3,:]/MagneticFieldMax, V=MagneticField[4,:]/MagneticFieldMax, W=(MagneticField[5,:]/MagneticFieldMax)*Z_VisualScaleFactor, arrow_length_ratio=0.33, normalize=False, length=MagneticScalarMap.get_clim()[1] / 0.5, colors=C)
            TemperatureField_Axes.scatter(xs=TemperatureField_NonZero[0,:], ys=TemperatureField_NonZero[1,:], zs=TemperatureField_NonZero[2,:], data=TemperatureField_NonZero[3,:], depthshade=False, c=TemperatureField_NonZero[3,:], cmap=ColourMap, vmin=TemperatureScalarMap.get_clim()[0], vmax=TemperatureScalarMap.get_clim()[1])

            #   Update the titles of the plots to provide summary values to the user.
            MagneticFieldTitle: str = f'Magnetic Field\nAverage = {MagneticField_Average:.1f}mT\nMean Magnitude = {MagneticFieldNorm_NonZero[3,:].mean():.1f}mT\nMaximum = {MagneticFieldNorm_NonZero[3,:].max():.1f}mT\nMean Vector = {MagneticField_NonZero.mean(axis=1)[3:].round(2)}mT'
            TemperatureFieldTitle: str = f'Temperature Field\nMean = {TemperatureField_NonZero[3,:].mean():.2f}C\nMaximum = {TemperatureField_NonZero[3,:].max():.2f}C\nMinimum = {TemperatureField_NonZero[3,:].min():.2f}C'
            if ( Config.MeasurementRate is not None ):
                #   If the sampling rate is known, also report this.
                MagneticFieldTitle += f"\nSampling Rate = {Config.MeasurementRate:.3f}Hz"
                TemperatureFieldTitle += f"\nSampling Rate = {Config.MeasurementRate:.3f}Hz"

            MagneticField_Axes.set_title(MagneticFieldTitle)
            TemperatureField_Axes.set_title(TemperatureFieldTitle)
            Fields_Figure.suptitle(f"Measurement Backlog: {len(Measurements)}\nCurrent Time: {CurrentTime:.6f}s")

            plt.draw_all(force=True)
            plt.pause(0.01)

            Count = 0

    #   Once there are not more measurements to read, halt and clean up the
    #   stream, and close the output files.
    Measurements.Halt()
    RawOutputStream.close()
    FormattedOutputStream.close()
    LogWriter.Println(f"MeasurementStream closed. Press [ q ] to close figure and end the program...")
    plt.show(block=True)

    return 0

def HandleArguments() -> bool:
    """
    HandleArguments

    This function handles setting up, parsing, validating, and applying the
    command-line arguments to the configuration state of the program.

    Return (bool):
        A boolean value indicating whether the program should continue past
        argument handling.
    """

    #   Initialize the argument parser to handle the command-line arguments.
    Parser: argparse.ArgumentParser = argparse.ArgumentParser(description="This script performs the field visualization for the hall-effect sensor array. This script can either read new measurements from the sensor over the serial port (--stream-type=serial), or replay previous meausrements saved to a file (--stream-type=file).", add_help=True)

    #   Add in all of the command-line flags this program will accept.
    Parser.add_argument("--list-serial-ports", dest="ListSerialPorts", action="store_true", required=False, default=False, help="List the possible available devices to use as serial port to read from.")
    Parser.add_argument("--stream-type", dest="StreamType", metavar="file|serial", type=str, required=False, default="", help="Which type of measurement stream is being read from? Either the Serial Port, or a File")
    Parser.add_argument("--serial-port", dest="SerialPort", metavar="device", type=str, required=False, default=None, help="The device name of the Serial Port to read measurements from. See --list-serial-ports to learn what devices are available. Only required with --stream-type=serial")
    Parser.add_argument("--baud-rate", dest="BaudRate", metavar="baud-rate", type=int, required=False, default=9600, help="The baud rate to use reading/writing the Serial Port. Only required for --stream-type=serial. Typically either 9600 for DEBUG mode, or 115200 for non-DEBUG mode.")
    Parser.add_argument("--filename", dest="Filename", metavar="file-path", type=str, required=False, default=None, help="The path to the formatted CSV file containing raw meausrements to replay. Only required for --stream-type=serial")
    Parser.add_argument("--position-reference", dest="PositionReference", metavar="corner-label", type=str, required=False, default="D", help="Which corner of Layer 0 was used as the indexing point to position the sensor relative to the magnetic source?")
    Parser.add_argument("--layer-separation", dest="LayerSeparation", metavar="mm", type=float, required=False, default=8.6, help="The consistent spacing between the layers of the sensor. If all layers are free-floating set to -1 to indicate this.")
    Parser.add_argument("--output-directory", dest="OutputDirectory", metavar="path", type=str, required=False, default=".", help="The output directory to write any output files or artefacts into.")

    #   Add in flags for manipulating the logging functionality of the script.
    Parser.add_argument("--log-file", dest="LogFile", metavar="file-path", type=str, required=False, default="-", help="File path to the file to write all log messages of this program to.")
    Parser.add_argument("--quiet",    dest="Quiet",   action="store_true",  required=False, default=False, help="Enable quiet mode, disabling logging of eveything except fatal errors.")

    #   Finally, add in scripts for modifying the basic environment or end-state of the script.
    Parser.add_argument("--dry-run",  dest="DryRun",   action="store_true", required=False, default=False, help="Enable dry-run mode, where no file-system alterations or heavy computations are performed.")
    Parser.add_argument("--validate", dest="Validate", action="store_true", required=False, default=False, help="Only validate the command-line arguments, do not proceed with the remainder of the program.")

    #   Actually parse the command-line arguments
    Arguments: argparse.Namespace = Parser.parse_args()

    Valid: bool = True

    #   Extract, set, and validate the command-line arguments provided to the program.
    if Arguments.ListSerialPorts:
        ListSerialPorts()
        return False

    Config.EnableDryRun = Arguments.DryRun
    if ( Arguments.Quiet ):
        LogWriter.SetOutputStream(os.devnull)
    elif ( Arguments.LogFile != "-" ) and ( not Config.EnableDryRun ):
        LogWriter.SetOutputFilename(Arguments.LogFile)

    #   Check which type of measurement stream should be read from.
    Config.StreamType = Arguments.StreamType.lower()
    if ( Config.StreamType == "file" ):
        LogWriter.Println(f"Attempting to replay previous measurements from file [ {Arguments.Filename} ]...")
        if ( Arguments.Filename is None ) or ( not os.path.exists(Arguments.Filename) ):
            LogWriter.Errorln(f"File [ {Arguments.Filename} ] is not provided or does does not exist!")
            Valid = False
        else:
            Config.MeasurementsFilename = Arguments.Filename
            Config.SerialPort = None
            Config.BaudRate = 0
    elif ( Config.StreamType == "serial" ):
        LogWriter.Println(f"Attempting to read new measurements from Serial Port [ {Arguments.SerialPort} ] at [ {Arguments.BaudRate} baud ]...")
        try:
            Config.SerialDevice = Arguments.SerialPort
            LogWriter.Println(f"Attempting to open Serial Port [ {Config.SerialDevice} ]...")

            Config.BaudRate = Arguments.BaudRate
            LogWriter.Println(f"Configuring Serial Port for communication at [ {Config.BaudRate} baud ]...")
            if ( Config.BaudRate not in serial.SerialBase.BAUDRATES ):
                for BaudRate in serial.SerialBase.BAUDRATES:
                    if ( BaudRate >= Config.BaudRate ):
                        LogWriter.Warnln(f"Requested baud rate of [ {Config.BaudRate} ] is not supported... Switching to supported speed of [ {BaudRate} ].")
                        Config.BaudRate = BaudRate
                        break

            Config.SerialPort = serial.Serial(Arguments.SerialPort, Arguments.BaudRate, timeout=50e-2, xonxoff=False)
        except Exception as e:
            LogWriter.Errorln(f"Failed to open Serial Port [ {Config.SerialDevice}]: [ {e} ]\n\n{''.join(traceback.format_exception(e, value=e, tb=e.__traceback__))}\n")
            Valid = False
            Config.SerialPort = None
            Config.BaudRate = 0
            Config.SerialDevice = ""
            LogWriter.Println(f"Listing the available serial ports able to be used with [ --stream-type=serial ]...")
            ListSerialPorts()
    else:
        LogWriter.Errorln(f"Unknown or invalid --stream-type specified. Must specify one of either [ --stream-type=serial ] or [ --stream-type=file ].")
        LogWriter.Println(f"No --stream-type was specified... Listing the available Serial Port devices to be used with [ --stream-type=serial ].")
        ListSerialPorts()
        Valid = False

    if ( Arguments.PositionReference.upper() not in "ABCD" ):
        LogWriter.Errorln(f"Invalid or unknown indexing corner: [ {Arguments.PositionReference.upper()} ]")
        Valid = False
    else:
        Config.PositionReference = Arguments.PositionReference.upper()
        LogWriter.Println(f"Working with indexing position defined by corner [ {Config.PositionReference} ] of Layer 0.")

    if ( Arguments.LayerSeparation < 0 ):
        LogWriter.Println(f"Layers of the sensor are free-floating. Defaulting to a separation of 1mm for visualization purposes only.")
        Config.SensorPitch_Z = 1.0
    else:
        LogWriter.Println(f"Layers of the sensor are separated by [ {Arguments.LayerSeparation:.2f}mm ].")
        Config.SensorPitch_Z = Arguments.LayerSeparation

    Config.OutputDirectory = os.path.abspath(Arguments.OutputDirectory)
    if ( not Config.EnableDryRun ):
        LogWriter.Println(f"Writing all output files to directory [ {Config.OutputDirectory} ]...")
        os.makedirs(Config.OutputDirectory, mode=0o755, exist_ok=True)
        LogWriter.Println(f"Created output directory [ {Config.OutputDirectory} ].")

    return Valid and ( not Arguments.Validate )

def ListSerialPorts() -> None:
    """
    ListSerialPorts

    This function reports a list of all of the available serial ports on the
    current machine, in a format suitable for using with the --serial-port
    command-line option

    Return (None):
        None, the resulting information is printed to the screen for the user to
        review.
    """

    Devices: typing.List[serial.tools.list_ports_common.ListPortInfo] = serial.tools.list_ports.comports()
    LogWriter.Println(f"Found the following {len(Devices)} candidate device(s) to read as Serial Ports:")
    for Index, Device in enumerate(Devices, start=1):
        LogWriter.Println(f"Device {Index}/{len(Devices)}: [ {Device.device} ]")

    return

def PositionLookup(LayerIndex: int, DeviceIndex: int, IndexingCorner: str) -> typing.Tuple[float, float, float]:
    """
    PositionLookup:

    This function performs the position look-up required to map a particular
    LayerIndex and DeviceIndex pair to a specific (x,y,z) location, using the
    provided indexing corner as the origin of the coordinate system.

    LayerIndex:
        The 0-indexed layer of the device to be mapped.
    DeviceIndex:
        The 0-indexed intra-layer index of the device to be mapped.
    IndexingCorner:
        The specific corner of the daughterboard to use as the origin of
        the coordinate system.

    Return (typing.Tuple[float, float, float]):
        [0] - float:
            X position (mm)
        [1] - float:
            Y position (mm)
        [2] - float:
            Z position (mm)
    """

    X, Y = 0.0, 0.0
    Z: float = (LayerIndex * Config.SensorPitch_Z) + Config.SensorOffset_Z

    if ( IndexingCorner.upper() == "A" ):
        X, Y = PositionLookup_A[DeviceIndex]
    elif ( IndexingCorner.upper() == "B" ):
        X, Y = PositionLookup_B[DeviceIndex]
    elif ( IndexingCorner.upper() == "C" ):
        X, Y = PositionLookup_C[DeviceIndex]
    elif ( IndexingCorner.upper() == "D" ):
        X, Y = PositionLookup_D[DeviceIndex]
    else:
        LogWriter.Errorln(f"Unknown indexing corner value: {IndexingCorner}")

    return (X, Y, Z)

def ComputeFormattedTemperature(RawValue: int) -> float:
    """
    ComputeFormattedTemperature

    This function performs the conversion for the temperature value reported by the ALS31313 from
    raw ADC bits to degrees Celcius. The sensitivity and offset values are taken directly from the data sheet
    of this device.

    RawValue:
        The raw 12-bit unsigned integer value reported by the ALS31313.

    Return (float):
        The corresponding dimensional temperature value, in units of Degrees Celcius.
    """
    return TemperatureSensitivity * float(RawValue - TemperatureOffset)

#   Allow this script to be called from the command-line and execute the main function.
#   If anything needs to happen before executing main, add it here before the call.
if __name__ == "__main__":
    try:
        if ( HandleArguments() ):
            main()
        else:
            LogWriter.Errorln(f"Failed to validate command-line arguments. Required arguments either missing or invalid.")
    except Exception as e:
        LogWriter.Errorln(f"Exception raised in main(): [ {e} ]\n\n{''.join(traceback.format_exception(e, value=e, tb=e.__traceback__))}\n")
        SafeShutdown(None, None)
