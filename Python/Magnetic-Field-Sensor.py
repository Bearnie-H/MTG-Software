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
import traceback
import typing
#   ...

#   Import the necessary third-part modules
import numpy as np
import matplotlib
from matplotlib.axes import Axes
from matplotlib import cm
from matplotlib.colors import Colormap
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
#   ...

#   Import the desired locally written modules
from MTG import Logger
#   ...

#   Define the globals to set by the command-line arguments
LogWriter: Logger.Logger = Logger.Logger(Prefix=os.path.basename(sys.argv[0]))
#   ...

#   Define the global constants of the program
MagneticFieldSensitivity: int = 1   #   LSB/gauss
TemperatureOffset: int = 1708       #   LSB
TemperatureSensitivity: float = (302.0 / 4096.0) #  LSB/degC

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

SensorPitch_Z: float = 8.6      #   mm
SensorOffset_Z: float = 0.65    #   mm

SerialMessageLength: int = 16   #   bytes, including start and stop bytes.

class RawSensorReading():
    """
    RawSensorReading

    This class...
    """

    I2C_Address: int

    DeviceIndex: int
    LayerIndex: int

    Field_X: int
    Field_Y: int
    Field_Z: int
    Temperature: int

    Timestamp: int

    def Format(self: RawSensorReading) -> FormattedSensorReading:
        """
        Format

        This function...

        Return (FormattedSensorReading):
            ...
        """

        #   Convert the raw field values to dimensional values
        FieldX: float = self.Field_X / float(MagneticFieldSensitivity)
        FieldY: float = self.Field_Y / float(MagneticFieldSensitivity)
        FieldZ: float = self.Field_Z / float(MagneticFieldSensitivity)

        Temperature: float = ComputeFormattedTemperature(self.Temperature)

        X, Y, Z = PositionLookup(self.LayerIndex, self.DeviceIndex, Config.IndexingCorner)

        Index = self.LayerIndex * 2**3 + self.DeviceIndex

        return FormattedSensorReading(np.array([FieldX, FieldY, FieldZ]), np.array([X, Y, Z]), Temperature, self.Timestamp, Index)

    def Valid(self: RawSensorReading) -> bool:
        """
        Valid

        This function...

        Return (bool):
            ...
        """

        if not ( 0 <= self.LayerIndex < 8 ):
            return False

        if not ( 0 <= self.DeviceIndex < 8 ):
            return False

        if ( self.I2C_Address != (self.LayerIndex * 16 + self.DeviceIndex + 8 )):
            return False

        for Field in [self.Field_X, self.Field_Y, self.Field_Z]:
            if not ( -2048 <= Field < 2047 ):
                return False

        if not ( 0 <= self.Temperature < 4096 ):
            return False

        return True

    def CSVHeader(self: RawSensorReading) -> str:
        """
        CSVHeader

        This function...

        Return (str):
            ...
        """

        return f"I2C Address,Layer Index,Device Index,Field X, Field Y, Field Z,Temperature,Timestamp\r\n"

    def ToString(self: RawSensorReading) -> str:
        """
        ToString

        This function

        Return (str):
            ...
        """

        return f"{self.I2C_Address:3d},{self.LayerIndex},{self.DeviceIndex},{self.Field_X:+4d},{self.Field_Y:+4d},{self.Field_Z:+4d},{self.Temperature:4d},{self.Timestamp}"

class FormattedSensorReading():
    """
    FormattedSensorReading

    This class...
    """

    #   3D position, relative to one out of a set of user-selectable reference points
    Position: np.ndarray

    #   3D vector, in units of milliTesla
    MagneticField: np.ndarray

    #   Temperature, in units of degrees Celcius
    Temperature: float

    #   ...
    Timestamp: int

    Index: int

    def __init__(self: FormattedSensorReading, MagneticField: np.ndarray = np.array([0, 0, 0]), Position: np.ndarray = np.array([0, 0, 0]), Temperature: float = 0.0, Timestamp: int = 0, Index: int = 0) -> None:
        """
        Constructor

        This function...

        MagneticField:
            ...
        Position:
            ...
        Temperature:
            ...
        Timestamp:
            ...

        Return (None):
            ...
        """

        self.MagneticField = MagneticField
        self.Position = Position
        self.Temperature = Temperature
        self.Timestamp = Timestamp
        self.Index = Index

        return

    def __str__(self: FormattedSensorReading) -> str:
        """
        __str__

        This function...

        Return (str):
            ...
        """

        return f"X={self.MagneticField[0]}Ga, Y={self.MagneticField[1]}G, Z={self.MagneticField[2]}Ga, T={self.Temperature}C"

    def CSVHeader(self: FormattedSensorReading) -> str:
        """
        CSVHeader

        This function...

        Return (str):
            ...
        """

        return f"X (mm),Y (mm),Z (mm),U (mT),V (mT), W (mT),T (deg C),Timestamp (s)\r\n"

    def ToString(self: FormattedSensorReading) -> str:
        """
        ToString

        This function...

        Return (str):
            ...
        """

        return f"{self.Position[0]:+.4f},{self.Position[1]:+.4f},{self.Position[2]:+.4f},{self.MagneticField[0] / 10.0:+4.1f},{self.MagneticField[1] / 10.0:+4.1f},{self.MagneticField[2] / 10.0:+4.1f},{self.Temperature:.2f},{self.Timestamp * 1e-6:.6f}"

    pass

class Configuration():
    """
    Configuration

    This class...
    """

    def __init__(self: Configuration, LogWriter: Logger.Logger = Logger.Discarder) -> None:
        """
        Constructor

        This function...

        LogWriter:
            ...

        Return (None):
            ...
        """

        self._LogWriter: Logger.Logger = LogWriter

        self.StreamType: str = None

        self.SerialPort: serial.Serial = None
        self.SerialDevice: str = None
        self.BaudRate: int = 0

        self.MeasurementsFilename: str = None

        self.PositionReference: str = None

        self.EnableDryRun: bool = False

        return

class MeasurementStream():
    """
    MeasurementStream

    This class...
    """

    LogWriter: Logger.Logger
    ArduinoLogWriter: Logger.Logger
    SerialPort: serial.Serial
    Filename: str

    Measurements: typing.List[RawSensorReading]

    _MonitorThread: threading.Thread
    _MonitorActive: bool

    def __init__(self: MeasurementStream, LogWriter: Logger.Logger = Logger.Discarder, SerialPort: serial.Serial = None, Filename: str = None) -> None:

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

        This function...

        Return (int):
            ....
        """

        return len(self.Measurements)

    def StartReading(self: MeasurementStream) -> None:
        """
        StartReading

        This function...

        Return (None):
            ...
        """

        if ( self.Filename is not None ):
            self._MonitorActive = True
            self._MonitorThread: threading.Thread = threading.Thread(target=self._ReadFile)
            self._MonitorThread.start()
        elif ( self.SerialPort is not None ):
            self._MonitorActive = True
            self._MonitorThread: threading.Thread = threading.Thread(target=self._ReadSerialPort)
            self._MonitorThread.start()
        else:
            self.LogWriter.Errorln(f"Failed to start MeasurementStream, neither Serial Port nor File streams were initialized!")
            #   ...
            pass

        return

    def Next(self: MeasurementStream) -> RawSensorReading:
        """
        Next

        This function...

        Return (RawSensorReading):
            ...
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

        This function...

        Return (bool):
            ...
        """

        return self._MonitorActive or len(self.Measurements) > 0

    def Halt(self: MeasurementStream) -> None:
        """
        Halt

        This function...

        Return (None):
            ...
        """

        self._MonitorActive = False
        if ( self._MonitorThread is not None ):
            self._MonitorThread.join()
            self.Measurements = []

        return

    def ShouldDisplay(self: MeasurementStream, Count: int) -> bool:
        """
        ShouldDisplay

        This function...

        Count:
            ...

        Return (bool):
            ...
        """

        #   If we're reading from the serial port, acutally monitor the backlog
        #   versus the count of samples since the last refresh...
        if ( self.Filename is None ) or ( self.Filename == "" ):
            #   Allow the backlog to grow to no more than 1 times the last refresh value.
            return self.__len__() < ( Count )
        else:
            #   If we're reading from a file, just update every n measurements
            return (Count >= 8)

    def _ReadSerialPort(self: MeasurementStream) -> None:
        """
        _ReadSerialPort

        This function...

        Return (None):
            ...
        """

        if ( not self.SerialPort.is_open ):
            self.SerialPort.open()

        Buffer: bytearray = bytearray()

        while ( self._MonitorActive ) and ( self.SerialPort.is_open ):
            RawBytes: bytes = self.SerialPort.read(min(Config.BaudRate, 1024))
            Buffer.extend(RawBytes[:])
            if ( len(Buffer) == 0 ):
                continue

            Done: bool = False
            while ( not Done ):
                Buffer, Measurement, Message, Done = self._ParseSerialBytes(Buffer)
                if ( Measurement is not None ):
                    # self.LogWriter.Println(f"Received measurement: {Measurement.ToString()}")
                    self.Measurements.append(Measurement)
                elif ( Message is not None ) and ( len(Message) > 0 ):
                    self.ArduinoLogWriter.Println(f'Log Message: {Message}')

        self.LogWriter.Println(f"_ReadSerialPort() exiting...")
        self._MonitorActive = False
        self.SerialPort.close()

        return

    def _ParseSerialBytes(self: MeasurementStream, Buffer: bytearray) -> typing.Tuple[bytearray, RawSensorReading, str, bool]:
        """
        _ParseSerialBytes

        This function...

        Buffer:
            ...

        Return (Tuple):
            [0] (bytearray):
                ...
            [1] (RawSensorReading):
                ...
            [2] (str):
                ...
            [3] (bool):
                ...
        """

        try:

            Measurement: RawSensorReading = None
            Message: str = None

            if ( len(Buffer) == 0 ):
                return (Buffer, None, None, True)

            if ( Buffer[0] == 0x00 ):
                #   It's a sensor reading, read a fixed number of bytes and strip the CR/LF bytes.
                #   Make sure we have enough bytes to fill an entire measurement.
                if ( len(Buffer) < SerialMessageLength ):
                    return (Buffer, None, None, True)

                MeasurementBytes: bytearray = Buffer[:SerialMessageLength]

                Measurement = RawSensorReading()
                Measurement.I2C_Address     = MeasurementBytes[1]
                Measurement.LayerIndex      = MeasurementBytes[2]
                Measurement.DeviceIndex     = MeasurementBytes[3]

                Measurement.Field_X         = int.from_bytes(MeasurementBytes[4:6],   byteorder='little', signed=True)
                Measurement.Field_Y         = int.from_bytes(MeasurementBytes[6:8],   byteorder='little', signed=True)
                Measurement.Field_Z         = int.from_bytes(MeasurementBytes[8:10],  byteorder='little', signed=True)
                Measurement.Temperature     = int.from_bytes(MeasurementBytes[10:12], byteorder='little', signed=False)

                Measurement.Timestamp       = int.from_bytes(MeasurementBytes[12:16], byteorder='little', signed=False)

                if ( not Measurement.Valid() ):
                    Measurement = None

                Buffer = Buffer[SerialMessageLength:]

            else:
                #   It's a log message, read to the next newline.
                MessageEnd: int = Buffer.find(ord('\n'))
                if ( MessageEnd == -1 ):
                    #   If no newline is found, then we only have part of the message, so we're finished reading.
                    return (Buffer, None, None, True)

                Message = Buffer[:MessageEnd].decode().rstrip()
                Buffer = Buffer[MessageEnd+1:]

            return (Buffer, Measurement, Message, False)

        except:
            if ( len(Buffer) == 0 ):
                return (Buffer, None, None, True)
            else:
                return (Buffer[1:], None, None, False)


    def _ReadFile(self: MeasurementStream) -> None:
        """
        _ReadFile

        This function...

        Return (None):
            ...
        """

        FirstLine: bool = True

        with open(self.Filename, "r") as Measurements:
            while ( self._MonitorActive ):
                Line: str = Measurements.readline()
                if ( FirstLine == True ):
                    if ( Line.split(",")[0].lower().startswith("i2c") ):
                        Line = Measurements.readline()
                    FirstLine = False

                if ( len(Line) == 0 ):
                    self._MonitorActive = False
                    return

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

        self._MonitorActive = False
        return

Config: Configuration = Configuration(LogWriter=LogWriter)
Measurements: MeasurementStream = None

def SafeShutdown(signal: typing.Any, frame: typing.Any) -> None:
    """
    SafeShutdown

    This function...

    Return (None):
        ...
    """

    if ( Measurements is not None ):
        Measurements.Halt()

    plt.close()

    return

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

    Fields_Figure: Figure = plt.figure()

    MagneticField_Axes: Axes = Fields_Figure.add_subplot(1, 2, 1, projection='3d')
    TemperatureField_Axes: Axes = Fields_Figure.add_subplot(1, 2, 2, projection='3d')

    MagneticField_Axes.set_xlabel('X Direction (mm)')
    MagneticField_Axes.set_zlabel('Z Direction (mm)')
    MagneticField_Axes.set_ylabel('Y Direction (mm)')

    TemperatureField_Axes.set_xlabel('X Direction (mm)')
    TemperatureField_Axes.set_zlabel('Z Direction (mm)')
    TemperatureField_Axes.set_ylabel('Y Direction (mm)')

    MagneticFieldMax: int = 2047
    ColourMap = cm.plasma
    MagneticNorm = matplotlib.colors.Normalize(vmin=0, vmax=MagneticFieldMax * math.sqrt(3) / 10.0)
    TemperatureNorm = matplotlib.colors.Normalize(vmin=ComputeFormattedTemperature(0), vmax=ComputeFormattedTemperature(4095))
    MagneticScalarMap = cm.ScalarMappable(norm=MagneticNorm, cmap=ColourMap)
    TemperatureScalarMap = cm.ScalarMappable(norm=TemperatureNorm, cmap=ColourMap)
    MagneticFieldColourBar = Fields_Figure.colorbar(MagneticScalarMap, ax=MagneticField_Axes, label='Magentic Field Strength (mT)', shrink=0.5, pad=.2, aspect=10)
    TemperatureFieldColourBar = Fields_Figure.colorbar(TemperatureScalarMap, ax=TemperatureField_Axes, label=r'Temperature $\degree$C', shrink=0.5, pad=.2, aspect=10)

    Fields_Figure.subplots_adjust(bottom=0.05, top=0.95)
    Fields_Figure.tight_layout()

    #   Initialize the output writer to append each measurement to a text file for later review or
    #   replay of the time-varying field measurements.
    RawOutputStream: typing.TextIO = open(os.devnull, "w+")
    FormattedOutputStream: typing.TextIO = open(os.devnull, "w+")
    if ( not Config.EnableDryRun ) and ( Config.SerialPort is not None ):
        MeasurementsFilename: str = f"Sensor-Readings - {datetime.now().strftime('%Y-%m-%d %H-%M-%S')}.csv"
        RawOutputStream = open("Raw " + MeasurementsFilename, "w+")
        FormattedOutputStream = open("Formatted " + MeasurementsFilename, "w+")

        RawOutputStream.write(RawSensorReading().CSVHeader())
        FormattedOutputStream.write(FormattedSensorReading().CSVHeader())

    #   Enter the main loop where we re-fresh the display of the vector plot with the most up-to-date measurement values
    #   as read from the MeasurementStream.
    Count: int = 0
    while ( Measurements.Active() ):
        CurrentMeasurement = Measurements.Next()
        if ( CurrentMeasurement is None ):
            plt.draw_all(force=True)
            plt.pause(0.1)
            continue

        Formatted: FormattedSensorReading = CurrentMeasurement.Format()
        RawOutputStream.writelines([CurrentMeasurement.ToString(), "\r\n"])
        FormattedOutputStream.writelines([Formatted.ToString(), "\r\n"])

        MagneticField[:,Formatted.Index] = np.concatenate((Formatted.Position, Formatted.MagneticField / 10.0))
        MagneticFieldNorms[:,Formatted.Index] = np.concatenate((Formatted.Position, np.array([np.linalg.norm(Formatted.MagneticField / 10.0)])))
        TemperatureField[:,Formatted.Index] = np.concatenate((Formatted.Position, np.array([Formatted.Temperature])))

        Count += 1
        if ( Measurements.ShouldDisplay(Count) ):
            # LogWriter.Println(f"Measurement Backlog: {len(Measurements)}")

            for artist in itertools.chain(MagneticField_Axes.lines, MagneticField_Axes.collections, TemperatureField_Axes.lines, TemperatureField_Axes.collections):
                artist.remove()

            FieldNonZeroMask: np.ndarray = TemperatureField[3,:] != 0
            MagneticField_NonZeroMask = np.tile(FieldNonZeroMask, 6).reshape(MagneticField.shape)
            TemperatureField_NonZeroMask = np.tile(FieldNonZeroMask, 4).reshape(TemperatureField.shape)

            MagneticField_NonZero = MagneticField[MagneticField_NonZeroMask]
            MagneticField_NonZero = MagneticField_NonZero.reshape((6, int(MagneticField_NonZero.shape[0] / 6)))

            MagneticFieldNorm_NonZero = MagneticFieldNorms[TemperatureField_NonZeroMask]
            MagneticFieldNorm_NonZero = MagneticFieldNorm_NonZero.reshape((4, int(MagneticFieldNorm_NonZero.shape[0] / 4)))

            TemperatureField_NonZero = TemperatureField[TemperatureField_NonZeroMask]
            TemperatureField_NonZero = TemperatureField_NonZero.reshape((4, int(TemperatureField_NonZero.shape[0] / 4)))

            C = (MagneticFieldNorm_NonZero[3,:]).copy()
            C = np.concatenate((C, np.repeat(C, 2)))
            C = ColourMap(MagneticNorm(C))

            MagneticField_Axes.quiver(X=MagneticField[0,:], Y=MagneticField[1,:], Z=MagneticField[2,:], U=MagneticField[3,:]/MagneticFieldMax, V=MagneticField[4,:]/MagneticFieldMax, W=MagneticField[5,:]/MagneticFieldMax, colors=C, arrow_length_ratio=0.33, normalize=False, length=MagneticScalarMap.get_clim()[1] / 5.0)
            TemperatureField_Axes.scatter(xs=TemperatureField_NonZero[0,:], ys=TemperatureField_NonZero[1,:], zs=TemperatureField_NonZero[2,:], data=TemperatureField_NonZero[3,:], depthshade=False, c=TemperatureField_NonZero[3,:], cmap=ColourMap, vmin=TemperatureScalarMap.get_clim()[0], vmax=TemperatureScalarMap.get_clim()[1])

            MagneticField_Axes.set_title(f'Magnetic Field\nMean = {MagneticFieldNorm_NonZero[3,:].mean():.1f}mT\nMaximum = {MagneticFieldNorm_NonZero[3,:].max():.1f}mT')
            TemperatureField_Axes.set_title(f'Temperature Field\nMean = {TemperatureField_NonZero[3,:].mean():.2f}C\nMaximum = {TemperatureField_NonZero[3,:].max():.2f}C\nMinimum = {TemperatureField_NonZero[3,:].min():.2f}C')

            plt.draw_all(force=True)
            plt.pause(0.01)

            Count = 0

    while ( len(Measurements) > 0 ):
        LogWriter.Println(f"Waiting for measurement stream to end...")
        plt.pause(1)

    Measurements.Halt()
    RawOutputStream.close()
    FormattedOutputStream.close()

    return 0

def HandleArguments() -> bool:
    """
    HandleArguments

    This function...

    Return (bool):
        ...
    """

    #   Initialize the argument parser to handle the command-line arguments.
    Parser: argparse.ArgumentParser = argparse.ArgumentParser(description="", add_help=True)

    #   Add in all of the command-line flags this program will accept.
    Parser.add_argument("--list-serial-ports", dest="ListSerialPorts", action="store_true", required=False, default=False, help="List the possible available devices to use as serial port to read from.")
    Parser.add_argument("--stream-type", dest="StreamType", metavar="file|serial", type=str, required=False, default="serial", help="")
    Parser.add_argument("--serial-port", dest="SerialPort", metavar="port", type=str, required=False, default=None, help="")
    Parser.add_argument("--baud-rate", dest="BaudRate", metavar="baud", type=int, required=False, default=9600, help="")
    Parser.add_argument("--filename", dest="Filename", metavar="file-path", type=str, required=False, default=None, help="")
    Parser.add_argument("--position-reference", dest="PositionReference", metavar="corner-label", type=str, required=False, default="A", help="")

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
        #   ...
        Config.MeasurementsFilename = Arguments.Filename
        Config.SerialPort = None
        Config.BaudRate = 0
    elif ( Config.StreamType == "serial" ):
        #   ...
        Config.BaudRate = Arguments.BaudRate
        Config.SerialDevice = Arguments.SerialPort
        Config.SerialPort = serial.Serial(Config.SerialDevice, Config.BaudRate, timeout=0.1, xonxoff=False)
    else:
        #   ...
        Valid = False
        pass
    #   ...


    Config.IndexingCorner = Arguments.PositionReference
    #   ...

    return Valid or ( not Arguments.Validate )

def ListSerialPorts() -> None:
    """
    ListSerialPorts

    This function...

    Return (None):
        ...
    """

    Devices: typing.List[serial.tools.list_ports_common.ListPortInfo] = serial.tools.list_ports.comports()
    LogWriter.Println(f"Found the following {len(Devices)} candidate devices to read as Serial Ports:")
    for Index, Device in enumerate(Devices, start=1):
        LogWriter.Println(f"Device {Index}/{len(Devices)}: {Device}")

    return

def PositionLookup(LayerIndex: int, DeviceIndex: int, IndexingCorner: str) -> typing.Tuple[float, float, float]:
    """
    PositionLookup:

    This function...

    LayerIndex:
        ...
    DeviceIndex:
        ...
    IndexingCorner:
        ...

    Return (typing.Tuple[float, float, float]):
        [0] - float:
            ...
        [1] - float:
            ...
        [2] - float:
            ...
    """

    X, Y = 0.0, 0.0
    Z: float = (LayerIndex * SensorPitch_Z) + SensorOffset_Z

    if ( IndexingCorner.upper() == "A" ):
        X, Y = PositionLookup_A[DeviceIndex]
    elif ( IndexingCorner.upper() == "B" ):
        X, Y = PositionLookup_B[DeviceIndex]
    elif ( IndexingCorner.upper() == "C" ):
        X, Y = PositionLookup_C[DeviceIndex]
    elif ( IndexingCorner.upper() == "D" ):
        X, Y = PositionLookup_D[DeviceIndex]
    else:
        LogWriter.Errorln(f"...")

    return (X, Y, Z)

def ComputeFormattedTemperature(RawValue: int) -> float:
    """
    ComputeFormattedTemperature

    This function...

    RawValue:
        ...

    Return (float):
        ...
    """
    return TemperatureSensitivity * float(RawValue - TemperatureOffset)

#   Allow this script to be called from the command-line and execute the main function.
#   If anything needs to happen before executing main, add it here before the call.
if __name__ == "__main__":
    try:
        if ( HandleArguments() ):
            main()
    except Exception as e:
        LogWriter.Errorln(f"Exception raised in main(): [ {e} ]\n\n{''.join(traceback.format_exception(e, value=e, tb=e.__traceback__))}\n")
        SafeShutdown(None, None)
