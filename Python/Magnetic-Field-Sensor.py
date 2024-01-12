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
import os
import serial
import signal
import sys
import threading
import time
import typing
#   ...

#   Import the necessary third-part modules
import numpy as np
import matplotlib
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
from mpl_toolkits.mplot3d import axes3d
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

SerialMessageLength: int = 17   #   bytes, including start and stop bytes.

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

    def FromBytes(self: RawSensorReading, RawBytes: bytearray) -> RawSensorReading:
        """
        FromBytes

        This function...

        Return (RawSensorReading):
            ...
        """

        if ( len(RawBytes) != SerialMessageLength ):
            LogWriter.Errorln(f"Message length over serial port is the wrong length! Expected [ {SerialMessageLength} ] - Got [ {len(RawBytes)} ]!")
            return None

        self.I2C_Address = RawBytes[1]
        self.LayerIndex = RawBytes[2]
        self.DeviceIndex = RawBytes[3]

        self.Field_X = RawBytes[5] * 256 + RawBytes[4]
        self.Field_Y = RawBytes[7] * 256 + RawBytes[6]
        self.Field_Z = RawBytes[9] * 256 + RawBytes[8]

        self.Temperature = RawBytes[11] * 256 + RawBytes[10]

        self.Timestamp = RawBytes[15] * 2**24 + RawBytes[14] * 2**16 + RawBytes[13] * 2**8 + RawBytes[12]

        return self

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

        Temperature: float = TemperatureSensitivity * float(self.Temperature - TemperatureOffset)

        X, Y, Z = PositionLookup(self.LayerIndex, self.DeviceIndex, Config.IndexingCorner)

        Index = self.LayerIndex * 2**3 + self.DeviceIndex

        return FormattedSensorReading(np.array([FieldX, FieldY, FieldZ]), np.array([X, Y, Z]), Temperature, self.Timestamp, Index)

    def ToString(self: RawSensorReading) -> str:
        """
        ToString

        This function

        Return (str):
            ...
        """

        return f"{self.I2C_Address},{self.LayerIndex},{self.DeviceIndex},{self.Field_X},{self.Field_Y},{self.Field_Z},{self.Temperature},{self.Timestamp}\n"

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

        self.SerialPort: serial.Serial = None
        self.MeasurementsFilename: str = None

        self.EnableDryRun: bool = False

        return

    pass

class MeasurementStream():
    """
    MeasurementStream

    This class...
    """

    LogWriter: Logger.Logger
    AdruinoLogWriter: Logger.Logger
    SerialPort: serial.Serial
    Filename: str

    Measurements: typing.List[RawSensorReading]

    _MonitorThread: threading.Thread
    _MonitorActive: bool

    def __init__(self: MeasurementStream, LogWriter: Logger.Logger = Logger.Discarder, SerialPort: serial.Serial = None, Filename: str = None) -> None:

        self.LogWriter = LogWriter
        self.AdruinoLogWriter = Logger.Logger(OutputStream=LogWriter.RawStream(), Prefix="Arduino")
        self.SerialPort: serial.Serial = SerialPort
        self.Filename = Filename

        self._MonitorThread: threading.Thread = None
        self._MonitorActive: bool = False

        self.Measurements: typing.List[RawSensorReading] = []

        #   ...

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
            self.LogWriter.Warnln(f"Cannot read next measurement from MeausrementStream, asynchronous thread is not active to read from the underlying stream!")
            return None

        return None

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

    def _ReadSerialPort(self: MeasurementStream) -> None:
        """
        _ReadSerialPort

        This function...

        Return (None):
            ...
        """

        while ( self._MonitorActive ) and ( self.SerialPort.is_open ):
            pass

        self._MonitorActive = False
        return

    def _ReadFile(self: MeasurementStream) -> None:
        """
        _ReadFile

        This function...

        Return (None):
            ...
        """

        with open(self.Filename, "r") as Measurements:
            while ( self._MonitorActive ):
                Line: str = Measurements.readline()
                if ( len(Line) == 0 ):
                    self._MonitorActive = False
                    return

                Fields: typing.List[str] = Line.split(",")
                RawMeasurement: RawSensorReading = RawSensorReading()
                RawMeasurement.I2C_Address: int = int(Fields[0])
                RawMeasurement.LayerIndex: int  = int(Fields[1])
                RawMeasurement.DeviceIndex: int = int(Fields[2])
                RawMeasurement.Field_X: int     = int(Fields[3])
                RawMeasurement.Field_Y: int     = int(Fields[4])
                RawMeasurement.Field_Z: int     = int(Fields[5])
                RawMeasurement.Temperature: int = int(Fields[6])
                RawMeasurement.Timestamp: int   = int(Fields[7])

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

    #   Initialize the output writer to append each measurement to a text file for later review or
    #   replay of the time-varying field measurements.
    OutputStream: typing.TextIO = open(os.devnull, "w+")
    if ( not Config.EnableDryRun ):
        OutputStream = open(f"Sensor-Readings - {datetime.now().strftime('%Y-%m-%d %H-%M-%S')}.csv", "w+")

    #   Enter the main loop where we re-fresh the display of the vector plot with the most up-to-date measurement values
    #   as read from the MeasurementStream.
    DrawInterval: int = 64
    Count: int = 0
    while (( CurrentMeasurement := Measurements.Next() ) is not None ):
        OutputStream.write(CurrentMeasurement.ToString())
        Formatted: FormattedSensorReading = CurrentMeasurement.Format()

        MagneticField[:,Formatted.Index] = np.concatenate((Formatted.Position, Formatted.MagneticField))
        MagneticFieldNorms[:,Formatted.Index] = np.concatenate((Formatted.Position, np.array([np.linalg.norm(Formatted.MagneticField)])))
        TemperatureField[:,Formatted.Index] = np.concatenate((Formatted.Position, np.array([Formatted.Temperature])))

        Count += 1
        if ( Count == DrawInterval ):
            MaxField: float = np.max(MagneticFieldNorms[3,:])
            LogWriter.Println(f"Measurement Backlog: {len(Measurements)}")

            MagneticField_Axes.cla()
            TemperatureField_Axes.cla()

            C = (MagneticFieldNorms[3,:]).copy()
            if ( C.ptp() == 0 ):
                C = np.zeros_like(C)
            else:
                C = (C.ravel() - C.min()) / C.ptp()
            C = np.concatenate((C, np.repeat(C, 2)))
            C = plt.cm.plasma(C)
            MagneticField_Axes.quiver(X=MagneticField[0,:], Y=MagneticField[1,:], Z=MagneticField[2,:], U=MagneticField[3,:]/MaxField, V=MagneticField[4,:]/MaxField, W=MagneticField[5,:]/MaxField, pivot='middle', length=4, arrow_length_ratio=0.5, colors=C)
            TemperatureField_Axes.scatter(xs=TemperatureField[0,:], ys=TemperatureField[1,:], zs=TemperatureField[2,:], data=TemperatureField[3,:], depthshade=False, c=TemperatureField[3,:], cmap='plasma')
            plt.draw_all()
            plt.pause(0.01)

            Count = 0

    while ( len(Measurements) > 0 ):
        LogWriter.Println(f"Waiting for measurement stream to end...")
        time.sleep(1)

    Measurements.Halt()
    OutputStream.close()

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
    Parser.add_argument("--stream-type", dest="StreamType", metavar="file|serial", type=str, required=True, default="serial", help="")
    Parser.add_argument("--serial-port", dest="SerialPort", metavar="port", type=str, required=False, default=None, help="")
    Parser.add_argument("--baud-rate", dest="BaudRate", metavar="baud", type=int, required=False, default=9600, help="")
    Parser.add_argument("--filename", dest="Filename", metavar="file-path", type=str, required=False, default=None, help="")
    Parser.add_argument("--position-reference", dest="PositionReference", metavar="corner-label", type=str, required=True, help="")
    #   ...

    #   Add in flags for manipulating the logging functionality of the script.
    Parser.add_argument("--log-file", dest="LogFile", metavar="file-path", type=str, required=False, default="-", help="File path to the file to write all log messages of this program to.")
    Parser.add_argument("--quiet",    dest="Quiet",   action="store_true",  required=False, default=False, help="Enable quiet mode, disabling logging of eveything except fatal errors.")

    #   Finally, add in scripts for modifying the basic environment or end-state of the script.
    Parser.add_argument("--dry-run",  dest="DryRun",   action="store_true", required=False, default=False, help="Enable dry-run mode, where no file-system alterations or heavy computations are performed.")
    Parser.add_argument("--validate", dest="Validate", action="store_true", required=False, default=False, help="Only validate the command-line arguments, do not proceed with the remainder of the program.")

    #   Actually parse the command-line arguments
    Arguments: argparse.Namespace = Parser.parse_args()

    #   Extract, set, and validate the command-line arguments provided to the program.
    Config.EnableDryRun = Arguments.DryRun
    Config.MeasurementsFilename = Arguments.Filename
    Config.IndexingCorner = Arguments.PositionReference
    #   ...

    return True or ( not Arguments.Validate )

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
    Z: float = LayerIndex * SensorPitch_Z + SensorOffset_Z

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

#   Allow this script to be called from the command-line and execute the main function.
#   If anything needs to happen before executing main, add it here before the call.
if __name__ == "__main__":
    if ( HandleArguments() ):
        main()
