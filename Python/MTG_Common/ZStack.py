#!/usr/bin/env python3

##      Author:     Joseph Sadden
##      Date:       21st February, 2024

# MIT License
#
# Copyright (c) 2024 Joseph Sadden
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

#   Standard Library Imports
import math
import os
import traceback
import typing

#   3rd Party Imports
import cv2
import numpy as np
import readlif
from readlif.reader import LifFile, LifImage
import czifile
#   ...

#   Custom Local Imports
from . import Logger
from . import Utils
#   ...

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
    _LogWriter: Logger.Logger
    #   ...

    ### Magic Methods
    def __init__(self: ZStack, LogWriter: Logger.Logger = Logger.Discarder, Name: str = None) -> None:
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

            Key = Utils.DisplayImage(
                Description=f"Z-Stack {self.Name} - Layer {CurrentLayer+1}/{self.Pixels.shape[0]}",
                Image=Utils.ConvertTo8Bit(self.Pixels[CurrentLayer,:]),
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

        #   Assert the arguments are provided...
        if (( SeriesName is None ) or ( SeriesName == "" )) and ( SeriesName < 0 ):
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
                self.SetName(Stack.name)
            elif ( SeriesName != "" ):
                for Index, Series in enumerate(LifStack.image_list):
                    if ( Series['name'].lower() == SeriesName.lower() ):
                        Stack = LifStack.get_image(Index)
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
            self.Pixels = np.zeros(shape=(Z, Y, X, T, C), dtype=f"uint{BitDepth}")

            for t in range(T):
                if ( ChannelIndex >= 0 ):
                    for z, Layer in enumerate(Stack.get_iter_z(t=t, c=ChannelIndex)):
                        self.Pixels[z,:,:,t,ChannelIndex] = Layer
                else:
                    for c in range(C):
                        for z, Layer in enumerate(Stack.get_iter_z(t=t, c=c)):
                            self.Pixels[z,:,:,t,c] = Layer

        except:
            return False

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
        return cv2.imwritemulti(os.path.join(Folder, f"{SeriesName}.tif"), [Utils.ConvertTo8Bit(x) for x in self.Pixels])

    #   ...

    ### Private Methods
    #   ...
