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
import typing

#   3rd Party Imports
import cv2
import numpy as np
from readlif.reader import LifFile, LifImage
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
    #   ...

    ##  Private Member Variables
    _Pixels: np.ndarray
    _LogWriter: Logger.Logger
    #   ...

    ### Magic Methods
    def __init__(self: ZStack, LogWriter: Logger.Logger = Logger.Discarder) -> None:
        """
        Constructor

        This function...

        Return (None):
            ...
        """

        self.Name = None

        self._Pixels = np.ndarray([])
        self._LogWriter = LogWriter

        return

    ### Static Class Methods
    @staticmethod
    def FromFile(Filename: str, SeriesName: str = "") -> ZStack:
        """
        FromFile

        This function...

        Filename:
            ...
        SeriesName:
            ...

        Return (ZStack):
            ...
        """

        if ( Filename.lower().endswith("lif") ):
            return ZStack.FromLIF(Filename, SeriesName)
        elif ( Filename.lower().endswith("tif") ) or ( Filename.lower().endswith("tiff") ):
            return ZStack.FromTIF(Filename)
        else:
            return None

    @staticmethod
    def FromLIF(Filename: str, SeriesName: str) -> ZStack:
        """
        FromLIF

        This function:
            ...

        Filename:
            ...
        SeriesName:
            ...

        Return (ZStack):
            ...
        """

        Stack: ZStack = ZStack()

        Success: bool = Stack.OpenLIFFile(Filename, SeriesName)
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

    ### Public Methods
    def Display(self: ZStack) -> None:
        """
        Display

        This function...

        Return (None):
            ...
        """

        if ( self._Pixels is None ):
            return 0

        CurrentLayer: int = 0
        Key: int = 0
        while ( Key not in [ord(x) for x in "qQ"] ):

            Key = Utils.DisplayImage(
                Description=f"Z-Stack {self.Name} - Layer {CurrentLayer}/{self._Pixels.shape[0]}",
                Image=Utils.GammaCorrection(self._Pixels[CurrentLayer,:]),
                HoldTime=0
            )

            if ( Key in [ord(x) for x in 'uU'] ):
                CurrentLayer += 1
                if ( CurrentLayer >= self._Pixels.shape[0] ):
                    CurrentLayer = self._Pixels.shape[0] - 1
            elif ( Key in [ord(x) for x in 'dD'] ):
                CurrentLayer -= 1
                if ( CurrentLayer < 0 ):
                    CurrentLayer = 0

        return

    def OpenFile(self: ZStack, Filename: str, SeriesName: str = "") -> bool:
        """
        OpenFile

        This function...

        Filename:
            ...
        SeriesName:
            ...

        Return (bool):
            ...
        """

        if ( Filename.lower().endswith("lif") ):
            return self.OpenLIFFile(Filename, SeriesName)
        elif ( Filename.lower().endswith("tif") ) or ( Filename.lower().endswith("tiff") ):
            return self.OpenTIFFile(Filename)
        else:
            return False

    def OpenLIFFile(self: ZStack, Filename: str, SeriesName: str) -> bool:
        """
        OpenLIFFile

        This function...

        Filename:
            ...
        SeriesName:
            ...

        Return (bool):
            ...
        """

        #   Assert the arguments are provided...
        if ( Filename is None ) or ( Filename == "" ):
            self._LogWriter.Errorln(f"Failed to open Z-Stack from LIF file, no filename provided.")
            return False

        #   Assert the arguments are provided...
        if ( SeriesName is None ) or ( SeriesName == "" ):
            self._LogWriter.Errorln(f"Failed to open Z-Stack from LIF file, no image series provided.")
            return False

        #   Try to parse the file as a LIF File, using the 3rd party "readlif" library
        try:
            #   Open and parse the file into a LifFile instance...
            LifStack: LifFile = LifFile(Filename)

            #   Search for the requested series name within the file, iterating
            #   over each of the sub-images within the file..
            for Index, Series in enumerate(LifStack.image_list):
                if ( SeriesName.lower() in str(Series['name']).lower() ):

                    #   Apply this name to the Z-Stack, for recordkeeping
                    self.SetName(str(Series['name']))

                    #   Extract out the individual layers of the z-stack from the series...
                    Images: LifImage = LifStack.get_image(Index)

                    #   Get the dimensions of the resulting series.
                    #   These correspond to the (x, y, z) size of the images, as well as the
                    #   possible time and colour-channel sequences.
                    X, Y, Z, T, C = Images.dims.x, Images.dims.y, Images.dims.z, Images.dims.t, Images.channels
                    if ( T > 1 ) or ( C > 1 ):
                        self._LogWriter.Warnln(f"Failed to open Z-Stack from LIF file, multiple channels or time-points are not currently supported...")
                        return False

                    #   Identify the bit depth of the stack...
                    BitDepth: int = Images.bit_depth[0]

                    #   Ensure the pixel array is created with the correct size and bit depth to support the image data...
                    BitDepth = int(math.ceil(BitDepth / 8.0) * 8)
                    self._Pixels = np.zeros(shape=(Z, X, Y), dtype=f"uint{BitDepth}")

                    for Index, Layer in enumerate(Images.get_iter_z()):
                        self._Pixels[Index,:] = Layer

                    return True

        except:
            return False

        return False

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

            self._Pixels = np.array(ImageStack)
            return True

        except:
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

    #   ...

    ### Private Methods
    #   ...
