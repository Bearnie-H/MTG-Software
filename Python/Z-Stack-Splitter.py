#!/usr/bin/env python3

#   Author: ...
#   Date:   ...

#   Script Purpose: ...
#                       ...

#   Import the necessary standard library modules
from __future__ import annotations
import os
import sys
import typing
#   ...

#   Import the necessary third-part modules
import numpy as np
#   ...

#   Import the desired locally written modules
#   ...
from MTG_Common import Utils
from MTG_Common import Logger
from MTG_Common import ZStack

#   Define the globals to set by the command-line arguments
#   ...
LogWriter: Logger.Logger = Logger.Logger(Prefix="Z-Stack-Splitter.py")

#   Main
#       This is the main entry point of the script.
def main() -> None:

    Folder: str = os.path.dirname(sys.argv[1])
    Filename: str = os.path.basename(sys.argv[1])

    LogWriter.Println(f"Working with input file [ {Filename} ]...")

    Stack: ZStack.ZStack = ZStack.ZStack.FromFile(sys.argv[1])
    TotalLayers: int = Stack.Pixels.shape[0]

    for Index, Layer in enumerate(Stack.Pixels, start=1):
        Layer = Utils.GammaCorrection(Layer.copy())
        # Utils.DisplayImage(f"Layer {Index}", Layer, 0.5, True)
        Utils.WriteImage(Layer, os.path.join(f"{Folder}", f"{os.path.splitext(Filename)[0]}", f"Layer {Index}.tif"))
        LogWriter.Println(f"Wrote layer [ {Index}/{TotalLayers} ].")

    LogWriter.Println(f"Finished working with input file [ {Filename} ]!")
    return

#   Allow this script to be called from the command-line and execute the main function.
#   If anything needs to happen before executing main, add it here before the call.
if __name__ == "__main__":
    main()
