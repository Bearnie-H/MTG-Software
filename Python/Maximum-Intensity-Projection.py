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
LogWriter: Logger.Logger = Logger.Logger(Prefix="Maximum-Intensity-Projection.py")

#   Main
#       This is the main entry point of the script.
def main() -> None:

    Description: str = ""
    Projection: np.ndarray = None

    if ( len(sys.argv) == 3 ):
        if ( sys.argv[2].lower() == "--min" ):
            Projection = ZStack.ZStack.FromFile(sys.argv[1]).MinimumIntensityProjection()
            Description = "Minimum Intensity Projection"
        elif ( sys.argv[2].lower() == "--max" ):
            Projection = ZStack.ZStack.FromFile(sys.argv[1]).MaximumIntensityProjection()
            Description = "Maximum Intensity Projection"
        elif ( sys.argv[2].lower() == "--avg" ):
            Projection = ZStack.ZStack.FromFile(sys.argv[1]).AverageIntensityProjection()
            Description = "Average Intensity Projection"
        elif ( sys.argv[2].lower() == "--display" ):
            ZStack.ZStack.FromFile(sys.argv[1]).Display()


    if ( Projection is not None ):
        Utils.DisplayImage(
            Description,
            Utils.ConvertTo8Bit(Projection),
            5,
            True
        )

        Utils.WriteImage(Utils.ConvertTo8Bit(Projection), os.path.join(os.path.dirname(sys.argv[1]), f"{os.path.splitext(os.path.basename(sys.argv[1]))[0]} - {Description}.tif"))

    return

#   Allow this script to be called from the command-line and execute the main function.
#   If anything needs to happen before executing main, add it here before the call.
if __name__ == "__main__":
    main()
