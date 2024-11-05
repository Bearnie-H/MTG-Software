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

    Utils.DisplayImage(
        "Maximum Intensity Projection",
        ZStack.ZStack.FromFile(sys.argv[1]).MaximumIntensityProjection(),
        0,
        True
    )

    #   ...

    return

#   Allow this script to be called from the command-line and execute the main function.
#   If anything needs to happen before executing main, add it here before the call.
if __name__ == "__main__":
    main()
