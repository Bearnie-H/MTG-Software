#!/usr/bin/env python3

#   Author: Joseph Sadden
#   Date:   5th March, 2025

#   Script Purpose: This script splits a LIF file into a set of TIFF files,
#                       one for each series of the LIF file.

#   Import the necessary standard library modules
from __future__ import annotations
import typing

import argparse
import os
import sys

#   ...

#   Import the necessary third-part modules
from readlif.reader import LifFile, LifImage
#   ...

#   Import the desired locally written modules
from MTG_Common import Logger
from MTG_Common import ZStack
#   ...

#   Define the globals to set by the command-line arguments
#   ...

#   Main
#       This is the main entry point of the script.
def main() -> None:

    Parser: argparse.ArgumentParser = argparse.ArgumentParser()
    Parser.add_argument("--file",     dest="InputFile", metavar="file-path", type=str, required=True,                 help="The file path to the *.LIF file to split into individual *.TIFF stacks.")
    Parser.add_argument("--describe", dest="Describe",  action="store_true",           required=False, default=False, help="Only describe the series names within the *.LIF file and exit.")

    Parser.add_argument("--series-name",   dest="SeriesName",   metavar="SeriesName",   type=str, required=False, default="", help="The name of the specific series to extract.")
    Parser.add_argument("--series-index",  dest="SeriesIndex",  metavar="SeriesIndex",  type=int, required=False, default=-1, help="The index of the specific series to extract.")
    Parser.add_argument("--channel-index", dest="ChannelIndex", metavar="ChannelIndex", type=int, required=False, default=-1, help="The index of the channel within the series to extract.")

    Arguments: argparse.Namespace = Parser.parse_args()

    InputFile: str = Arguments.InputFile
    DescribeOnly: bool = Arguments.Describe
    SeriesName: str = Arguments.SeriesName
    SeriesIndex: str = Arguments.SeriesIndex
    ChannelIndex: str = Arguments.ChannelIndex

    if ( DescribeOnly ):
        #   Open and parse the file into a LifFile instance...
        LifStack: LifFile = LifFile(InputFile)

        #   Identify all of the series names within the file.
        SeriesNames: typing.List[str] = [x["name"] for x in LifStack.image_list]

        print(f"File [ {os.path.basename(InputFile)} ] contains the following series:")
        [print(f"Series Name: {x} - {LifStack.get_image(Index).dims} - Channels: {LifStack.get_image(Index).channels}") for (Index, x) in enumerate(SeriesNames)]

        return 0

    if (( SeriesName is None ) or ( SeriesName == "" )) and ( SeriesIndex == -1 ):
            #   Open and parse the file into a LifFile instance...
        LifStack: LifFile = LifFile(InputFile)

        #   Identify all of the series names within the file.
        SeriesNames: typing.List[str] = [x["name"] for x in LifStack.image_list]

        for SeriesName in SeriesNames:
            Stack: ZStack.ZStack = ZStack.ZStack(Logger.Logger(), Name=f"{os.path.basename(InputFile)} - {SeriesName=:},{SeriesIndex=:}")
            if ( not Stack.OpenLIFFile(InputFile, SeriesName=SeriesName) ):
                return -1

            if ( ChannelIndex < 0 ):
                for Channel in Stack.SplitChannels():
                    Channel.SaveTIFF(os.path.dirname(InputFile))
            else:
                Stack.SaveTIFF(os.path.dirname(InputFile))
    else:

        Stack: ZStack.ZStack = ZStack.ZStack(Logger.Logger(), Name=f"{os.path.basename(InputFile)} - {SeriesName=:},{SeriesIndex=:}")
        if ( not Stack.OpenLIFFile(InputFile, SeriesName=SeriesName, SeriesIndex=SeriesIndex, ChannelIndex=ChannelIndex) ):
            return -1

        if ( ChannelIndex < 0 ):
            for Channel in Stack.SplitChannels():
                Channel.SaveTIFF(os.path.dirname(InputFile))
        else:
            Stack.SaveTIFF(os.path.dirname(InputFile))

    return

#   Allow this script to be called from the command-line and execute the main function.
#   If anything needs to happen before executing main, add it here before the call.
if __name__ == "__main__":
    main()
