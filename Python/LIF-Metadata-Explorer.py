#!/usr/bin/env python3

#   Author: Joseph Sadden
#   Date:   11th July, 2025

#   Script Purpose: ...
#                       ...

#   Import the necessary standard library modules
from __future__ import annotations
import typing

import argparse
import xml.etree.ElementTree as ET
#   ...

#   Import the necessary third-part modules
from readlif.reader import LifFile
#   ...

#   Import the desired locally written modules
from MTG_Common import Logger
#   ...

#   Define the globals to set by the command-line arguments
LogWriter: Logger.Logger = Logger.Logger(Prefix="LIF-Metadata-Explorer")
#   ...

#   Main
#       This is the main entry point of the script.
def main() -> None:

    Parser: argparse.ArgumentParser = argparse.ArgumentParser()
    Parser.add_argument("--file",     dest="InputFile", metavar="file-path", type=str, required=True,                 help="The file path to the *.LIF file to split into individual *.TIFF stacks.")
    Arguments: argparse.Namespace = Parser.parse_args()

    InputFile: str = Arguments.InputFile

    LIF: LifFile = LifFile(InputFile)

    XMLData: ET = ET.ElementTree(ET.fromstring(LIF.xml_header))

    for Element in XMLData.iter():
        print(f"Element: {Element.tag}")
        for Attribute in Element.attrib:
            print(f"\tAttribute: {Attribute} - Value: {Element.attrib[Attribute]}")


    #   ...

    return



#   Allow this script to be called from the command-line and execute the main function.
#   If anything needs to happen before executing main, add it here before the call.
if __name__ == "__main__":
    main()
