#!/usr/bin/env bash

#   Author: Joseph Sadden
#   Date:   16th May, 2025

#   Purpose:
#
#       This script performs checksumming of all LIF files in a given directory,
#       allowing for explicit validation of file integrity when transferring
#       between systems.

#   Terminal Colour Codes
FATAL="\e[7m\e[31m"
RED="\e[91m"
GREEN="\e[92m"
YELLOW="\e[93m"
AMBER="\e[33m"
BLUE="\e[96m"
WHITE="\e[97m"
CLEAR="\e[0m"

# Global Logging variables
Quiet=
OutputFile="-"  #    "-" indicates STDOUT
ColourLog=1  #  Flag for adding colours to the log, empty means no colouring
ColourFlag=
LogPrefix="$(basename "$0")"
LOG_INFO=0
LOG_NOTICE=1
LOG_ATTENTION=2
LOG_WARNING=3
LOG_ERROR=4
LOG_FATAL=5
LogLevels=( "INFO:  " "NOTE:  " "ATTN:  " "WARN:  " "ERROR: " "FATAL: ")
LogColours=("$WHITE"  "$BLUE"   "$YELLOW" "$AMBER"  "$RED"    "$FATAL")
TimeFormat="+%F %H:%M:%S"
if ! date "$TimeFormat" > /dev/null 2>&1; then
    TimeFormat=
fi

#   Catch errors, interrupts, and more to allow a safe shutdown
trap 'cleanup 1' 1 2 3 6 8 9 15

HelpFlag=

#   Command-line / Global variables
ChecksumFilename="LIF-File-Checksums.sha256"
UpdateFlag=
SourceDirectory="$(pwd)"

#   Function to display a help/usage menu to the user in a standardized format.
function helpMenu() {

    #   Get the name of the executable
    local scriptName=$(basename "$0")

    #   Print the current help menu/usage information to the user
    echo -e "
    $BLUE$scriptName   -   A bash tool to create and verify the checksums of *.LIF files to ensure they do not get truncated or otherwise broken between imaging and analysis.$WHITE

    $GREEN$scriptName$YELLOW -d <Directory> [-h] [-o Output-File] [-q] [-z]$WHITE

    "$YELLOW"Checksum Options:$WHITE
        $BLUE-d$WHITE  -    Source Directory. Which base directory should be searched for LIF files to check.
        $BLUE-u$WHITE  -    Update. Ensure that the checksum file is updated, if it exists, instead of comparing checksums.

    "$YELLOW"Output Options:$WHITE
        $BLUE-o$WHITE  -    Log File. Redirect STDOUT to the given file, creating it if it doesn't exist yet.
        $BLUE-q$WHITE  -    Quiet mode. Only print out fatal errors and suppress all other output.
        $BLUE-z$WHITE  -    Raw Mode. Disables colouring, useful when the ANSI escape codes would be problematic.

    "$YELLOW"Miscellaneous Options:$WHITE
        $BLUE-h$WHITE  -    Help Menu. Display this help menu and exit.

    "$GREEN"Note:$WHITE
        <...>
    "$CLEAR
}

function cleanup() {

    #   Implement whatever cleanup logic is needed for the specific script, followed by resetting the terminal and exiting.

    # if [ $1 -eq 0 ]; then
    #     log $LOG_INFO "Successfully executed and beginning cleanup..."
    # else
    #     log $LOG_ATTENTION "Unsuccessfully executed and beginning cleanup..."
    # fi

    stop $1
}

function stop() {
    exit $1
}

function SetLogPrefix() {
    LogPrefix="$1"
}

#   $1 -> Log Level
#   $2 -> Log Message
function log() {

    local Level=$1

    #   Only log if not in quiet mode, or it's a fatal error
    if [[ -z "$Quiet" ]] || [[ $Level -eq $LOG_FATAL ]]; then

        local Message="$2"
        local Timestamp=""

        if [ ! -z "$TimeFormat" ]; then
            Timestamp="[$(date "$TimeFormat")] "
        fi

        local ToWrite=

        if [ -z "$LogPrefix" ]; then
            ToWrite="$Timestamp${LogLevels[$Level]} $Message"
        else
            ToWrite="$Timestamp[ $LogPrefix ] ${LogLevels[$Level]} $Message"
        fi

        #   If log colouring is on, check if it's writing to an output file
        if [ ! -z "$ColourLog" ] && [[ "$OutputFile" == "-" ]]; then
            ToWrite="${LogColours[$Level]}""$ToWrite""$CLEAR"
        fi

        #   Attention and higher should be logged to STDERR, Info and Notice to STDOUT
        if [ $Level -ge $LOG_ATTENTION ]; then
            echo -e "$ToWrite" >&2
        else
            if [[ "$OutputFile" == "-" ]]; then
                echo -e "$ToWrite" >&1
            else
                echo -e "$ToWrite" >> "$OutputFile"
            fi
        fi

        #   If it's a fatal error, full exit
        if [ $Level -eq $LOG_FATAL ]; then
            cleanup 1
        fi
    fi
}

#   Helper function to allow asserting that required arguments are set.
function argSet() {
    local argToCheck="$1"
    local argName="$2"

    if [ -z "$argToCheck" ]; then
        log $LOG_FATAL "Required argument [ $argName ] not set!"
    fi
}

#   Helper function to allow checking for the existence of files on disk.
function fileExists() {
    local FilenameToCheck="$1"

    if [ ! -f "$FilenameToCheck" ] && [ ! $(which "$FilenameToCheck") ]; then
        if [ $# -le 1 ]; then
            log $LOG_ATTENTION "File [ $FilenameToCheck ] does not exist."
        fi
        return 1
    fi

    return 0
}

#   Helper function to allow checking for the existence of directories on disk.
function directoryExists() {
    local DirectoryToCheck="$1"

    if [ ! -d "$DirectoryToCheck" ]; then
        if [ $# -le 1 ]; then
            log $LOG_ATTENTION "Directory [ $DirectoryToCheck ] does not exist."
        fi
        return 1
    fi

    return 0
}

#   Helper function to either assert that a given directory does exist (creating it if necessary) or exiting if it cannot.
function assertDirectoryExists() {

    local DirectoryToCheck="$1"

    if ! directoryExists "$DirectoryToCheck"; then
        if ! mkdir -p "$DirectoryToCheck"; then
            log $LOG_FATAL "Failed to create directory [ $DirectoryToCheck ]!"
        fi

        log $LOG_NOTICE "Successfully created directory [ $DirectoryToCheck ]."
    fi
}

#   Main function, this is the entry point of the actual logic of the script, AFTER all of the input validation and top-level script pre-script set up has been completed.
function main() {

    ChecksumFile="$SourceDirectory/$ChecksumFilename"

    #   Check if the checksum file already exists
    if ! fileExists "$ChecksumFile" || [[ ! -z "$UpdateFlag" ]]; then
        log $LOG_INFO "Computing checksums for *.LIF files and writing them to [ $ChecksumFile ]..."
        CreateChecksums
    else
        log $LOG_INFO "Checksum file exists: [ $ChecksumFile ]!"
        CompareChecksums
    fi

    return
}

function CompareChecksums() {

    cd "$SourceDirectory"

    IFS=$'\n'
    LIFFiles=$(find . -type f -iname '*.lif' | sort)
    for F in $LIFFiles; do

        log $LOG_INFO "Verifying checksum for file [ $F ]..."

        CurrentChecksum="$(shasum -a 256 "$F" | awk '{print $1}')"
        CachedChecksum="$(grep "$(basename "$F")" "$ChecksumFilename" | awk -F, '{print $2}' | head -1)"

        #   Make sure the current file exists in the set of pre-computed checksums
        if [ -z "$(grep "$(basename "$F")" "$ChecksumFilename" | head -1)" ]; then
            log $LOG_ERROR "File [ $F ] does not have a checksum calculated for it. Unable to verify file integrity..."
            continue
        fi

        #   Make sure the cached checksum is not empty
        if [ -z "$CachedChecksum" ]; then
            log $LOG_ERROR "Cached checksum for file [ $F ] is empty. Unable to verify file integrity..."
            continue
        fi

        #   Check if the checksum of the current file matches what was cached earlier.
        if ! [[ "$CurrentChecksum" == "$CachedChecksum" ]]; then
            log $LOG_ERROR "File [ $F ] checksum is not correct! Expected [ $CachedChecksum ] - Got [ $CurrentChecksum ]."
        else
            log $LOG_NOTICE "File [ $F ] checksum matches cached value!"
        fi

    done

    return
}

function CreateChecksums() {

    cd "$SourceDirectory"

    IFS=$'\n'
    LIFFiles=$(find . -type f -iname '*.lif' | sort)
    for F in $LIFFiles; do
        log $LOG_INFO "Computing checksum for file [ $F ]..."

        CurrentChecksum=$(shasum -a 256 "$F" | awk '{print $1}')
        log $LOG_INFO "File [ $F ] has sha256 hash [ $CurrentChecksum ]"

        echo "$(basename "$F"),$CurrentChecksum" >> "$ChecksumFilename"
    done

    #   Make sure there's no duplicates in the file.
    Deduplicated="$(sort -u "$ChecksumFilename")"
    echo "$Deduplicated" > "$ChecksumFilename"

    return
}

#   Parse the command line arguments.  Add the flag name to the list (in alphabetical order), and add a ":" after if it requires an argument present.
#   The value of the argument will be located in the "$OPTARG" variable
while getopts "d:ho:quz" opt; do
    case "$opt" in
    d)  SourceDirectory="$OPTARG"
        ;;
    h)  HelpFlag=1
        ;;
    o)  OutputFile="$OPTARG"
        ;;
    q)  Quiet="-q"
        ;;
    u)  UpdateFlag=1
        ;;
    z)  ColourLog=
        ColourFlag="-z"
        ;;
    \?) HelpFlag=2
        ;;
    esac
done

case $HelpFlag in
    1)  helpMenu
        cleanup 0
        ;;
    2)  helpMenu
        cleanup 1
        ;;
esac

argSet "$OutputFile" "-o"

if [[ ! "$OutputFile" == "-" ]]; then

    #   Only assert this here, in case multiple -o arguments are given.
    #   Only create the file of the final argument.
    assertDirectoryExists "$(dirname "$OutputFile")"

    if ! fileExists "$OutputFile"; then
        #   Create the empty file.
        >"$OutputFile"
    fi
fi

#   Assert all of the required arguments are set here

#   argSet <Variable> <Command Line Flag>
argSet "$SourceDirectory" "-d"
#   ...

#   Other argument validation here...


#   Call main, running the full logic of the script.
main

cleanup 0
