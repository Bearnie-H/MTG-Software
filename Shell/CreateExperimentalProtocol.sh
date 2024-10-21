#!/usr/bin/env bash

#   Author: Joseph Sadden
#   Date:   10th October, 2024

#   Purpose:
#
#       This script prepares and creates a new LaTeX file and surrounding
#       directory for preparing a stand-alone experimental protocol for
#       experiments I need to perform and share with others.

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
ProtocolDirectory=
ProtocolName=
DocumentResourcesDirectory=
ProtocolFilename=

#   Function to display a help/usage menu to the user in a standardized format.
function helpMenu() {

    #   Get the name of the executable
    local scriptName=$(basename "$0")

    #   Print the current help menu/usage information to the user
    echo -e "
    $BLUE$scriptName   -   A bash tool to prepare a new default starting LaTeX document describing an experimental protocol.$WHITE

    $GREEN$scriptName$YELLOW [-h] [-o Output-File] [-q] [-z]$WHITE

    "$YELLOW"Document Options:$WHITE
        $BLUE-d$WHITE  -    Directory. The directory in which to create the protocol document.
        $BLUE-n$WHITE  -    Protocol Name. The name of the experiment protocol to create.

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

    #   Create the base directory to create the protocol document structure into.
    CreateProtocolDirectory "$ProtocolDirectory"

    #   Create the default LaTeX files within the directory...
    CreateLaTeXFiles "$ProtocolDirectory" "$DocumentResourcesDirectory" "$ProtocolName"

    #   Create the Makefile to actually build the final PDF.
    CreateMakefile "$ProtocolDirectory" "$ProtocolFilename"

    #   Add a .gitignore file to ignore everything in the build directory
    echo "build/**" >> "$ProtocolDirectory/.gitignore"

    return
}

function CreateProtocolDirectory() {

    local RootDirectory="$1"
    DocumentResourcesDirectory="$RootDirectory/Resources"

    assertDirectoryExists "$RootDirectory"
    assertDirectoryExists "$RootDirectory/build"
    assertDirectoryExists "$RootDirectory/Sections"
    assertDirectoryExists "$DocumentResourcesDirectory"
    assertDirectoryExists "$DocumentResourcesDirectory/Bibliography"
    assertDirectoryExists "$DocumentResourcesDirectory/Figures"

    return
}

function CreateLaTeXFiles() {

    local RootDirectory="$1"
    local ResourcesDirectory="$2"
    local ProtocolName="$3"

    local Filename="$(echo "$ProtocolName" | sed 's/[ \t]\+/-/g' | sed 's/-\+/-/g' | sed 's/\.tex$//g')"
    Filename="$RootDirectory/$Filename.tex"

    ProtocolFilename="$(basename "$Filename")"

    if fileExists "$Filename"; then
        log $LOG_FATAL "Will not overwrite existing LaTeX file!"
        stop 1
    fi

    local Title="$ProtocolName"

    local Contents="
%%  Author: Joseph Sadden
%%  Date:   $(date "+%-d %B, %Y")

\documentclass[a4paper, 11pt]{article}

\def\RevisionNumber{1.0}

\input{Resources/Packages.tex}
\input{Resources/Macros.tex}

\hypersetup{
    pdftitle    = {$Title},
    pdfauthor   = {Joseph Sadden},
    pdfsubject  = {Placeholder Subject},
}

\title{Experimental Protocol\\\\$Title}
\author{Joseph Sadden\thanks{jsadden@ece.ubc.ca}}
\date{Document Created: $(date "+%B %-d, %Y")\\\\Last Updated: \today{}}

\bibliography{Resources/Bibliography/Bibliography}

\begin{document}

\maketitle{}
\tableofcontents{}
\cleardoublepage{}

%   Start document contents here...
\input{Sections/Purpose.tex}
\input{Sections/Required-Materials.tex}
\input{Sections/Preparations.tex}
\input{Sections/Experimental-Protocol.tex}
\input{Sections/Clean-Up.tex}
\input{Sections/Change-Log.tex}

\end{document}
"

    echo -n "$Contents" > "$Filename"

    echo "
%%  Author: Joseph Sadden
%%  Date:   $(date "+%-d %B, %Y")

\\section{Purpose of the Experiment}\\label{sec:Experimental Purpose}
" >> "$RootDirectory/Sections/Purpose.tex"

    echo "
%%  Author: Joseph Sadden
%%  Date:   $(date "+%-d %B, %Y")

\\section{Required Equipment \& Materials}\\label{sec:Required Equipment and Materials}
" >> "$RootDirectory/Sections/Required-Materials.tex"

    echo "
%%  Author: Joseph Sadden
%%  Date:   $(date "+%-d %B, %Y")

\\section{Equipment \& Material Preparations}\\label{sec:Equipment and Material Preparations}
" >> "$RootDirectory/Sections/Preparations.tex"

    echo "
%%  Author: Joseph Sadden
%%  Date:   $(date "+%-d %B, %Y")

\\section{Experimental Protocol}\\label{sec:Experimental Protocol}
" >> "$RootDirectory/Sections/Experimental-Protocol.tex"

    echo "
%%  Author: Joseph Sadden
%%  Date:   $(date "+%-d %B, %Y")

\\section{Experiment \& Equipment Clean-Up}\\label{sec:Experiment and Equipment Clean-Up}
" >> "$RootDirectory/Sections/Clean-Up.tex"

    echo "
%%  Author: Joseph Sadden
%%  Date:   $(date "+%-d %B, %Y")

\cleardoublepage{}
\\section{Change Log}\\label{sec:Change Log}

\\begin{longtable}[c]{ l | R{0.25} | l | R{0.4} }

    \\toprule
    Revision & Contributors & Date & Description \\\\
    \\endhead

    \\midrule
    \\multicolumn{4}{c}{\\tablename\\  \\thetable\\ --- Continued on Next Page} \\\\
    \\bottomrule \\endfoot

    \\bottomrule \\caption{$Title Change Log}\\label{tab:$Title Change Log} \\endlastfoot

    \\midrule
    \RevisionNumber{} & Joseph Sadden & \today{} & Initial Document Creation. \\\\

\\end{longtable}
" >> "$RootDirectory/Sections/Change-Log.tex"

    touch "$ResourcesDirectory/Bibliography/Bibliography.bib"

    CreateDefaultMacrosFile "$ResourcesDirectory/Macros.tex"

    CreateDefaultPackagesFile "$ResourcesDirectory/Packages.tex"

    #   ...

    return

}

function CreateDefaultMacrosFile() {

    local Filename="$1"

    local Contents="
%%  Author: Joseph Sadden
%%  Date:   $(date "+%-d %B, %Y")

%   Define a new column type to allow paragraphs with ragged-right line endings of a defined width.
\newcolumntype{R}[1]{>{\raggedright\let\newline\arraybackslash\hspace{0pt}}p{#1 \textwidth}}
\newcolumntype{L}[1]{>{\raggedleft\let\newline\arraybackslash\hspace{0pt}}p{#1 \textwidth}}

\newcommand{\subfigureautorefname}{\figureautorefname}

\renewcommand{\chapterautorefname}{Chapter}
\renewcommand{\sectionautorefname}{\(\S\)}
\renewcommand{\subsectionautorefname}{\(\S\)}
\renewcommand{\subsubsectionautorefname}{\(\S\)}

\newcommand{\UPDATE}[1]{\textbf{UPDATE\@: #1}}
\newcommand{\NOTE}[1]{\textbf{[NOTE: #1]}}
\newcommand{\AddMore}{\NOTE{Add more to this section or remove it entirely\dots}}

\newcommand{\NA}{\textsc{n/a}}
\newcommand{\eg}{e.g.,\ }
\newcommand{\ie}{i.e.,\ }
\newcommand{\etal}{\emph{et al}}

\newcommand{\nicefrac}[2]{\sfrac{#1}{#2}}

\newcommand{\LaTeXPackage}[1]{\href{http://www.ctan.org/macros/latex/contrib/#1}{\texttt{#1}}}
\newcommand{\LaTeXMiscPackage}[1]{\href{http://www.ctan.org/macros/latex/contrib/misc/#1.sty}{\texttt{#1}}}
\newcommand{\BibTeX}{Bib\TeX}

\DeclareUrlCommand\DOI{}
\newcommand{\doi}[1]{\href{http://dx.doi.org/#1}{\DOI{doi:#1}}}
\newcommand{\webref}[2]{\href{#1}{#2}\footnote{\url{#1}}}

%   Typeset common types of words or names
\newcommand{\Filename}[1]{\texorpdfstring{\texttt{#1}}{#1}}
\newcommand{\FileType}[1]{\Filename{*.#1}}

\newcommand{\ProductName}[1]{\texorpdfstring{\texttt{#1}}{#1}}

\newcommand{\ProgramName}[1]{\texorpdfstring{\texttt{#1}}{#1}}

%   Typesetting of suppliers or manufacturers
\newcommand{\CompanyName}[1]{\textbf{\texttt{#1}}}

%   Common chemicals or reagents.
\newcommand{\ChemicalName}[2]{\texorpdfstring{\ensuremath{\texttt{#1}}}{#2}}

%   Typesetting of common units or fractions
\newcommand{\Dimension}[1]{\ensuremath{\textrm{#1}}}
\newcommand{\DimensionalQuantity}[2]{#1~\Dimension{#2}}
\newcommand{\DimensionalQuantityPM}[3]{#1\ensuremath{\pm}#2~\Dimension{#3}}

\newcommand{\micro}{\Dimension{\textmu}}

\newcommand{\degC}{\Dimension{\text{\textdegree{}C}}}

\newcommand{\wtbyvolFrac}{\Dimension{\%~\nicefrac{wt.}{v.}}}
\newcommand{\wtbywtFrac}{\Dimension{\%~\nicefrac{wt.}{wt.}}}
\newcommand{\volbyvolFrac}{\Dimension{\%~\nicefrac{v.}{v.}}}

\newcommand{\SciNotation}[2]{\ensuremath{#1 \times 10^{#2}}}

\newcommand{\IntervalCC}[2]{\ensuremath{\left[#1, #2\right]}}
\newcommand{\IntervalCO}[2]{\ensuremath{\left[#1, #2\right)}}
\newcommand{\IntervalOC}[2]{\ensuremath{\left(#1, #2\right]}}
\newcommand{\IntervalOO}[2]{\ensuremath{\left(#1, #2\right)}}

\newcommand{\Differential}[1]{\ensuremath{d#1}}
\newcommand{\PartialDifferential}[1]{\ensuremath{\partial #1}}
\newcommand{\DifferentialN}[2]{\ensuremath{d^{#2}#1}}
\newcommand{\PartialDifferentialN}[2]{\ensuremath{\partial^{#2} #1}}

\newcommand{\Derivative}[2]{\ensuremath{\frac{\Differential{#1}}{\Differential{#2}}}}    %   Simple derivative expression.
\newcommand{\DerivativeN}[3]{\ensuremath{\frac{\DifferentialN{#1}{#3}}{\DifferentialN{#2}{#3}}}} %   Simple expression for Nth derivative.
\newcommand{\PartialDerivative}[2]{\ensuremath{\frac{\PartialDifferential{#1}}{\PartialDifferential{#2}}}}   %   Simple expression for the partial derivative.
\newcommand{\PartialDerivativeN}[3]{\ensuremath{\frac{\PartialDifferentialN{#1}{#3}}{\PartialDifferentialN{#2}{#3}}}}   %   Simple expression for the partial derivative.

\newcommand{\order}[1]{\ensuremath{\Function{\mathcal{O}}{#1}}}
\newcommand{\oforder}[1]{\ensuremath{\sim \order{#1}}}
"

    echo "$Contents" > "$Filename"
}

function CreateDefaultPackagesFile() {

    local Filename="$1"

    local Contents="
%%  Author: Joseph Sadden
%%  Date:   $(date "+%-d %B, %Y")

\usepackage[printonlyused,nohyperlinks]{acronym}
\usepackage{amsfonts}
\usepackage{amsmath}
\usepackage{amssymb}
\usepackage{booktabs}
\usepackage{color}
\usepackage{fancyhdr}
\usepackage{geometry}
\usepackage{graphicx}
\usepackage{listings}
\usepackage{longtable}
\usepackage{multirow}
\usepackage[super]{nth}
\usepackage{paralist}
\usepackage{subfig}
\usepackage{tabularx}
\usepackage{textgreek}
\usepackage{xfrac}

\ifdefined\isdraft

    %   Skip including figures in draft-mode.
    \usepackage[allfiguresdraft]{draftfigure}

    % Until this is ready to submit, add a watermark to the document to make it
    % VERY clear that this is a draft and not a final copy.
    \usepackage{draftwatermark}
    \DraftwatermarkOptions{text=DRAFT (DO NOT DISTRIBUTE)\today}

    \fancypagestyle{mypagestyle}[fancydefault]{
        \lhead{Experimental Protocol Document}
        \chead{DRAFT}
        \rhead{Author: Joseph Sadden}
        \lfoot{Last Updated: \today}
        \rfoot{Revision: \RevisionNumber}
    }

\else

    \fancypagestyle{mypagestyle}[fancydefault]{
        \lhead{Experimental Protocol Document}
        \rhead{Author: Joseph Sadden}
        \lfoot{Last Updated: \today}
        \rfoot{Revision: \RevisionNumber}
    }

\fi

\pagestyle{mypagestyle}

\usepackage[
    sorting=none,
    backend=biber,
    backref=true,
    backrefstyle=two,
    citestyle=numeric-comp,
    bibstyle=numeric,
]{biblatex}

\usepackage[
    bookmarks,
    bookmarksnumbered,
    linktocpage,
    hidelinks,
]{hyperref}
"

    echo "$Contents" > "$Filename"

    return
}

function CreateMakefile() {

    local RootDirectory="$1"
    local RootFilename="$2"

    local RootDocumentName="$RootDirectory/$RootFilename"

    NewLaTeXMakefile.sh -n "$RootDocumentName"

    return
}


#   Parse the command line arguments.  Add the flag name to the list (in alphabetical order), and add a ":" after if it requires an argument present.
#   The value of the argument will be located in the "$OPTARG" variable
while getopts "d:hn:o:qz" opt; do
    case "$opt" in
    d)  ProtocolDirectory="$OPTARG"
        ;;
    h)  HelpFlag=1
        ;;
    n)  ProtocolName="$OPTARG"
        ;;
    o)  OutputFile="$OPTARG"
        ;;
    q)  Quiet="-q"
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
argSet "$ProtocolDirectory" "-d"
argSet "$ProtocolName" "-n"
#   ...

#   Other argument validation here...
ProtocolDirectory="$(dirname "$ProtocolDirectory/$ProtocolName")"
ProtocolName="$(basename "$ProtocolName")"

#   Call main, running the full logic of the script.
main

cleanup 0
