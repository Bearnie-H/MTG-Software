#   Mend-the-Gap: Python Software

This subsection of the Mend-the-Gap software repository contains the full set of
`Python` scripts, classes, and tools developed for image analysis, data
manipulation, and results presentation. All of the tools are collected here
together in a single directory for simplicity of access, with all `Python`
classes collected together in the shared `MTG` folder.

##  Installing the Repository

As this is part of the larger `MTG-Software` `git` repository, simply clone the
base repository and you will download all of the scripts and tools. You can
either copy and paste the following command into a suitable terminal, or
navigate to the web interface for the repository
[here](https://github.com/Bearnie-H/MTG-Software) and clone through your
browser.

```bash
git clone https://github.com/Bearnie-H/MTG-Software.git
```

Once you have the repository installed, navigate to the `Python/` directory and
find the `requirements.txt` text file. This contains the full set of required
`Python` modules required to run the software. This is formatted such that it
works directly with the standard `pip` package manager, and the required
packages can be installed with the following command:

```bash
pip install -r requirements.txt
```

##  Tool Interface Requirements

For each of the tools provided in this repository, they must conform to a basic
set of user interface requirements. The aim of this is to ease the learning
curve for either onboarding new collaborators into using the tools, or
introducing new tools to our existing collaborators. These requirements are
enumerated as follows:

1) A `help` menu within the tool itself, provided via the `argparse` package.
    This must contain a complete and meaningful description of what operations
    or use-case the tool serves, along with any additional details a user should
    know prior to actually using the tool. In short, this must contain enough
    information for a user to know what the tool does at a high-level.

2) A description of the set of command-line arguments or operational flags the
    tool accepts. Using the `argparse` package, this is provided by default.
    Whether an argument is required or optional must be specified, as well as
    what *type* of argument to provide, such as an integer or fractional number.
    Finally, this must include a description of what this argument does in terms
    of the operation of the tool, and how variations of the value affect the
    operation.

3) A thorough utilization of the `Logger.py` class to provide user feedback and
    progress logging of the operation of the tool. In order to maximize user
    understanding of the usage and operation of the tool, all notable operations
    must have accompanying log messages. These may be re-directed to a
    user-specifiable log file with a command-line argument, and there may be a
    separate flag to change the level of logging verbosity. By default, logging
    must be sufficient for a user to understand where within the workflow the
    tool is at any reasonable time interval.
