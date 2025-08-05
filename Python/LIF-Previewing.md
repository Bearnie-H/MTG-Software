
# LIF Image Previewing

As part of the process of finishing the acquisition of cortical explant images,
all of the captured images should be previewed to assess for imaging issues or
artefacts which may make analysis challenging. A **Python** script has been
written to streamline this process. This script is called *PreviewLIFFiles.py*.

## Running the Script

The preview script can be run from the command-line, like other scripts. You can
either provide the path to the folder containing the LIF files you want to
preview on the command-line, or you can copy the script from the desktop into
the folder and double-click on it to run it.

This script will search through the provided folder (or the one it's run from)
for all LIF files. For each file it finds, it will display to you the number of
image series contained, as well as the number of colour channels for each
series. In addition to displaying a printout of the details of the images within
the file, it will also display to the screen both a Minimum- and
Maximum-Intensity Projection of each colour channel for each series. These
projections are intended to provide a visualization for the fluorescent and
bright-field/transmitted light modalities to assess the images for artefacts or
errors.
