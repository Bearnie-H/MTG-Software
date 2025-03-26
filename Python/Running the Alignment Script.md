# Running Joseph's Neurite/Rod Alignment Script

This document provides a guide for how to run the script for quantifying the alignment of rods or neurites with an image, Z-Stack, or video file. This guide assumes you have some minor familiarity with running Python scripts from a terminal window, but will cover everything necessary to run the script.

##  Where is the script?

The script and accompanying libraries are hosted on [GitHub](https://github.com/Bearnie-H/MTG-Software.git), and a local clone of this repository is located on the Desktop of this computer in the **MTG-Software/Python** folder. The script itself is named **Alignment-Analysis.py**, due to the original intention being solely rods. Fortunately, the orientation detection algorithm appears to provide meaningful and useful results on neurites as well, so it's being applied there too!

##  Running the Script

From a terminal window, navigate to the folder holding the script:

```bash
    cd ~/Desktop/MTG-Software/Python/
```

Next, you can run the script as follows to request the built-in help menu:

```bash
    python3 Alignment-Analysis.py --help
```

This help menu lists out **all** of the available parameters and inputs to the script. In order to provide a value to the script through the terminal, you must tell in *which* value you are setting with the double-dash name, followed by the value to use. For example, to work with the a file located on the desktop named "TestFile.png", you would pass this to the script as:

```bash
    python3 Alignment-Analysis.py --to-process "~/Desktop/TestFile.png"
```

Alternatively, if you have a set of images to analyze, all taken under meaningfully comparable conditions, you can analyze *all* images within a given folder by specifying the path to the folder of interest. For example, a folder on the desktop named "Neurites-2024-07-10" can be processed as:

```bash
    python3 Alignment-Analysis.py --to-process "~/Desktop/Neurites-2024-07-10"
```

Note that the value should be enclosed in double quotes in case there's spaces or other special characters in the filename. The script is smart enough to handle a variety of input file types. The script can handle 2D images, such as Maximum Intensity Projections, 3D Image stacks in either *.LIF or *.czi format, or a variety of video formats in the case of a time-series set of images. When processing an entire folder, the script need not process only one type of input file. It's entirely fine to mix and match between 2D images, Z-Stacks, and time-series videos.

The next input parameters of interest are those related to identifying the orientation of the **features** within the image:

 - --method [Value]
 - --angular-resolution [Value]
 - --length-scale [Value]
 - --image-resolution [Value]

The **--method** parameter determines which of the 4 algorithms within the script to use to determine alignment of the subjects of the image. This parameter is set to select the elliptical spatial filter by default, and this parameter does not need to be specified with another value. The other 3 algorithms were test-pieces to explore alternative methods of determining alignment which proved unsuitable for the highly overlapping rod+neurite images which must be analyzed.

The **--angular-resolution** parameter defines the number of degrees in each "bin" of the final histogram of orientations to report. This also has a performance impact, as finer angular resolutions require more processing to differentiate between the ranges of each histogram bin.

The **--length-scale** parameter defines the length, in physical units of micrometers, over which the features to process are meaningfully straight and coherent. In the case of rods, this should be the length of the rods themselves, while for neurites this should be a length-scale over which the neurites do not significantly curve or branch.

The **--image-resolution** parameter defines the relationship between pixels and physical distances within the image. This value is the number of micrometers per pixel of the image, and is used with the **--length-scale** parameter to dimensionalize the image.

The remaining arguments handle quality-of-life, and script overhead functions:

 - --log-file [Value]
 - --quiet
 - --dry-run
 - --validate
 - --headless

The **--log-file** parameter defines the path to a file into which to write all of the logging and status messages generated as the script executes. By default, the script prints all log messages to the running terminal, but this allows the log to be stored for later review.

The **--quiet** argument, if provided, disables all logging aside from fatal irrecoverable errors.

The **--dry-run** argument, if provided, disables all funtionality which modifies the file system. This will disable all file or folder generation from the script, allowing you to test parameters without necessarily wanting to store the results.

The **--validate** argument, if provided, will trigger the script to only validate the command-line arguments and exit.

The **--headless** argument, if provided, will disable all screen-based operations or interactions with the script. This is helpful if the script is being run via a network connection.

## Workflow of the Elliptical Spatial Filter

Assuming that the script is running with the elliptical spatial filter as the orientation determination algorithm, the workflow of this algorithm can be broken down into the following three stages.

### Stage 1 - Background Removal

The first stage of processing is to determine and subtract the background signal from the image to be processed. This is done by applying a Gaussian blur to the image of a specified size and width and treating the resulting blurred image as the background signal. Larger kernel sizes blur larger features away, and higher sigmas also more strongly mix all of the pixels within the blurring kernel.

### Stage 2 - Foreground Smoothing

The second stage of processing is to smooth out the result following the background subtraction. This can leave sharp brightness changes or otherwise negatively impact the edge features of the image. To counteract this, another Gaussian blur is applied to the image in order to re-smooth out these artefacts, using a smaller kernel than was used for background subtraction. Larger kernel sizes blur larger features away, and higher sigma values also more strongly mix all of the pixels within the blurring kernel.

### Stage 3 - Elliptical Spatial Filtering

The third stage of processing is to generate a highly elliptical "Mexican-Hat" filter to convolve with the background-subtracted and smoothed image. This filter is approximated as the difference of two Gaussian kernels, as described [here](https://en.wikipedia.org/wiki/Difference_of_Gaussians). These parameters define the total size of the filter, the spread of the Gaussians in the long and short axes of the ellipse, and the scale factor between the two Gaussians used to approxiamte the Mexican Hat. Larger kernels correlate strongly to larger features, and higher sigmas lead to "longer" ellipses in the given axis. Increasing scale factor increases the overall area of the ellipse, uniformly in both axes.

##  Script Outputs

The script generates a total of five outputs when not operating in **--dry-run** mode:

 1. An annotated video, showing the resulting orientation information overlaid on the original image, video, or Z-Stack. This is colour coded such that orientation angle maps to Hue, with 0 degrees being red, moving through yellow and green towards magenta as the angle approaches 180 degrees.
 2. A video (or still), showing the distribution of orientation angles within the image. This is normalized such that the mean angle of the histogram is 0 degrees, and the plot shows the distribution relative to the mean angle.
 3. A video showing the mean orientation angle and angular standard deviation as a function of time (or layer, in the case of a Z-Stack). This shows the actual mean angle, with 0 degrees corresponding to the horizontal.
 4. A .CSV file containing the numerical values for the time associated with each image, the mean angle, the angular standard deviation, measurement count, and the fraction of measurement values within one standard deviation of the mean.
 5. A .JSON file containing a breakdown of all of the configuration settings and provided values which the script was run with. This is useful for checking previous analysis attempts on a given file to see what parameters appeared to work well, or provide tracing for optimizing the parameters for a given file or set of files.

## Understanding "Good Alignment"

From the metrics generated and reported by this script, two stand out to define what a "well aligned" sample would look like. These are the angular standard deviation and the alignment fraction. Both must be taken into account, as under the case of a very large angular standard deviation, one would expect most of the measured orientations to be within one sigma of the mean. Therefore, one needs not only a high value for the alignment fraction, but more importantly a low value for the angular standard deviation. A simple way to generate a single reportable metric for the "goodness" of alignment can be to form the following quotient:

```
    AlignmentScore = AlignmentFraction / AngularStandardDeviation
```
