#!/usr/bin/env python3

#   Author: Joseph Sadden
#   Date:   29th September, 2022

from __future__ import annotations

#   Import standard library packages as required
import typing
import os
#   ...

#   Import third-party libraries as required
import cv2
import numpy as np
import matplotlib
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.image import AxesImage
import matplotlib.pyplot as plt
#   ...

#   Import any local libraries or packages as required
from .Logger import Logger
#   ...

#   Define any module-level globals
DefaultHoldTime: int = 3

#   Provide a Logger, to allow writing log messages.
#       By default, force it to write to /dev/null (or equivalent) so it does nothing.
LogWriter: Logger = Logger(OutputStream=open(os.devnull, "w"))

def Default_SIGINT_Handler(signal, frame) -> None:
    """
    SigINT_Handler

    This function explicitly handles SIGINT being sent to this process, as this
    can occur either from a keyboard interrupt or from the batch-job script.
    """
    raise KeyboardInterrupt(f"SIGINT ({signal}) received by script, raising exception and failing out now!\n{frame}")

def DisplayImage(Description: str = "", Image: np.ndarray = None, HoldTime: int = DefaultHoldTime, Topmost: bool = False, ShowOverride: bool = True) -> int:
    """
    DisplayImage

    This function displays a given image to the screen in an OpenCV NamedWindow, allowing viewing of the image.

    Description:
        The title of the window to display the image in.
    Image:
        The image to be displayed.
    HoldTime:
        The duration (in seconds) to display the image for. 0 indicates to
        display indefinitely.
    Topmost:
        Boolean flag for whether the window should be displayed in front of all
        other windows.
    ShowOverride:
        An override to disable showing images when set to False. Allows simpler
        enabling/disabling during logging or development.

    Return (int):
        The key-code which was pressed during display of the image, if any.
    """
    return DisplayImages(Images=[(Description, Image)], HoldTime=HoldTime, Topmost=Topmost, ShowOverride=ShowOverride)

def DisplayImages(Images: typing.List[typing.Tuple[str, np.ndarray]] = ["", None], HoldTime: int = DefaultHoldTime, Topmost: bool = False, ShowOverride: bool = True) -> int:
    """
    DisplayImages

    Like DisplayImage(), but for more than one image.

    Multiple images will be displayed in distinct Windows, and can be rearranged as desired.

    Images:
        A list of tuples, containing the window name and image pixels to display
    HoldTime:
        The duration (in seconds) to display the image for. 0 indicates to
        display indefinitely.
    Topmost:
        Boolean flag for whether the window should be displayed in front of all
        other windows.
    ShowOverride:
        An override to disable showing images when set to False. Allows simpler
        enabling/disabling during logging or development.

    Return (int):
        The key-code which was pressed during display of the image, if any.
    """

    if ( not ShowOverride ):
        return

    if ( HoldTime <= 0 ):
        HoldTime = 0

    WindowFlags: int = cv2.WINDOW_NORMAL

    #   Keep a record of the descriptions generated for the un-described images to display,
    #       to be able to destroy their display windows later.
    UnmanagedDescriptions: typing.List[str] = []
    ImagesActive: bool = False

    XPos: int = 0
    YPos: int = 0
    for Index, (Description, Image) in enumerate(Images):

        if ( Image is None ):
            LogWriter.Warnln(f"Provided \"Image\" {Index} is None, nothing to display.")
            continue

        ImagesActive = True
        if ( Description is None ) or ( Description == "" ):
            LogWriter.Warnln(f"No \"Description\" provided for Image {Index}...")
            Description = f"Display Image {Index} - {Image.shape[1]}x{Image.shape[0]}"
            if ( len(Image.shape) == 3 ):
                Description += " (RGB)"
            elif ( len(Image.shape) == 4):
                Description += " (RGBA)"
            UnmanagedDescriptions.append(Description)

        if ( XPos != 0 ) and ( XPos + Image.shape[1] >= 1440 ):
            YPos += Image.shape[0]
            XPos = 0

        cv2.namedWindow(Description, WindowFlags)
        if ( Topmost ):
            cv2.setWindowProperty(Description, cv2.WND_PROP_TOPMOST, 1)
        cv2.moveWindow(Description, XPos, YPos)
        cv2.imshow(Description, Image)

        XPos += Image.shape[1]
        if ( XPos >= 1440 ):
            YPos += Image.shape[0]
            XPos = 0

    Key: int = 0
    if ( ImagesActive ):
        Key = cv2.waitKey(round(HoldTime * 1000))

        #   Allow pressing the "P" key to pause, overriding the HoldTime setting until another key is pressed.
        while ( Key in [ord(x) for x in 'pP']):
            LogWriter.Println(f"[ P ] key pressed, pausing display until another key is pressed...")
            Key = cv2.waitKey(0)

        #   Allow for the "S" key to save all currently displayed images to disk in the local directory.
        if ( Key in [ord(x) for x in 'sS' ]):
            [cv2.imwrite(os.path.join(os.getcwd(), os.path.splitext(Description)[0] + '.png'), Image) for (Description, Image) in Images]

        [cv2.destroyWindow(Description) for (Description, _) in Images]
        [cv2.destroyWindow(Description) for Description in UnmanagedDescriptions]
        _ = cv2.waitKey(1)

    return Key

def DisplayFigure(Description: str = "", Fig: Figure = None, HoldTime: int = DefaultHoldTime, Topmost: bool = False, ShowOverride: bool = True) -> int:
    """
    DisplayFigure

    Like DisplayImage(), but for a matplotlib Figure instance. This will convert
    the figure to a rasterized canvas pixel bitmap before displaying.

    Description:
        The title of the window to display the Figure in.
    Fig:
        The Figure to be displayed.
    HoldTime:
        The duration (in seconds) to display the Figure for. 0 indicates to
        display indefinitely.
    Topmost:
        Boolean flag for whether the window should be displayed in front of all
        other windows.
    ShowOverride:
        An override to disable showing Figures when set to False. Allows simpler
        enabling/disabling during logging or development.

    Return (int):
        The key-code which was pressed during display of the image, if any.
    """

    return DisplayFigures(Figures=[(Description, Fig)], HoldTime=HoldTime, Topmost=Topmost, ShowOverride=ShowOverride)

def DisplayFigures(Figures: typing.List[typing.Tuple[str, Figure]], HoldTime: int = DefaultHoldTime, Topmost: bool = False, ShowOverride: bool = True) -> int:
    """
    DisplayFigures

    Like DisplayFigure(), but for multiple Figures.
    Like DisplayImages(), but for matplotlib Figure() instances.

    Figures:
        A list of Tuples containing Description and Fig values as if this were a
        call to DisplayFigure().
    HoldTime:
        The duration (in seconds) to display the Figure for. 0 indicates to
        display indefinitely.
    Topmost:
        Boolean flag for whether the window should be displayed in front of all
        other windows.
    ShowOverride:
        An override to disable showing Figures when set to False. Allows simpler
        enabling/disabling during logging or development.

    Return (int):
        The key-code which was pressed during display of the image, if any.
    """

    return DisplayImages(Images=[(Description, CanvasToImage(FigureCanvasAgg(Fig))) for (Description, Fig) in Figures], HoldTime=HoldTime, Topmost=Topmost, ShowOverride=ShowOverride)

def GammaCorrection(Image: np.ndarray = None, Gamma: float = 1.0, Minimum: int = None, Maximum: int = None) -> np.ndarray:
    """
    GammaCorrection

    This function applies the standard gamma-based rescaling algorithm of image
    brightness values.  First, the image values are exponentiated to the
    Gamma'th power, then linearly rescaled to the range defined by Minimum and
    Maximum.

    A Gamma value of 1 indicates a linear contrast stretch to fill out the full
    range [Minimum, Maximium].

    Image:
        The image to have the brightness values rescaled for.
    Gamma:
        The exponent to raise the pixel values by during the rescaling.
    Minimum:
        The final minimum brightness value to rescale to.
    Maximum:
        The final maximum brightness value to rescale to.

    Return (np.ndarray):
        A new np.ndarray instance containing the brightness-scaled original image.
    """

    if ( Image is None ):
        raise ValueError(f"Image must be provided.")

    if ( 0 >= Gamma ):
        raise ValueError(f"Gamma must be provided and be a positive real number.")

    OriginalDtype: np.dtype = Image.dtype
    Limits = None
    if ( np.issubdtype(OriginalDtype, np.integer)):
        Limits = np.iinfo(OriginalDtype)
    elif ( np.issubdtype(OriginalDtype, np.floating)):
        Limits = np.finfo(OriginalDtype)
    else:
        raise TypeError(f"Numpy NDArray has non-integral and non-floating point dtype!")

    if ( Minimum is None ):
        Minimum = Limits.min

    if ( Maximum is None ):
        Maximum = Limits.max

    #   Perform the non-linear exponentiation operation
    #       allowing short-cutting for the no-op of exponentiation by 1.
    Scaled = Image.astype(np.float64)
    if ( Gamma != 1.0 ):
        Scaled = Scaled ** Gamma

    #   Linearly re-scale the resulting image to the desired min/max range provided.
    Offset = np.min(Scaled) - Minimum
    if ( Offset != 0.0 ):
        Scaled = Scaled - (np.min(Scaled) - Minimum)

    if ( np.max(Scaled) != 0 ):
        ScaleFactor = Maximum / np.max(Scaled)
        if ( ScaleFactor != 1.0 ):
            Scaled *= ScaleFactor

    return Scaled.astype(OriginalDtype)

def ConvertTo8Bit(Image: np.ndarray) -> np.ndarray:
    """
    ConvertTo8Bit

    This function will take the current image and both convert it to 8-bit, while
    also scaling the brightness linearly to fill the full range [0, 255].

    Image:
        The original image, to convert and linearly rescale brightness values for.

    Return (np.ndarray):
        A new np.ndarray instance, containing the converted and brightness-scaled
        pixel values.
    """

    if ( Image is None ):
        raise ValueError(f"Image must not be None")

    return GammaCorrection(Image=Image, Gamma=1, Minimum=0, Maximum=255).astype(np.uint8)

def RotateFrame(Frame: np.ndarray = None, Theta: float = 0.0) -> np.ndarray:
    """
    RotateFrame

    This function performs a counter-clockwise rotation of the angle Theta (in
    degrees) of the provided image. The rotation occurs about a point exactly at
    the centre of the image.

    Frame:
        The original image to rotate.
    Theta:
        The counter-clockwise rotation angle, in degrees, to rotate the frame
        by.

    Return (np.ndarray):
        The new, rotated image.
    """

    if ( Frame is None ):
        return Frame

    if ( Theta is None ) or ( Theta == 0.0 ):
        return Frame

    FrameHeight, FrameWidth = Frame.shape[0], Frame.shape[1]
    Centre = (FrameWidth / 2.0, FrameHeight / 2.0)

    RotationMatrix = cv2.getRotationMatrix2D(Centre, Theta, scale=1.0)

    return cv2.warpAffine(Frame, RotationMatrix, (FrameWidth, FrameHeight), borderMode=cv2.BORDER_REPLICATE)

def PrepareFigure(FigureDPI: int = 96, FigureSizeIn: typing.Tuple[float, float] = (10.8, 7.2), FigureSizePx: typing.Tuple[int, int] = None, Interactive: bool = False) -> Figure:
    """
    PrepareFigure

    This function wraps creating a new figure for generating the relevant
    matplotlib plotted figures for the results of the flow profile computations.
    This safely abstracts away whether or not we're working in dry-run mode
    or not.

    FigureDPI:
        The dots-per-inch resolution of the figure to generate.
    FigureSizeIn:
        The figure size, in units of inches. Width x Height.
    FigureSizePx:
        The figure size, in units of pixels. Width x Height.
        If not None, this overrides the value for FigureSizeIn

    Return (Figure):
        The returned figure,
    """

    if ( FigureSizePx is not None ):
        FigureSizeIn = (FigureSizePx[0] / float(FigureDPI), FigureSizePx[1] / float(FigureDPI))

    if ( Interactive ):
        plt.figure(figsize = FigureSizeIn, dpi = FigureDPI)
    else:
        return Figure(figsize = FigureSizeIn, dpi = FigureDPI)

def SetFigureSize(Figure: Figure, Resolution: typing.Tuple[int, int]) -> Figure:
    """
    SetFigureSize

    This function is a simple helper function for resizing
    matplotlib.pyplot.Figure() objects with dimensions based on desired pixel
    values, rather than the library-standard measure of inches.

    Figure:
        The matplotlib.pyplot.Figure object to resize.
    Resolution:
        The desired (width, height) resolution, expressed in pixels.

    Return (matplotlib.pyplot.Figure):
        The resized matplotlib.pyplot.Figure object.
    """

    dpi = float(Figure.get_dpi())

    Figure.set_size_inches(Resolution[0] / dpi, Resolution[1] / dpi)

    return Figure

def FigureToImage(Fig: Figure = None) -> np.ndarray:
    """
    FigureToImage

    This function converts the canvas of a matplotlib Figure() into a bitmap
    pixel array, suitable for use with OpenCV image manipulation functions.

    Fig:
        The matplotlib Figure() instance to convert to an image.

    Return (np.ndarray):
        The newly created pixel bitmap array containing the view of the Figure canvas.
    """

    if ( Fig is None ):
        return None

    return CanvasToImage(FigureCanvasAgg(Fig))

def CanvasToImage(Canvas: FigureCanvasAgg = None) -> np.ndarray:
    """
    CanvasToImage

    This function converts the underlying Canvas of a matplotlib Figure
    into a byte array suitable for use as a BGR-colour image with OpenCV
    functions.

    Canvas:
        The Canvas (using Agg backend) of the figure to be converted into an image.

    Return (np.ndarray):
        The corresponding BGR-colour byte array of the provided Canvas, suitable
        for use with all of OpenCV.
    """

    if ( Canvas is None ):
        raise ValueError(f"CanvasToImage parameter Canvas is [ None ]!")

    Canvas.draw()
    Image: np.ndarray = cv2.cvtColor(
            np.frombuffer(
                Canvas.tostring_rgb(),
                dtype=np.uint8
                ).reshape(
                    Canvas.get_width_height()[::-1] + (3,)
                    ),
            cv2.COLOR_RGB2BGR
        )

    return Image

def BGRToGreyscale(Image: np.ndarray = None) -> np.ndarray:
    """
    BGRToGreyscale

    This function converts a BGR colour image to a greyscale image. This is safe
    to call on an image which is already greyscale, unlike the OpenCV cvtColor()
    function.

    Image:
        The original image to convert from BGR colour to greyscale.

    Return (np.ndarray):
        The newly created greyscale pixel bitmap from the original image.
    """

    if ( Image is None ) or ( len(Image.shape) == 2 ):
        return Image

    return cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)

def GreyscaleToBGR(Image: np.ndarray = None) -> np.ndarray:
    """
    GreyscaleToBGR

    This function converts a greyscale image to BGR colour, by duplicating the grey channel
    to fill out all three of the output colour channels. This is safe to call on
    an existing multi-colour image, unlike the OpenCV cvtColor() function. This is a no-op
    on multi colour-channel images.

    Image:
        The greyscale image to convert to BGR colour space.

    Return (np.ndarray):
        The newly created BGR colour-space pixel bitmap array.
    """

    if ( Image is None ) or ( len(Image.shape) == 3 ):
        return Image

    return cv2.cvtColor(Image, cv2.COLOR_GRAY2BGR)

def UniformRescaleImage(Image: np.ndarray = None, ScalingFactor: float = 1.0, Interpolation: int = cv2.INTER_AREA) -> np.ndarray:
    """
    UniformRescaleImage

    This function performs a uniform size rescaling of a given image, increasing or decreasing it along both the X and Y
    axes by some uniform scaling factor, and using a given interpolation method.

    Image:
        The original image to rescale.
    ScalingFactor:
        The multiplicative scaling factor to increase the size of the image by.
    Interpolation:
        The pixel-value interpolation method to use. See OpenCV interpolation methods for details.

    Return (np.ndarray):
        The new, rescaled pixel bitmap array.
    """

    if ( Image is None ):
        raise ValueError(f"Image must be provided")

    if ( ScalingFactor <= 0 ):
        raise ValueError(f"ScalingFactor must be a positive real number")

    NewShape: typing.Tuple[int, int] = (int(round(ScalingFactor * Image.shape[1])), int(round(ScalingFactor * Image.shape[0])))

    return cv2.resize(Image, NewShape, interpolation=Interpolation)

def ComputeZeroNormalizedCrossCorrelation(TemplateImage: np.ndarray = None, TestImage: np.ndarray = None) -> float:
    """
    ComputeZeroNormalizedCrossCorrelation

    This function computes the Zero-Mean Normalized Cross-Correlation between
    two images. The images must be the same size, and the same colour-space.
    This is useful as a determination of the similarity of two images.

    This computes the full cross-correlation of the two images with the mean
    removed and pixel standard deviation normalized to 1.0 over the image.

    TemplateImage:
        The "template", or ideal image to compare against.
    TestImage:
        The "test", or real image to compare against the template.

    Return (float):
        The cross-correlation score, a value on the range [-1, 1] indicating how
        well the test image matches the template image.
    """

    if ( TestImage is None ) or ( TemplateImage is None ):
        raise ValueError(f"TestImage and TemplateImage must both be provided.")

    if ( TemplateImage.shape != TestImage.shape ):
        raise ValueError(f"Image shapes must be the same.")

    NormalizingFactor: float = (1.0 / np.prod(TemplateImage.shape))
    TemplateStdev: float = np.std(TemplateImage)
    TestStdev: float = np.std(TestImage)

    if ( 0 == TemplateStdev ) or ( 0 == TestStdev ):
        return -1

    return float(
        NormalizingFactor *
        np.sum(
            (1.0 / (TemplateStdev * TestStdev)) * \
            (TemplateImage.astype(np.float64) - np.mean(TemplateImage)) * \
            (TestImage.astype(np.float64) - np.mean(TestImage))
        )
    )

def LowPassFilterImage(*, Filename: str = None, Image: np.ndarray = None, CutOff: float = 1.0) -> np.ndarray:
    """
    LowPassFilterImage

    This function performs a spatial spectral low-pass filter over the given
    image (pixel array or file), returning the image with high frequency
    components removed.

    Filename:
        The full file path to open and read the input pixel data from
    Image:
        The original pixel data to filter
    CutOff:
        The cut-off frequency, normalized to the scale [0,1]. Any frequencies
        higher than this value multiplied by the maximum spatial frequency will
        be removed from the image.

    Return (np.ndarray):
        The resulting spatially low-pass filtered image.
    """
    return BandPassFilterImage(
        Filename=Filename,
        Image=Image,
        PassBand=(0, CutOff),
    )

def HighPassFilterImage(*, Filename: str = None, Image: np.ndarray = None, CutIn: float = 1.0) -> np.ndarray:
    """
    HighPassFilterImage

    This function performs a spatial spectral high-pass filter over the given
    image (pixel array or file), returning the image with low frequency
    components removed.

    Filename:
        The full file path to open and read the input pixel data from
    Image:
        The original pixel data to filter
    CutOff:
        The cut-off frequency, normalized to the scale [0,1]. Any frequencies
        lower than this value multiplied by the maximum spatial frequency will
        be removed from the image.

    Return (np.ndarray):
        The resulting spatially high-pass filtered image.
    """
    return BandPassFilterImage(
        Filename=Filename,
        Image=Image,
        PassBand=(CutIn, 1.0),
    )

def BandPassFilterImage(*, Filename: str = None, Image: np.ndarray = None, PassBand: typing.Tuple[float, float] = (0, 1.0)) -> np.ndarray:
    """
    BandPassFilterImage

    This function performs a spatial spectral band-pass filter over the given
    image (pixel array or file), returning the image with frequency components
    outside of the passband removed.

    Filename:
        The full file path to open and read the input pixel data from
    Image:
        The original pixel data to filter
    PassBand:
        A tuple of values on the range [0, 1], indicating the cut-in and cut-off
        of the passband. These are the values provided to LowPassFilterImage()
        and HighPassFilterImage() respectively.

    Return (np.ndarray):
        The resulting spatially low-pass filtered image.
    """

    return ReconstructImageFromSpectrum(
        Spectrum=ComputeFilteredDCTSpectrum(
            Filename=Filename,
            Image=Image,
            PassBand=PassBand,
        )
    )

def ComputeDCTSpectrum(*, Filename: str = None, Image: np.ndarray = None) -> np.ndarray:
    """
    ComputeDCTSpectrum

    This function computes and returns the Discrete Cosine Spectrum coefficients
    of a given image (file or pixel array), to be used for spectral processing.

    Filename:
        The full file path to open and read the input pixel data from
    Image:
        The original pixel data to process

    Return (np.ndarray):
        The resulting 2D array of cosine spectrum coefficients computed from the
        provided image.
    """

    if ( Filename is None ) and ( Image is None ):
        raise ValueError(f"Either Filename or Image must be provided.")

    if ( Filename is not None ) and ( Filename != "" ):
        Image = cv2.imread(Filename, cv2.IMREAD_GRAYSCALE)

    NewSize: typing.Tuple[int, int] = Image.shape[0:2]
    if ( NewSize[0] % 2 != 0 ):
        NewSize = (NewSize[0]+1, NewSize[1])
    if ( NewSize[1] % 2 != 0 ):
        NewSize = (NewSize[0], NewSize[1]+1)

    if ( any([((n % 2) != 0) for n in Image.shape[0:2]])):
        NewImage = np.zeros(NewSize, dtype=Image.dtype)
        NewImage[0:Image.shape[0], 0:Image.shape[1]] = Image
        Image = NewImage

    Image = BGRToGreyscale(Image)
    Image = GammaCorrection(Image=Image.copy().astype(np.float64), Minimum=0, Maximum=1)

    Spectrum: np.ndarray = cv2.dct(Image)

    return Spectrum

def ComputeFilteredDCTSpectrum(*, Filename: str = None, Image: np.ndarray = None, PassBand: typing.Tuple[float, float] = (0, 1.0)) -> np.ndarray:
    """
    ComputeFilteredDCTSpectrum

    Like ComputeDCTSpectrum(), but also applies a BandPass filter over the spectral
    coefficients to remove high and/or low frequency components from the spectrum.

    Filename:
        The full file path to open and read the input pixel data from
    Image:
        The original pixel data to filter
    PassBand:
        A tuple of values on the range [0, 1], indicating the cut-in and cut-off
        of the passband. These are the values provided to LowPassFilterImage()
        and HighPassFilterImage() respectively.

    Return (np.ndarray):
        The resulting 2D array of cosine spectrum coefficients computed from the
        provided image, filtered based off the provided PassBand values.
    """

    Spectrum = ComputeDCTSpectrum(Filename=Filename, Image=Image)

    xLow, xHigh = tuple(int(round(Image.shape[1] * n)) for n in PassBand)
    yLow, yHigh = tuple(int(round(Image.shape[0] * n)) for n in PassBand)

    FilteredSpectrum: np.ndarray = np.zeros_like(Spectrum)
    FilteredSpectrum[:yHigh, :xHigh] = Spectrum[:yHigh, :xHigh]
    FilteredSpectrum[:yLow, :xLow] = np.zeros((yLow, xLow))

    return FilteredSpectrum

def ReconstructImageFromSpectrum(*, Spectrum: np.ndarray = None) -> np.ndarray:
    """
    ReconstructImageFromSpectrum:

    This function is the inverse of ComputeDCTSpectrum(), converting the cosine spectrum
    coefficients back into an image.

    Spectrum:
        The 2D cosine spectrum coefficients, as computed by ComputeDCTSpectrum().

    Return (np.ndarray):
        The reconstructed image pixels, built from the spectral coefficients.
    """

    if ( Spectrum is None ):
        raise ValueError(f"Spectrum must be given.")

    Image: np.ndarray = cv2.dct(Spectrum, flags=cv2.DCT_INVERSE)
    Image[Image<0] = 0

    Image = GammaCorrection(Image, Minimum=0, Maximum=255)

    return Image.astype(np.uint8)
