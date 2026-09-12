#!/usr/bin/env python3

#   Author: ...
#   Date:   ...

#   Script Purpose: ...
#                       ...

#   Extending this library from 2D to 3D:
#
#   Within this library, all points are implicitly assumed to be 2D.  A clear
#   future extension to this library involves extending all of this
#   functionality to operate on 3D images and work natively on 3-dimensional
#   image data.
#
#   In order to do this, the primary change to this library is to implement a
#   "Point" class to represent the locations within the image data, with
#   suitable constructor checks to enable proper overloading to handle either 2D
#   or 3D points. Then, all references within the algorithms below must be
#   modified to not work with individual numpy arrays or coordinate tuples, but
#   instead work with these Point class instances. With proper internal checks
#   and validations, all of the algorithms can be made to not require knowledge
#   of the dimensionality of the Points. Beyond that, the module-level constants
#   related to connectivity all need to be adjusted and updated to allow for
#   either 2D or 3D connectivity of the varying types.

#   Import the necessary standard library modules
from __future__ import annotations
import typing

import collections
import hashlib
import heapq
import pickle
import math
import os
import random

#   ...

#   Import the necessary third-part modules
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist, pdist, squareform
import skimage.morphology
#   ...

#   Import the desired locally written modules
from . import Utils
from . import Logger

ENABLE_DEBUGGING_VIEWS: bool = False
# ENABLE_DEBUGGING_VIEWS: bool = True

#   ...

DefaultLogWriter: Logger.Logger = Logger.Logger(AlwaysFlush=True)

Connectivity8Kernel = np.array([
    [1, 1, 1],
    [1, 0, 1],
    [1, 1, 1],
])

Connectivity4Kernel = np.array([
    [0, 1, 0],
    [1, 0, 1],
    [0, 1, 0],
])

ColourSimilarityThreshold: int = 25

ForwardConeWidth: float = np.cos(np.deg2rad(45))

RYGColourMap = plt.colormaps['RdYlGn']

#   DEMONSTRATION ONLY!
DebuggingCanvas: np.ndarray = None

def NumpyCoordinateToTuple(Coordinate: np.ndarray) -> typing.Tuple[int, int]:
    return tuple(int(x) for x in Coordinate[::-1])

def CoordinateTupleToNumpyArray(Coordinate: typing.Tuple[int, int]) -> np.ndarray:
    return np.array([Coordinate[::-1]]).squeeze()

def VectorsAreColinear(u: np.ndarray, v: np.ndarray, Theta: float) -> bool:

    u /= np.linalg.norm(u)
    v /= np.linalg.norm(v)

    return np.inner(u, v) >= Theta

def GetNeighbours(Space: np.ndarray, Location: np.ndarray, ConnectivityKernel: np.ndarray) -> np.ndarray:

    #   Get the indices of the neighbourhood around this point
    Xs: slice = slice(Location[1]-1, Location[1]+2)
    Ys: slice = slice(Location[0]-1, Location[0]+2)

    #   Search the neighbourhood for pixels to connect to, which have not yet been visited
    CandidateSubset: np.ndarray = Space[Ys, Xs] != 0

    #   Find only those pixels which are novel and valid candidates, and translate to the full indices
    Neighbours = np.argwhere((CandidateSubset * ConnectivityKernel) != 0) + (Location - [1, 1])

    return Neighbours

def SaveNeurites(Filepath: str, Neurites: typing.Sequence[Neurite]) -> bool:
    """
    """

    Directory: str = os.path.dirname(Filepath)
    if ( Directory is None ) or ( Directory == "" ):
        Directory = "./"

    if ( not os.path.exists(Directory) ):
        os.makedirs(Directory, 0o755, exist_ok=True)

    if ( not Filepath.endswith(".npy") ):
        Filepath += ".npy"

    with open(Filepath, "wb+") as OutFile:
        pickle.dump(Neurites, OutFile)

    return True

def LoadNeurites(Filepath: str) -> typing.Sequence[Neurite]:
    """
    """

    if ( not os.path.exists(Filepath) ):
        return list()

    with open(Filepath, "rb") as InFile:
        Neurites: typing.List[Neurite] = pickle.load(InFile)

    return Neurites

def IdentifyNeurites(Image: np.ndarray, *, LatticeSize: int = 5, NeighbourhoodRadius: int = 9, ExplantContours: np.ndarray = None) -> typing.Sequence[Neurite]:
    """
    IdentifyNeurites

    This function...

    Image:
        ...
    (Optional):
    LatticeSize:
        ...
    NeighbourhoodRadius:
        ...
    ExplantOrigins:
        ...
    ColinearityThreshold:
        ...

    Return (Sequence[Neurite]):
        ...
    """

    #   DEBUGGING
    global DebuggingCanvas
    DebuggingCanvas = Utils.ConvertTo8Bit(Utils.GreyscaleToBGR(Image.copy()))
    #   DEBUGGING

    #   Perform the initial tracing of pixels to transform them to Neurite objects
    SkeletonizedImage = skimage.morphology.skeletonize(Image.copy())
    InitialNeurites: typing.Sequence[Neurite] = _TraceNeurites(SkeletonizedImage)

    #   Transform from the neurites to the connectivity graph representation
    ConnectivityGraph: NeuriteGraph = NeuriteGraph()
    if ( ENABLE_DEBUGGING_VIEWS ):
        ConnectivityGraph.SetBackgroundImage(DebuggingCanvas)
    ConnectivityGraph.AddNeurites(InitialNeurites)

    #   Apply the default simplification heuristics to the connectivity graph
    ConnectivityGraph.Simplify(LatticeSize, NeighbourhoodRadius)

    #   Now, attempt to reconstruct the neurites from the graph.
    Neurites: typing.Sequence[Neurite] = ConnectivityGraph.ReconstructNeurites(NeighbourhoodSize=NeighbourhoodRadius, ExplantCoreContours=ExplantContours)

    #   Apply some post-facto filtering to these identified neurites in order to
    #   remove "redundant" neurites
    Neurites = _FilterRedundantNeurites(Neurites)
    #   ...

    return list(sorted(Neurites, key=lambda x: NumpyCoordinateToTuple(x.Root()), reverse=True))

def EstimateBranchingRatio(Neurites: typing.Sequence[Neurite]) -> float:
    """
    """

    UniqueRoots: np.ndarray = np.unique(np.array([x.Root() for x in Neurites]))
    UniqueTails: np.ndarray = np.unique(np.array([x.Tail() for x in Neurites]))

    return (len(UniqueTails) / len(UniqueRoots))

def _TraceNeurites(NeuritePixelMask: np.ndarray) -> typing.Sequence[Neurite]:
    """
    _TraceNeurites

    This function...

    NeuritePixelMask:
        ...
    ExplantCoresMask:
        ...
    ExplantCoreCentroids:
        ...

    Return (Sequence[Neurite]):
        ...
    """

    #   Prepare the total list of neurites identified across the stack
    Filaments: typing.Sequence[Neurite] = list()

    HistoryMatrix: np.ndarray = np.full_like(NeuritePixelMask, fill_value=False, dtype=bool)

    #   Now, for each starting point, we want to trace the filament "out" until we have no more pixels to connect to.
    while ( np.count_nonzero(CandidatePoints := (NeuritePixelMask * ~HistoryMatrix)) > 0 ):

        CandidateStarts: np.ndarray = np.argwhere(CandidatePoints != 0)

        #   Pick a random point in the set of candidate neurite pixels to start from...
        StartingPoint: np.ndarray = CandidateStarts[random.randint(0, CandidateStarts.shape[0] - 1)]

        #   Initialize a neurite starting from this point.
        Filament: Neurite = Neurite(Origin=StartingPoint)

        #   Find the set of neighbouring pixel(s) which we may either
        #   connect to this filament, or consider as a branch beginning a
        #   new filament.
        Filament.Extend(NeuritePixelMask, Connectivity=8, History=HistoryMatrix.view())

        # OutputVideo.WriteFrame(Filament.Draw(Utils.GreyscaleToBGR(Utils.GammaCorrection(NeuritePixelMask.copy(), Minimum=0, Maximum=255))), PlaybackMode=vwr.PlaybackMode_NoDelay)
        #   ...

        #   Append this to the set of identified filaments
        Filaments.append(Filament)

    if ( np.count_nonzero(MissingFilaments := NeuritePixelMask & ~HistoryMatrix) > 0 ):
        Utils.DisplayImage("Missed Filaments", Utils.ConvertTo8Bit(MissingFilaments), 0, True, True)

    #   Before returning the set of identified filaments, clean up the "tree",
    #   to remove duplicate or obviously erroneous filaments
    # Background: np.ndarray = Utils.GreyscaleToBGR(Utils.GammaCorrection(NeuritePixelMask.copy(), Minimum=0, Maximum=127))
    # for Filament in Filaments:
    #     Background = Filament.Draw(Background, IncludeChildren=True)

    return Filaments

def _FilterRedundantNeurites(Neurites: typing.Sequence[Neurite]) -> typing.Sequence[Neurite]:
    """
    """

    #   We have a set of Neurites which we've identified from the graph. Now we
    #   need to filter them to remove known "redundancies" like total or significant overlap,
    #   or joining trunks or tails.

    #   Remove any "Neurites" of length 0
    FilteredNeurites = list(filter(lambda x: x.ContourLength() > 0, Neurites))

    #   Remove any shorter neurites which are "sufficiently overlapped by" other neurites...
    FilteredNeurites = _RemoveOverlappingNeurites(FilteredNeurites, OverlapFraction=0.975)

    #   ...

    return FilteredNeurites

def _RemoveOverlappingNeurites(Candidates: typing.Sequence[Neurite], *, OverlapFraction: float = 1.0) -> typing.Sequence[Neurite]:
    """
    """

    FilteredNeurites: typing.Sequence[Neurite] = list()

    #   Sort the list of neurites in descending order of length
    Candidates = list(sorted(Candidates, key=lambda x: x.ContourLength(), reverse=True))
    ContainedIndices: typing.Set[int] = set()

    #   Look at each neurite...
    for (i, Candidate) in enumerate(Candidates, start=1):
        DefaultLogWriter.Write(f"Checking for significantly overlapping Neurites, [ {(len(Candidates) - i) + 1} ] left to check...\r")
        if ( (i - 1) in ContainedIndices ):
            continue

        #   And find all non-overlapping pairwise combinations with the rest of the neurites...
        for j, Test in enumerate(Candidates[i:]):
            if ( (i + j) in ContainedIndices ):
                continue

            #   For any given pair, check if the candidate neurite overlaps the test neurite by at least
            #   desired threshold
            if ( Candidate.CalculateOverlapFraction(Test) >= OverlapFraction ):
                #   If it does, then we know this Candidate is "sufficiently contained by" this
                #   particular Test Neurite and can skip it.

                ContainedIndices.add(i + j)

                # #   DEBUGGING
                # global DebuggingCanvas
                # Canvas: np.ndarray = DebuggingCanvas.copy()
                # Canvas = Candidate.Draw(Canvas, Colour=(0, 255, 0))
                # Canvas = Test.Draw(Canvas, Colour=(0, 0, 255))
                # Utils.DisplayImage(f"Overlap Fraction: {Candidate.CalculateOverlapFraction(Test) * 100:.3f}%", Canvas, 1, True, True)
                # #   DEBUGGING

        FilteredNeurites.append(Candidate)

    DefaultLogWriter.Println(f"Finished removing significantly overlapping Neurites, [ {len(FilteredNeurites)} ] remaining.")
    return FilteredNeurites

class Neurite():
    """
    Neurite

    This class...
    """

    _UID: int
    _Points: np.ndarray
    _Children: typing.List[Neurite]

    _Oriented: bool

    _Connectivity8Kernel: np.ndarray
    _Connectivity4Kernel: np.ndarray

    ##  Magic Methods
    def __init__(self, Origin: np.ndarray = None) -> None:
        """
        Constructor:

        ...
        """

        self._UID = random.randint(1,2**32)
        self._Oriented = False

        if ( Origin is not None ):
            self._Points = np.array([Origin])
        else:
            self._Points = np.empty((0, 2), dtype=np.int64)
        self._Children = list()

        self._Connectivity8Kernel = Connectivity8Kernel.copy()
        self._Connectivity4Kernel = Connectivity4Kernel.copy()

        # DefaultLogWriter.Println(f"{self._UID}: Neurite originating at [ {self._Points[0]} ].")

        return

    def __hash__(self: Neurite) -> int:
        Status: bool = self._Points.flags.writeable
        self._Points.flags.writeable = False
        Value: int = int(hashlib.md5(self._Points.data).hexdigest(), 16)
        self._Points.flags.writeable = Status
        return Value

    #   ...

    ### Static Class Methods
    @staticmethod
    def FromVertices(Vertices: typing.Sequence[typing.Tuple[int, int]] | np.ndarray, *, CoordinateType: int = 0) -> Neurite:
        """
        CoordinateType: 0 - Coordinate Tuples, 1 - Numpy Array
        """

        Points: np.ndarray = None
        if ( CoordinateType == 0 ):
            Points = np.array([CoordinateTupleToNumpyArray(x) for x in Vertices])
        else:
            Points = Vertices.copy()

        N: Neurite = Neurite(None)
        N._Points = Points.copy()

        return N

    ##  Public Methods
    def Root(self: Neurite) -> np.ndarray:
        return self._Points[0]

    def Vertices(self: Neurite) -> np.ndarray:
        return self._Points.copy()

    def Tail(self: Neurite) -> np.ndarray:
        return self._Points[-1]

    def Copy(self: Neurite) -> Neurite:
        return Neurite.FromVertices(self.Vertices(), CoordinateType=1)

    def Reverse(self: Neurite) -> Neurite:
        return Neurite.FromVertices(np.array(list(reversed(self.Vertices()))), CoordinateType=1)

    def StartsWith(self: Neurite, Other: Neurite) -> bool:
        """
        """

        for i, (a, b) in enumerate(zip(self.Vertices(), Other.Vertices())):
            if ( np.any(a != b) ):
                return (i > 1)

        return True

    def CalculateOverlapFraction(self: Neurite, Other: Neurite) -> float:
        """
        """

        if ( Other.ContourLength() == 0 ):
            return 0.0

        #   Get the vertices of the two filaments
        TheseVertices, ThoseVertices = self.Vertices(), Other.Vertices()

        #   Calculate the pairwise distances between the two filaments...
        Distances: np.ndarray = cdist(TheseVertices, ThoseVertices)

        #   If there are no overlapping points, return 0 overlap as a short-cut
        if ( not np.any(Distances == 0) ):
            return 0.0

        #   If there are overlaps, they must be continuous due to how the Neurites are built via A*.
        #   Thus, find the first and last overlapping indices of the shorter filament and compute
        #   the contour length along this sub-section.
        OverlapStartIndex: int = np.min(np.argwhere(Distances == 0)[:,0])
        OverlapEndIndex:   int = np.max(np.argwhere(Distances == 0)[:,0])

        OverlapLength: float = cv2.arcLength(ThoseVertices[OverlapStartIndex:OverlapEndIndex+1], closed=False)

        OverlapFraction: float = OverlapLength / Other.ContourLength()

        # #   DEBUGGING
        # global DebuggingCanvas
        # Canvas: np.ndarray = DebuggingCanvas.copy()
        # Canvas = self.Draw(Canvas, Colour=(0, 255, 0))
        # Canvas = Other.Draw(Canvas, Colour=(0, 0, 255))
        # Utils.DisplayImage(f"Overlap Fraction: {OverlapFraction * 100:.3f}%", Canvas, 1, True, True)
        # #   DEBUGGING

        return OverlapFraction

    def EndsWith(self: Neurite, Other: Neurite) -> bool:
        """
        """

        for i, (a, b) in enumerate(zip(reversed(self.Vertices()), reversed(Other.Vertices()))):
            if ( np.any(a != b) ):
                return (i > 1)

        return True

    def Join(self: Neurite, Other: Neurite) -> Neurite:
        """
        Join

        Adds the given branch to the end of the caller.
        """
        StartIndex: int = 0
        if ( np.all(self.Tail() == Other.Root()) ):
            StartIndex = 1

        return Neurite.FromVertices(np.append(self._Points.copy(), Other._Points[StartIndex:].copy(), axis=0), CoordinateType=1)

    def Split(self: Neurite, Indices: typing.Sequence[int] | int) -> typing.Sequence[Neurite]:

        # #   If there's only one split index, and it's either the first or last index of the neurite,
        # #   we can short-circuit and just return itself.
        # if ( len(Indices) == 1 ):
        #     if ( Indices[0] == 0 ) or ( Indices[0] == (len(self._Points)-1) ):
        #         return [Neurite.FromVertices(self.Vertices(), CoordinateType=1)]

        Vertices: np.ndarray = self.Vertices()
        Segments: typing.List[Neurite] = list()
        Start: int = 0
        for End in Indices:
            Segment: Neurite = Neurite.FromVertices(Vertices[Start:End+1].copy(), CoordinateType=1)
            if ( Segment.ContourLength() > 0 ):
                Segments.append(Segment)
                Start = End+1

        Segment: Neurite = Neurite.FromVertices(Vertices[Start:].copy(), CoordinateType=1)
        if ( Segment.ContourLength() > 0 ):
            Segments.append(Segment)

        return Segments

    def Root(self: Neurite) -> np.ndarray:
        return self._Points[0]

    def Tail(self: Neurite) -> np.ndarray:
        return self._Points[-1]

    def TreeSize(self: Neurite) -> int:
        return 1 + sum([x.TreeSize() for x in self._Children])

    def Extend(self: Neurite, CandidatePixels: np.ndarray, *, Connectivity: int = 4, History: np.ndarray = None) -> Neurite:
        """
        Extend

        This function...

        CandidatePixels:
            ...
        Connectivity:
            ...
        History:
            ...
        Vertices:
            ...

        Return (self):
            ...
        """

        if ( History is None ):
            History = np.full_like(CandidatePixels, fill_value=0, dtype=np.uint8)

        NeuritesToExtend: typing.Deque[Neurite] = collections.deque()
        NeuritesToExtend.append(self)

        CurrentNeurite: Neurite = None
        while ( len(NeuritesToExtend) > 0 ):

            #   Get the current neurite to work with
            if ( CurrentNeurite is None ):
                CurrentNeurite: Neurite = NeuritesToExtend.popleft()

            #   Identify where the current "end" of this neurite is
            Here = CurrentNeurite._Points[-1]

            #   If this point has already been visited, this neurite is finished and we can stop processing it further
            if ( History[tuple(Here)] ):
                CurrentNeurite = None
                continue

            #   DEBUGGING
            # if ( random.randint(0, 50) >= 0 ):
            #     CurrentNeurite.Draw(Utils.GreyscaleToBGR(Utils.GammaCorrection((CandidatePixels & ~History).astype(np.uint8), Minimum=0, Maximum=255)), IncludeChildren=False)
            #   DEBUGGING

            #   Mark the current location as visited, to prevent cycling back to this point again
            History[tuple(Here)] = True

            #   Get the indices of the neighbourhood around this point
            Xs: slice = slice(Here[1]-1, Here[1]+2, 1)
            Ys: slice = slice(Here[0]-1, Here[0]+2, 1)

            #   Prepare the connectivity-checking kernel
            ConnectivityKernel: np.ndarray = CurrentNeurite._Connectivity8Kernel.copy() if Connectivity == 8 else CurrentNeurite._Connectivity4Kernel.copy()

            #   Handle the literal edge cases of the slices extending past the four edges of the image...
            KernelCentre: typing.List[int] = [1, 1]
            if ( Xs.start < 0 ) and ( Ys.start < 0 ):
                Xs = slice(0, 2, 1)
                Ys = slice(0, 2, 1)
                ConnectivityKernel = ConnectivityKernel[1:, 1:]
                KernelCentre = [0, 0]
            elif ( Xs.start < 0 ):
                Xs = slice(0, 2, 1)
                ConnectivityKernel = ConnectivityKernel[:, 1:]
                KernelCentre = [1, 0]
            elif ( Ys.start < 0 ):
                Ys = slice(0, 2, 1)
                ConnectivityKernel = ConnectivityKernel[1:, :]
                KernelCentre = [0, 1]

            if ( Xs.stop > CandidatePixels.shape[1] ) and ( Ys.stop > CandidatePixels.shape[0] ):
                Xs = slice(CandidatePixels.shape[1]-2,CandidatePixels.shape[1], 1)
                Ys = slice(CandidatePixels.shape[0]-2,CandidatePixels.shape[0], 1)
                ConnectivityKernel = ConnectivityKernel[:-1, :-1]
            elif ( Xs.stop > CandidatePixels.shape[1] ):
                Xs = slice(CandidatePixels.shape[1]-2,CandidatePixels.shape[1], 1)
                ConnectivityKernel = ConnectivityKernel[:, :-1]
            elif ( Ys.stop > CandidatePixels.shape[0] ):
                Ys = slice(CandidatePixels.shape[0]-2,CandidatePixels.shape[0], 1)
                ConnectivityKernel = ConnectivityKernel[:-1, :]

            #   Search the neighbourhood for pixels to connect to, which have not yet been visited
            CandidateSubset: np.ndarray = CandidatePixels[Ys, Xs] != 0
            HistorySubset: np.ndarray = (History[Ys, Xs] == 0)

            #   Find only those pixels which are novel and valid candidates, and translate to the full indices
            Neighbours = np.argwhere((CandidateSubset * ConnectivityKernel * HistorySubset) != 0) + (Here - KernelCentre)

            #   Find the angle between each neighbour and the current end-to-end vector of the neurite. We want to prioritize neighbours which minimize the turning or curving of the neurite.
            NeighbourDirections: np.ndarray = np.array([np.inner(CurrentNeurite.EndToEndVector(Normalized=True), x - CurrentNeurite._Points[-1]) for x in Neighbours])
            Neighbours = Neighbours[np.argsort(-NeighbourDirections)[:np.count_nonzero(np.isfinite(NeighbourDirections))]]

            if ( len(Neighbours) == 0 ):
                # DefaultLogWriter.Println(f"{CurrentNeurite._UID}: Neurite started at [ {CurrentNeurite._Points[0]} ] terminating at [ {Here} ]. Tree Size={CurrentNeurite.TreeSize()}")
                pass
            else:
                for (Index, Neighbour) in enumerate(Neighbours):
                    if ( Index == 0 ):
                        CurrentNeurite._Points = np.append(CurrentNeurite._Points, [Neighbour], axis=0)
                        NeuritesToExtend.appendleft(CurrentNeurite)
                    else:
                        # DefaultLogWriter.Println(f"{CurrentNeurite._UID}: Neurite branching at [ {Here} ]. Tree Size={CurrentNeurite.TreeSize()}")
                        Child: Neurite = Neurite(Origin=Here)
                        Child._Points = np.append(Child._Points, [Neighbour], axis=0)
                        CurrentNeurite._Children.append(Child)
                        NeuritesToExtend.append(Child)

            CurrentNeurite = None

            if ( random.randint(0, 99) == 0 ):
                TotalPixels: int = np.count_nonzero(CandidatePixels)
                VisitedPixels: int = np.count_nonzero(History)
                DefaultLogWriter.Write(f"Neurite Tracing [ {VisitedPixels / TotalPixels * 100:3.3f}% ]...\r")

        # self.Draw(Utils.GreyscaleToBGR(Utils.GammaCorrection(CandidatePixels.copy(), Minimum=0, Maximum=127)), IncludeChildren=True)

        return self

    def Orientation(self: Neurite, EndToEnd: bool = False) -> np.ndarray:

        #   This function returns one of two representations of the orientation
        #   of a given Neurite.
        #
        #   If EndToEnd is true, this simply constructs the end-to-end vector
        #   for the Neurite, and returns the four-quadrant arctangent of the
        #   resulting vector, measured in radians.
        #
        #   If EndToEnd is false, this returns a richer distribution
        #   representing how the orientation varies over the contour length of
        #   the neurite. This returns an Nx2 array containing the following
        #   information:
        #
        #   [0] - The four-quadrant arctangent of the piecewise linear segment of the neurite, measured in radians
        #   [1] - A weighting factor, representing how much of the neurite points in this direction.
        #
        #   This weighting is necessary in order to distinguish between two
        #   neurites, pointing largely in directions A and B respectively, but
        #   where they "kink" near the end to point in B and A for a short
        #   segment. The weighting factors allow easy distinguishing of these
        #   two neurites, whereas just reporting the raw angles would not
        #   accurately reflect the physical arrangement.

        if ( EndToEnd ):
            return np.array([[float(np.arctan2(*self.EndToEndVector(Normalized=True))), self.ContourLength()]])

        #   Prepare the mapping between orientation angle and the associated weighting
        OrientationMap: typing.Dict[float, float] = {}

        for (From, To) in zip(self._Points, self._Points[1:]):

            Angle: float = np.arctan2(*(To - From))
            Weight: float = np.linalg.norm(To - From)

            if ( OrientationMap.get(Angle) is None ):
                OrientationMap[Angle] = Weight
            else:
                OrientationMap[Angle] += Weight

        Stats: np.ndarray = np.array([(k, OrientationMap[k]) for k in OrientationMap.keys()])
        return Stats

    def ContourLength(self: Neurite) -> float:
        if ( len(self._Points) <= 1 ):
            return 0.0
        return cv2.arcLength(self._Points, closed=False)

    def EstimateThickness(self: Neurite, ThicknessMask: np.ndarray) -> np.ndarray:
        """
        """

        ThicknessMapping: np.ndarray = np.array([])

        #   For each segment of the filament...
        for (Start, End) in zip(self.Vertices(), self.Vertices()[1:]):

            Midpoint = ((Start + End) / 2).astype(np.int_)

            #   Find the nearest non-zero point in the distance mapping to this midpoint
            nonZero = cv2.findNonZero(ThicknessMask).squeeze()
            Distances = cdist([Midpoint], nonZero).flatten()
            Closest = nonZero[np.argmin(Distances)]

            if ( np.linalg.norm(Closest - Midpoint) > ThicknessMask[NumpyCoordinateToTuple(Closest)]):
                ThicknessMapping = np.append(ThicknessMapping, 1)
            else:
                ThicknessMapping = np.append(ThicknessMapping, ThicknessMask[NumpyCoordinateToTuple(Closest)])

        return ThicknessMapping

    def EndToEndLength(self: Neurite) -> float:

        Q: np.ndarray = self.EndToEndVector()

        return np.sqrt(np.inner(Q, Q))

    def EndToEndVector(self: Neurite, *, Normalized: bool = False) -> np.ndarray:

        Vector: np.ndarray = None
        if ( len(self._Points) < 2 ):
            return np.array([0,0])
        else:
            Vector = self._Points[-1] - self._Points[0]
            if ( Normalized ):
                Vector = Vector.astype(np.float64) / np.linalg.norm(Vector)

        return Vector

    def TipVector(self: Neurite, *, Normalized: bool = False) -> np.ndarray:

        Vector: np.ndarray = None
        if ( len(self._Points) < 2 ):
            return np.array([0,0])
        else:
            Vector = self._Points[-1] - self._Points[-2]
            if ( Normalized ):
                Vector = Vector.astype(np.float64) / np.linalg.norm(Vector)

        return Vector

    def RootVector(self: Neurite, *, Normalized: bool = False) -> np.ndarray:

        Vector: np.ndarray = None
        if ( len(self._Points) < 2 ):
            return np.array([0,0])
        else:
            Vector = self._Points[1] - self._Points[0]
            if ( Normalized ):
                Vector = Vector.astype(np.float64) / np.linalg.norm(Vector)

        return Vector

    def Draw(self: Neurite, Image: np.ndarray, *, Colour: int | typing.Tuple[int, int, int] = None, IncludeChildren: bool = True, FilamentThickness: np.ndarray = None) -> np.ndarray:
        """
        Draw

        This function...

        Image:
            ...

        Return (np.ndarray):
            ...
        """

        if ( Colour is None ):
            Colour = tuple(random.sample(range(64, 255, 1), 3))
            while ( np.std(Colour) < ColourSimilarityThreshold ):
                Colour = tuple(random.sample(range(64, 255, 1), 3))
            if ( len(Image.shape) == 2 ):
                Colour = (Colour[0],)

        if ( len(Image.shape) == 3 ) and ( len(Colour) == 1):
            Colour = tuple(random.sample(range(64, 255, 1), 3))
            while ( np.std(Colour) < ColourSimilarityThreshold ):
                Colour = tuple(random.sample(range(64, 255, 1), 3))

        for i, (From, To) in enumerate(zip(self._Points, self._Points[1:])):
            SegmentLength: float = np.linalg.norm(To - From)
            if ( FilamentThickness is None ):
                Image = cv2.arrowedLine(Image.view(), NumpyCoordinateToTuple(From), NumpyCoordinateToTuple(To), Colour, 1, line_type=cv2.LINE_8, tipLength=5.0 / SegmentLength)
            else:
                NormalizedThickness = FilamentThickness[i] / FilamentThickness[-1]
                Colour = tuple(int(x * 255) for x in (RYGColourMap(NormalizedThickness))[:3][::-1])
                Colour = tuple([int(x * (255 / max(Colour))) for x in Colour])
                # Image = cv2.arrowedLine(Image.view(), NumpyCoordinateToTuple(From), NumpyCoordinateToTuple(To), Colour, 1, line_type=cv2.LINE_8, tipLength=5.0 / SegmentLength)
                Image = cv2.line(Image.view(), NumpyCoordinateToTuple(From), NumpyCoordinateToTuple(To), Colour, FilamentThickness[i], lineType=cv2.LINE_8)

        if ( IncludeChildren ):
            for Child in self._Children:
                Image = Child.Draw(Image.view(), Colour=Colour)

        # Utils.DisplayImage(f"", Image, 0, True, True, UpdateWindows=True)
        return Image

    ##  Private Methods

class NeuriteGraph():
    """
    NeuriteGraph

    This class...
    """

    _Edges: typing.Set[typing.Tuple[GraphNode, GraphNode]]
    _Nodes: typing.Dict[typing.Tuple[int, int], GraphNode]

    _EdgeForm: bool
    _NodeForm: bool

    _BackgroundImage: np.ndarray

    def __init__(self: NeuriteGraph, NodeForm: bool = False) -> None:
        """
        Constructor

        This function...

        NodeForm:
            ...

        Return (None):
            ...
        """

        self._Edges = set()
        self._Nodes = {}

        self._EdgeForm = (not NodeForm)
        self._NodeForm = NodeForm

        self._BackgroundImage = None

        return

    def __len__(self: NeuriteGraph) -> int:
        if ( self._EdgeForm ):
            return len(self._Edges)
        else:
            return len(self._Nodes)

    ### Public Methods
    def SetBackgroundImage(self: NeuriteGraph, BackgroundImage: np.ndarray) -> NeuriteGraph:
        self._BackgroundImage = BackgroundImage.copy()
        return self

    def Draw(self: NeuriteGraph, *, Background: np.ndarray = None) -> np.ndarray:

        Canvas: np.ndarray = None
        MaximumX, MaximumY = 0, 0

        if ( Background is not None ):
            MaximumX, MaximumY = Background.shape[::-1][:2]
        elif ( self._BackgroundImage is not None ):
            MaximumX, MaximumY = self._BackgroundImage.shape[::-1][:2]
        else:
            for (From, To) in self._Edges:
                MaximumX, MaximumY = max(MaximumX, From.Coordinates[0], To.Coordinates[0]), max(MaximumY, From.Coordinates[1], To.Coordinates[1])
                MaximumX = math.ceil(MaximumX / 100) * 100
                MaximumY = math.ceil(MaximumY / 100) * 100

        Extent: int = max(MaximumX, MaximumY)

        if ( Extent == 0 ):
            Extent = 500

        if ( Background is not None ):
            Canvas = Background.copy()
        elif ( self._BackgroundImage is not None ):
            Canvas = self._BackgroundImage.copy()
        else:
            Canvas = np.zeros((Extent, Extent, 3), dtype=np.uint8)

        if ( self._EdgeForm ):
            for (From, To) in self._Edges:
                self._DrawConnection(Canvas, From, To)
        else:
            for Node in self._Nodes.values():
                for From in Node._IncomingConnections:
                    self._DrawConnection(Canvas, From, Node)
                for To in Node._OutgoingConnections:
                    self._DrawConnection(Canvas, Node, To)

        return Canvas

    def AddNeurites(self: NeuriteGraph, Neurites: typing.Sequence[Neurite], *, Simplify: bool = False, LatticeSize: float = 5, NeighbourhoodSize: float = 9) -> NeuriteGraph:
        """
        AddNeurites

        This function...

        Neurites:
            ...

        Return (self):
            ...
        """

        for Neurite in Neurites:
            self.AddNeurite(Neurite, Simplify=Simplify, LatticeSize=LatticeSize, NeighbourhoodSize=NeighbourhoodSize)

        return self

    def AddNeurite(self: NeuriteGraph, Neurite: Neurite, *, Simplify: bool = False, LatticeSize: float = 5, NeighbourhoodSize: float = 9) -> NeuriteGraph:
        """
        AddNeurite

        This function...

        Neurite:
            ...

        Return (self):
            ...
        """

        Previous: GraphNode = None
        for (From, To) in zip(Neurite._Points, Neurite._Points[1:]):

            Now: GraphNode = GraphNode().SetOrigin(From)
            Next: GraphNode = GraphNode().SetOrigin(To)

            self._AddEdge(Previous, Now)
            self._AddEdge(Now, Next)

            Previous = Now

        for Child in Neurite._Children:
            self.AddNeurite(Child, Simplify=Simplify, LatticeSize=LatticeSize, NeighbourhoodSize=NeighbourhoodSize)

        if ( Simplify ):
            self.Simplify(LatticeSize=LatticeSize, NeighbourhoodSize=NeighbourhoodSize)

        return self

    def Simplify(self: NeuriteGraph, LatticeSize: float, NeighbourhoodSize: float) -> NeuriteGraph:

        DefaultLogWriter.Println(f"Neurite Graph has [ {len(self)} ] connections prior to simplification...")

        #   Collapse "nearby" nodes into the same node, losing a small amount of spatial resolution in order
        #   to have a simpler graph
        self.CollapseConnections(LatticeSize=LatticeSize, NeighbourhoodSize=NeighbourhoodSize)
        DefaultLogWriter.Println(f"Neurite Graph has [ {len(self)} ] connections after collapsing spatially nearby nodes...")

        if ( ENABLE_DEBUGGING_VIEWS ):
            Utils.DisplayImage(f"Simplified Neurite Graph...", self.Draw(), 0.5, True, True)

        return self

    def CollapseConnections(self: NeuriteGraph, LatticeSize: float = 3, NeighbourhoodSize: float = 5) -> NeuriteGraph:
        """
        CollapseConnections

        This function...

        Size:
            ...

        Return (self):
            ...
        """

        self._ToEdgeForm()

        self._SnapToLattice(LatticeSize)

        self._CollapseNeighbourhoods(NeighbourhoodSize)

        if ( ENABLE_DEBUGGING_VIEWS ):
            Utils.DisplayImage(f"Collapsing connections...", self.Draw(), 0.1, True, True)

        return self

    def ReconstructNeurites(self: NeuriteGraph, NeighbourhoodSize: float, ExplantCoreContours: np.ndarray = None) -> typing.Sequence[Neurite]:
        """
        """

        global DebuggingCanvas

        def DistanceToNearestExplant(Point: typing.Tuple[int, int], ExplantContours: np.ndarray, *, Index: bool = False) -> float | typing.Tuple[float, int]:
            """
            Compute the distance from the given point to the nearest point lying
            on the boundary of one of the explant contours.
            """

            ContourIndex, Distance = -1, np.inf

            for i, Contour in enumerate(ExplantContours):
                d: float = abs(cv2.pointPolygonTest(Contour, Point, True))
                if ( d < Distance ):
                    Distance = d
                    ContourIndex = i

            if ( Index ):
                return (Distance, ContourIndex)

            return Distance

        def SearchHeuristic(Point: typing.Tuple[int, int], Contour: np.ndarray) -> float:
            #   The search heuristic is the Euclidian distance to the nearest point on the contour
            return abs(cv2.pointPolygonTest(Contour, Point, measureDist=True))

        #   Make the graph undirected to start...
        Edges = self._Edges.copy()
        for (From, To) in Edges:
            self._AddEdge(To, From)

        #   Convert the graph to node form.
        self._ToNodeForm()

        #   Order the nodes by decreasing distance to any point on any explant contour
        Nodes: typing.List[typing.Tuple[float, int, typing.Tuple[int, int]]] = list(sorted(
            [(*DistanceToNearestExplant(x, ExplantCoreContours, Index=True), x) for x in self._Nodes.keys()],
            key=lambda x: DistanceToNearestExplant(x[2], ExplantCoreContours),
            reverse=True
        ))

        #   Keep track of the nodes we've already assigned to a neurite, so they cannot be considered as
        #   valid destination nodes, as well as the set of Neurites we've found
        SeenNodes: typing.Set[typing.Tuple[int, int]] = set()
        Neurites: typing.Sequence[Neurite] = list()

        #   While we still have nodes yet to assign to a Neurite...
        while ( len(Nodes) > 0 ):

            #   Get the destination node we are going to search for, and the explant contour
            #   from which we expect it to originate from.
            _, i, DestinationCoordinates = Nodes.pop(0)
            if ( DestinationCoordinates in SeenNodes ):
                continue

            OriginContour: np.ndarray = ExplantCoreContours[i]

            DefaultLogWriter.Write(f"Reconstructing Neurites: [ {len(Nodes):0d} ] nodes remaining to check...\r")

            #   Apply A* path-finding to find the shortest path from this
            CandidateNeurite: Neurite = self._FindShortestNeurite(DestinationCoordinates, OriginContour, SearchHeuristic, NeighbourhoodSize)
            if ( CandidateNeurite is not None ) and ( CandidateNeurite.ContourLength() > 0 ):
                Neurites.append(CandidateNeurite)
                #   Mark all (but the first) node as seen, preventing re-calculating shortest paths to nodes we've already calculated,
                #   but allowing for finding potentially novel "prefixes".
                SeenNodes.update([NumpyCoordinateToTuple(x) for x in CandidateNeurite.Vertices()[1:]])

        return Neurites

    ### Private Methods
    def _ToEdgeForm(self: NeuriteGraph) -> NeuriteGraph:
        """
        """

        if ( self._EdgeForm ):
            return self

        for Node in self._Nodes.values():
            for Outgoing in Node._OutgoingConnections:
                self._AddEdge(Node, Outgoing)
            for Incoming in Node._IncomingConnections:
                self._AddEdge(Incoming, Node)

        self._EdgeForm = True

        self._NodeForm = False
        self._Nodes = dict()

        return self

    def _ToNodeForm(self: NeuriteGraph) -> NeuriteGraph:
        """
        """

        if ( self._NodeForm ):
            return self

        for (From, To) in self._Edges:

            Node: GraphNode = self._Nodes.get(From.Coordinates, None)
            if ( Node is None ):
                From._OutgoingConnections.add(To)
                self._Nodes[From.Coordinates] = From
            else:
                Node._OutgoingConnections.add(To)

            Node = self._Nodes.get(To.Coordinates, None)
            if ( Node is None ):
                To._IncomingConnections.add(From)
                self._Nodes[To.Coordinates] = To
            else:
                Node._IncomingConnections.add(From)

        self._NodeForm = True

        self._EdgeForm = False
        self._Edges = set()

        return self

    def _AddEdge(self: NeuriteGraph, From: GraphNode, To: GraphNode) -> NeuriteGraph:
        """
        _AddEdge

        This function...

        From:
            ...
        To:
            ...

        Return (NeuriteGraph):
            ...
        """

        if ( From is not None ) and ( To is not None ) and ( From != To ):
            if ( self._EdgeForm ):
                self._Edges.add((From, To))
            else:
                F: GraphNode = self._Nodes.get(From.Coordinates, None)
                if ( F is None ):
                    From._OutgoingConnections.add(To)
                    self._Nodes[From.Coordinates] = From
                else:
                    F._OutgoingConnections.update(From._OutgoingConnections)
                    F._IncomingConnections.update(From._IncomingConnections)
                    F._OutgoingConnections.add(To)

                T: GraphNode = self._Nodes.get(To.Coordinates, None)
                if ( T is None ):
                    To._IncomingConnections.add(From)
                    self._Nodes[To.Coordinates] = To
                else:
                    T._OutgoingConnections.update(To._OutgoingConnections)
                    T._IncomingConnections.update(To._IncomingConnections)
                    T._IncomingConnections.add(From)

        return self

    def _RemoveEdge(self: NeuriteGraph, From: GraphNode, To: GraphNode) -> NeuriteGraph:
        """
        """

        if ( From is not None ) and ( To is not None ) and ( From != To ):
            if ( self._EdgeForm ):
                self._Edges.discard((From, To))
            else:
                F: GraphNode = self._Nodes.get(From.Coordinates, None)
                if ( F is not None ):
                    F._OutgoingConnections.discard(To)
                    if ( len(F._OutgoingConnections) == 0 ) and ( len(F._IncomingConnections) == 0 ):
                        self._Nodes.pop(From.Coordinates, None)

                T: GraphNode = self._Nodes.get(To.Coordinates, None)
                if ( T is not None ):
                    T._IncomingConnections.discard(From)
                    if ( len(T._OutgoingConnections) == 0 ) and ( len(T._IncomingConnections) == 0 ):
                        self._Nodes.pop(To.Coordinates, None)

        return self

    def _DrawConnection(self: NeuriteGraph, Canvas: np.ndarray, From: GraphNode, To: GraphNode) -> np.ndarray:
        """
        _DrawConnection

        This function...

        From:
            ...
        To:
            ...

        Return (np.ndarray):
            ...
        """

        Epsilon: float = 1e-5

        #   Draw a circle at the location of each node.
        Canvas = cv2.circle(Canvas, From.Coordinates, 2, (255, 255, 255), -1)
        Canvas = cv2.circle(Canvas, To.Coordinates, 2, (255, 255, 255), -1)

        Distance: float = DistanceBetween(From, To)
        if ( Distance >= Epsilon ):
            cv2.arrowedLine(Canvas, From.Coordinates, To.Coordinates, color=(255, 255, 255), thickness=1, tipLength=5.0 / Distance)

        return Canvas

    def _CollapseNeighbourhoods(self: NeuriteGraph, Size: float) -> NeuriteGraph:
        """
        """

        Neighbourhoods: np.ndarray = self._IdentifyNeighbourhoods(Size=Size)

        DefaultLogWriter.Println(f"Created a total of [ {len(Neighbourhoods)} ] neighbourhoods...")
        if ( ENABLE_DEBUGGING_VIEWS ):
            I: np.ndarray = self._BackgroundImage.copy()
            for n in Neighbourhoods:
                I = cv2.circle(I, NumpyCoordinateToTuple(n), 3, (255, 255, 255), -1)
            Utils.DisplayImage(f"Creating Neighbourhoods", I, 2, True, True)

        #   Now, with the set of neighbourhoods identified, we just need to map each connection to the nearest one.
        #   Iterate over the set of connections we are working with, and search for the nearest neighbourhood to each
        #   end of the connection. We replace the original connection with one linking these two neighbourhoods,
        #   so long as they are distinct.
        CollapsedConnections: typing.Set[typing.Tuple[GraphNode, GraphNode]] = set()
        for Index, (From, To) in enumerate(self._Edges):

            DefaultLogWriter.Write(f"Collapsing Edges [ {((Index+1) / len(self._Edges)) * 100:3.3f}% ] - [ {len(CollapsedConnections)} ] remaining...\r")

            Source, Destination = CoordinateTupleToNumpyArray(From.Coordinates), CoordinateTupleToNumpyArray(To.Coordinates)
            SourceDistances = cdist(np.reshape(Source, (1,2)), Neighbourhoods)[0]
            DestinationDistances = cdist(np.reshape(Destination, (1,2)), Neighbourhoods)[0]

            SourceNeighbourhood = Neighbourhoods[np.argmin(SourceDistances)]
            DestinationNeighbourhood = Neighbourhoods[np.argmin(DestinationDistances)]

            if ( np.linalg.norm(SourceNeighbourhood - DestinationNeighbourhood) <= 1e-3 ):
                continue

            CollapsedConnections.add((GraphNode().SetOrigin(SourceNeighbourhood), GraphNode().SetOrigin(DestinationNeighbourhood)))

            #   DEBUGGING
            if ( ENABLE_DEBUGGING_VIEWS ):
                Temp: NeuriteGraph = NeuriteGraph()
                Temp.SetBackgroundImage(self._BackgroundImage)
                Temp._Edges = CollapsedConnections
                Utils.DisplayImage(f"Collapsing connections...", Temp.Draw(), 0.001, True, True, UpdateWindows=True)
            #   DEBUGGING

        DefaultLogWriter.Println(f"Finished collapsing edges of the graph - [ {len(CollapsedConnections)} ] remaining.")
        self._Edges = CollapsedConnections.copy()
        if ( ENABLE_DEBUGGING_VIEWS ):
            Utils.DisplayImage(f"Collapsing connections...", self.Draw(), 0.001, True, True)

        return self

    def _IdentifyNeighbourhoods(self: NeuriteGraph, Size: float) -> typing.Sequence[GraphNode]:
        """
        """

        #   Get the location of all the distinct nodes in the graph.
        Nodes: typing.Set[GraphNode] = set()
        for Edge in self._Edges:
            Nodes.update(Edge)
        DefaultLogWriter.Println(f"Graph has [ {len(Nodes)} ] distinct nodes...")

        Locations: np.ndarray = np.empty((len(Nodes), 2), dtype=np.uint16)
        for Index, Node in enumerate(Nodes):
            Locations[Index,:] = CoordinateTupleToNumpyArray(Node.Coordinates)
        Locations = Locations[np.lexsort((Locations[:,1], Locations[:,0]))]

        #   Let's take advantage of the known locality of nodes when identifying neighbourhoods.
        #   We only need to check pairs of nodes which are both within a given region. We can
        #   find all such nodes in O(n), and then do the pairwise comparisons on this much smaller
        #   subset!
        BlockSize: int = 100
        Neighbourhoods: np.ndarray = np.empty((0, 2))
        SubNeighbourhoods: np.ndarray = np.empty((0, 2))
        for y in range(math.ceil(np.max(Locations[:,0]) / BlockSize)):
            for x in range(math.ceil(np.max(Locations[:,1]) / BlockSize)):
                Left, Right = max((x * BlockSize) - Size, 0), (((x+1) * BlockSize) + Size)
                Top, Bottom = max((y * BlockSize) - Size, 0), (((y+1) * BlockSize) + Size)

                DefaultLogWriter.Write(f"Identifying neighbourhoods in the region [ ({Top},{Left})-({Bottom},{Right}) ]...\r")

                CandidateMask: np.ndarray = (Top <= Locations[:,0]) & (Locations[:,0] <= Bottom) & (Left <= Locations[:,1]) & (Locations[:,1] <= Right)

                CandidateLocations: np.ndarray = Locations[CandidateMask]
                if ( len(CandidateLocations) == 0 ):
                    continue

                Distances: np.ndarray = squareform(pdist(CandidateLocations))
                Distances[np.diag_indices(len(CandidateLocations))] = np.inf

                #   Iterate over the rows of the distance matrix...
                for Index in range(Distances.shape[0]):

                    #   If this node is marked to not be included, skip it
                    if ( np.all(np.isnan(Distances[Index, :])) ):
                        continue

                    #   Identify which pairs are within the neighbourhood size of this node...
                    NearbyIndices: np.ndarray = np.argwhere(Distances[Index, :] <= Size)

                    #   For each of these, we need to mark these to not be included in later iterations over the distance matrix
                    for NearbyIndex in NearbyIndices:
                        Distances[:, NearbyIndex] = np.nan
                        Distances[NearbyIndex, :] = np.nan

                    SubNeighbourhoods = np.append(SubNeighbourhoods, np.reshape(CandidateLocations[Index], (1,2)), axis=0)

            Neighbourhoods = np.append(Neighbourhoods, SubNeighbourhoods, axis=0)
            SubNeighbourhoods = np.empty((0, 2))

        return Neighbourhoods

    def _SnapToLattice(self: NeuriteGraph, Size: float) -> NeuriteGraph:
        """
        """

        if ( Size == 1.0 ):
            return self

        CollapsedConnections: typing.Set[typing.Tuple[GraphNode, GraphNode]] = set()
        EdgeCount: int = len(self._Edges)
        for Index, (From, To) in enumerate(self._Edges):
            DefaultLogWriter.Write(f"Snapping connections to lattice [ {(( Index + 1) / EdgeCount ) * 100:.3f}% ] - [ {len(CollapsedConnections)} ]...\r")
            LatticeFrom, LatticeTo = From.SnapToLattice(int(Size)), To.SnapToLattice(int(Size))
            if ( LatticeFrom != LatticeTo ):
                CollapsedConnections.add((LatticeFrom, LatticeTo))

                #   DEBUGGING
                if ( ENABLE_DEBUGGING_VIEWS ):
                    Temp: NeuriteGraph = NeuriteGraph().SetBackgroundImage(self._BackgroundImage)
                    Temp._Edges = CollapsedConnections
                    Utils.DisplayImage(f"Collapsing connections...", Temp.Draw(), 0.001, True, True, UpdateWindows=True)
                #   DEBUGGING

        DefaultLogWriter.Println(f"[ {len(CollapsedConnections)} ] connections remaining after snapping to lattice.")
        self._Edges = CollapsedConnections.copy()
        if ( ENABLE_DEBUGGING_VIEWS ):
            Utils.DisplayImage(f"Collapsing connections...", self.Draw(), 1, True, True)
        return self

    def _FindNearestSourceNode(self: NeuriteGraph, Origins: np.ndarray, ReferenceDirection: bool = False) -> GraphNode | typing.Tuple[GraphNode, np.ndarray]:

        # DefaultLogWriter.Println(f"Searching for the closest source node to one of the provided centroids: {[f'{x}, ' for x in Origins]} ")

        Nodes: typing.List[GraphNode] = list()

        if ( self._EdgeForm ):
            #   Identify all of the nodes which are the start of a connection...
            Sources: typing.Set[GraphNode] = set([x[0] for x in self._Edges])
            #   ...and all of those which are the end of a connection
            Destinations: typing.Set[GraphNode] = set([x[1] for x in self._Edges])

            #   Take only those which are sources, i.e. not the termination of a connection
            Nodes = list(Sources.difference(Destinations))

            if ( len(Nodes) == 0 ):
                if ( len(self.self._Edges) == 0 ):
                    if ( ReferenceDirection ):
                        return (None, None)
                    return None
                Nodes = list(Sources)
        else:
            Nodes = [x for x in self._Nodes.values() if ((len(x._IncomingConnections) == 0) and (len(x._OutgoingConnections) > 0))]
            if ( len(Nodes) == 0 ):
                if ( len(self._Nodes) == 0 ):
                    if ( ReferenceDirection ):
                        return (None, None)
                    return None
                Nodes = list(self._Nodes.values())

        #   Extract just the coordinates as a numpy array
        NodeCoordinates: np.ndarray = np.array([CoordinateTupleToNumpyArray(x.Coordinates) for x in Nodes])

        #   Compute the distances of all the source nodes to each of the origins
        Distances: np.ndarray = cdist(NodeCoordinates, Origins)

        #   For each node, take only the single minimum distance, setting the other possible values to infinity so as to be ignored.
        Distances[Distances > np.min(Distances, axis=1)[:, None]] = np.inf

        #   If we somehow have a node equidistant to multiple origins, pick one arbitrarily
        for Row in Distances:
            Row[np.argmin(Row)+1:] = np.inf

        #   Save the indices for which origin each node was closest to
        OriginIndices: np.ndarray = np.argmax(np.isfinite(Distances), axis=1)

        #   Remove the unnecessary infinities now
        Distances = Distances[np.isfinite(Distances)]

        #   Indirectly sort the distances by their closest centroid
        SortIndices: np.ndarray = np.argsort(Distances)

        #   We want to start by searching for nodes with just one outgoing connection...
        DesiredConnectionCount: int = 1
        SourceNode: GraphNode = None
        Direction: np.ndarray = None
        while ( SourceNode is None ):

            if ( DesiredConnectionCount > 10 ):
                DefaultLogWriter.Warnln(f"Failed to find nearby node with only outgoing connections...")
                DefaultLogWriter.Println(f"The closest node to a centroid is located at [ {Node.Coordinates} ]...")
                SourceNode = Nodes[SortIndices[0]]
                Direction = ConnectionToVector(GraphNode().SetOrigin(Origins[OriginIndices[SortIndices[0]]]), SourceNode, UnitVector=True)
            else:
                #   Iterate over these sorted indices, looking for the first one which has the desired number of outgoing connections
                for SortIndex in SortIndices:
                    Node = Nodes[SortIndex]

                    OutgoingConnectionCount: int = 0
                    if ( self._EdgeForm ):
                        OutgoingConnectionCount = sum([
                            1 for (x, y) in self._Edges if (x == Node) and (y != Node)
                        ])
                    else:
                        OutgoingConnectionCount = len(Node._OutgoingConnections)

                    if ( OutgoingConnectionCount == DesiredConnectionCount ):
                        # DefaultLogWriter.Println(f"The closest node to a centroid is located at [ {Node.Coordinates} ] and has [ {OutgoingConnectionCount} ] outgoing connections...")
                        SourceNode = Node
                        Direction = ConnectionToVector(GraphNode().SetOrigin(Origins[OriginIndices[SortIndex]]), SourceNode, UnitVector=True)
                        break
                else:
                    DesiredConnectionCount += 1

        if ( ReferenceDirection ):
            return (SourceNode, Direction)

        return SourceNode

    def _RemoveCycles(self: NeuriteGraph) -> NeuriteGraph:
        """
        """

        #   MUST ONLY BE CALLED WITIN self.OrientConnections_3

        DefaultLogWriter.Println(f"Asserting no cycles are found within the graph...")
        TotalConnections: int = len(self._Edges)
        for Index, (From, To) in enumerate(self._Edges.copy()):
            if ( (To, From) in self._Edges ):
                DefaultLogWriter.Write(f"Checking and removing possible cycles [ {((Index+1)/TotalConnections) * 100:.3f}% ]...\r")
                self._Edges.discard((To, From))

        return

    def _OrientSubtrees(self: NeuriteGraph, NodesToProcess: typing.Deque[typing.Tuple[GraphNode, np.ndarray]]) -> typing.Set[typing.Tuple[GraphNode, GraphNode]]:
        """
        """

        self._ToNodeForm()

        OrientedConnections: typing.Set[typing.Tuple[GraphNode, GraphNode]] = set()

        while ( len(NodesToProcess) > 0 ):
            DefaultLogWriter.Write(f"Propagating orientation information throughout sub-trees. [ {len(self._Nodes)} ] left to check...\r")
            CurrentNode, ReferenceDirection = NodesToProcess.popleft()

            #   Identify all connections which terminate at this node...
            IncomingConnections: typing.List[typing.Tuple[GraphNode, GraphNode]] = list([(x, CurrentNode) for x in CurrentNode._IncomingConnections])

            #   Identify all connections which originate at this node...
            OutgoingConnections: typing.List[typing.Tuple[GraphNode, GraphNode]] = list([(CurrentNode, x) for x in CurrentNode._OutgoingConnections])

            if ( len(IncomingConnections) + len(OutgoingConnections) == 0 ):
                continue

            OrientedConnections.update(self._OrientVertex(NodesToProcess, CurrentNode, ReferenceDirection, IncomingConnections, OutgoingConnections))

            #   DEBUGGING
            if ( ENABLE_DEBUGGING_VIEWS ):
                T: NeuriteGraph = NeuriteGraph().SetBackgroundImage(self._BackgroundImage)
                T._Edges = OrientedConnections.copy()
                Utils.DisplayImages([(f"Partial", T.Draw()), ("Original", self.Draw())], 0.01, True, True, UpdateWindows=True)
            #   DEBUGGING

        return OrientedConnections

    def _OrientVertex(self: NeuriteGraph, NodesToProcess: typing.Deque[typing.Tuple[GraphNode, np.ndarray]], Vertex: GraphNode, ReferenceDirection: np.ndarray, IncomingConnections: typing.Set[typing.Tuple[GraphNode, GraphNode]], OutgoingConnections: typing.Set[typing.Tuple[GraphNode, GraphNode]]) -> typing.Set[typing.Tuple[GraphNode, GraphNode]]:
        """
        """

        if ( ENABLE_DEBUGGING_VIEWS ):
            VisualizeVertexConnections(self, Vertex, ReferenceDirection, IncomingConnections, OutgoingConnections)

        OrientedConnections: typing.Set[typing.Tuple[GraphNode, GraphNode]] = set()

        #   We want to treat all connections on the same footing, so we need to "flip" the incoming
        #   connections ahead of time, so that the "from" it always the vertex. We keep track of
        #   whether these are incoming or outgoing in the computation of the inner products
        Connections: typing.List[typing.Tuple[GraphNode, np.ndarray]] = [(self._Nodes[x.Coordinates], ConnectionToVector(Vertex, self._Nodes[x.Coordinates], UnitVector=True)) for (_, x) in OutgoingConnections]
        Connections.extend([(self._Nodes[x.Coordinates], ConnectionToVector(self._Nodes[x.Coordinates], Vertex, UnitVector=True)) for (x, _) in IncomingConnections])

        if ( len(Connections) >= 2 ):

            #   Compute the inner products with the reference direction
            ReferenceInnerProducts: np.ndarray = np.array([np.inner(ReferenceDirection, x) for (_, x) in Connections])

            #   Compute the pairwise inner products of each connection
            PairwiseInnerProducts: np.ndarray = np.array([[np.inner(x, y) for (_, x) in Connections] for (_, y) in Connections])

            #   Remove the lower-triangular elements, since these are duplicates of the upper-triangular indices
            PairwiseInnerProducts[np.tril_indices_from(PairwiseInnerProducts)] = 0

            #   Are there any pairs of vectors with inner products which are close to being anti-parallel, i.e. diverging?
            AntiParallelConnectionsMask: np.ndarray = (-PairwiseInnerProducts >= ForwardConeWidth)
            AntiParallelConnectionsIndices: np.ndarray = np.argwhere(AntiParallelConnectionsMask)

            #   Iterate over all pairs of diverging vectors
            for i, j in AntiParallelConnectionsIndices:
                NodePair = [Connections[i][0], Connections[j][0]]
                DirectionPair = [Connections[i][1], Connections[j][1]]

                #   Which of these two is pointing closest to the reference direction?
                CorrectedDirection = DirectionPair[np.argmax(np.array([ReferenceInnerProducts[i], ReferenceInnerProducts[j]]))]

                for Index, (Node, Direction) in enumerate(zip(NodePair, DirectionPair)):

                    Orientation: float = np.inner(CorrectedDirection, ConnectionToVector(Vertex, Node, UnitVector=True))
                    if ( Orientation >= 0 ):
                        # DefaultLogWriter.Println(f"(2) Adding connection [ {Vertex.Coordinates}->{Node.Coordinates} ] - ({CorrectedDirection})...")
                        OrientedConnections.add((Vertex, Node))
                    else:
                        # DefaultLogWriter.Println(f"(2) Adding connection [ {Node.Coordinates}->{Vertex.Coordinates} ] - ({CorrectedDirection})...")
                        OrientedConnections.add((Node, Vertex))
                    NodesToProcess.append((Node, CorrectedDirection))
                    self._RemoveEdge(Vertex, Node)
                    self._RemoveEdge(Node, Vertex)

            for Index in sorted(np.unique(AntiParallelConnectionsIndices.flatten()), reverse=True):
                Connections.pop(Index)

            if ( len(Connections) == 0 ):
                return OrientedConnections

        #   Compute the inner products with the reference direction
        ReferenceInnerProducts: np.ndarray = np.array([np.inner(ReferenceDirection, x) for (_, x) in Connections])

        #   Identify any definitely forward or definitely reversed connections
        ReferenceInnerProductsForwardMask: np.ndarray = ReferenceInnerProducts >= ForwardConeWidth
        ReferenceInnerProductsReverseMask: np.ndarray = -ReferenceInnerProducts >= ForwardConeWidth

        #   Add the correctly oriented "forward" connections
        for Index in np.argwhere(ReferenceInnerProductsForwardMask).flatten():
            Node, Direction = Connections[Index]
            #   Assert the connection we're adding is correctly oriented
            Orientation: float = np.inner(Direction, ConnectionToVector(Vertex, Node, UnitVector=True))
            if ( np.isclose(Orientation, 1) ):
                # DefaultLogWriter.Println(f"(1) Adding connection [ {Vertex.Coordinates}->{Node.Coordinates} ] - ({Direction})...")
                OrientedConnections.add((Vertex, Node))
            elif ( np.isclose(Orientation, -1)):
                # DefaultLogWriter.Println(f"(1) Adding connection [ {Node.Coordinates}->{Vertex.Coordinates} ] - ({Direction})...")
                OrientedConnections.add((Node, Vertex))
            else:
                # DefaultLogWriter.Errorln(f"(1) Unexpected forward orientation!")
                pass
            NodesToProcess.append((Node, Direction))
            self._RemoveEdge(Vertex, Node)
            self._RemoveEdge(Node, Vertex)

        #   Add the correctly oriented "reversed" connections
        for Index in np.argwhere(ReferenceInnerProductsReverseMask).flatten():
            Node, Direction = Connections[Index]
            #   Assert the connection we're adding is correctly oriented
            Orientation: float = np.inner(Direction, ConnectionToVector(Vertex, Node, UnitVector=True))
            if ( np.isclose(Orientation, 1) ):
                # DefaultLogWriter.Println(f"(1) Adding connection [ {Node.Coordinates}->{Vertex.Coordinates} ] - ({-Direction})...")
                OrientedConnections.add((Node, Vertex))
            elif ( np.isclose(Orientation, -1)):
                # DefaultLogWriter.Println(f"(1) Adding connection [ {Vertex.Coordinates}->{Node.Coordinates} ] - ({-Direction})...")
                OrientedConnections.add((Vertex, Node))
            else:
                # DefaultLogWriter.Errorln(f"(1) Unexpected reverse orientation!")
                pass
            NodesToProcess.append((Node, -Direction))
            self._RemoveEdge(Vertex, Node)
            self._RemoveEdge(Node, Vertex)

        #   Remove these now-addressed connections.
        for Index in reversed(sorted(np.argwhere(np.abs(ReferenceInnerProducts) >= ForwardConeWidth).flatten())):
            Connections.pop(Index)

        #   Short-cut exit if all connections have been addressed
        if ( len(Connections) == 0 ):
            return OrientedConnections

        ReferenceInnerProducts: np.ndarray = np.array([np.inner(ReferenceDirection, x) for (_, x) in Connections])

        #   If there are any remaining connections, they are not in the forward cone of the reference direction,
        #   and are not diverging. These may be single connections which are approximately perpendicular to the reference direction,
        #   so we can't do much about them. Assert they point in at least the positive reference direction, and append them to the right side
        #   of the deque, in case we can process this junction better, later.
        for Index, (Node, Direction) in enumerate(Connections):
            if ( ReferenceInnerProducts[Index] >= 0 ):
                Orientation: float = np.inner(Direction, ConnectionToVector(Vertex, Node, UnitVector=True))
                if ( np.isclose(Orientation, 1) ):
                    # DefaultLogWriter.Println(f"(3) Adding connection [ {Vertex.Coordinates}->{Node.Coordinates} ] - ({Direction})...")
                    OrientedConnections.add((Vertex, Node))
                elif ( np.isclose(Orientation, -1)):
                    # DefaultLogWriter.Println(f"(3) Adding connection [ {Node.Coordinates}->{Vertex.Coordinates} ] - ({Direction})...")
                    OrientedConnections.add((Node, Vertex))
                else:
                    # DefaultLogWriter.Errorln(f"(3) Unexpected forward orientation!")
                    pass
                NodesToProcess.append((Node, Direction))
            else:
                Orientation: float = np.inner(Direction, ConnectionToVector(Vertex, Node, UnitVector=True))
                if ( np.isclose(Orientation, 1) ):
                    # DefaultLogWriter.Println(f"(3) Adding connection [ {Node.Coordinates}->{Vertex.Coordinates} ] - ({-Direction})...")
                    OrientedConnections.add((Node, Vertex))
                elif ( np.isclose(Orientation, -1)):
                    # DefaultLogWriter.Println(f"(3) Adding connection [ {Vertex.Coordinates}->{Node.Coordinates} ] - ({-Direction})...")
                    OrientedConnections.add((Vertex, Node))
                else:
                    # DefaultLogWriter.Errorln(f"(3) Unexpected reverse orientation!")
                    pass
                NodesToProcess.append((Node, -Direction))

            self._RemoveEdge(Vertex, Node)
            self._RemoveEdge(Node, Vertex)

        return OrientedConnections

    def _TraceNeuriteSegment(self: NeuriteGraph, Root: GraphNode) -> typing.Sequence[GraphNode]:
        """
        _TraceNeuriteSegment

        This function...
        """

        ColinearityThreshold: float = np.cos(np.deg2rad(30))
        Segment: typing.List[GraphNode] = [Root]

        #   Follow the outgoing connections from this node, only allowing some
        #   pre-determined maximum angle change per node. End when either a
        #   branch is found, or the filament curves "too much".
        CurrentNode: GraphNode = Root
        while ( len(CurrentNode._OutgoingConnections) >= 1 ):
            Candidates: typing.List[GraphNode] = list(CurrentNode._OutgoingConnections)
            #   If we don't have a previous node, then just pick the next node arbitrarily
            if ( CurrentNode == Root ):
                CurrentNode = Candidates[0]
                Segment.append(CurrentNode)
                continue
            #   If we're past the root, and have hit a branch-point, break
            elif ( len(Candidates) > 1 ):
                break

            #   Determine the direction the segment is currently pointing...
            CurrentDirection: np.ndarray = ConnectionToVector(Segment[-2], Segment[-1], UnitVector=True)

            #   Determine the possible directions it may continue in.
            NextDirection: np.ndarray = ConnectionToVector(CurrentNode, Candidates[0], UnitVector=True)

            #   Determine if the next node is sufficiently co-linear with the previous nodes.
            if ( np.inner(CurrentDirection, NextDirection) < ColinearityThreshold ):
                break

            CurrentNode = Candidates[0]
            Segment.append(CurrentNode)

        return Segment

    def _RemoveNeuriteSegment(self: NeuriteGraph, Segment: typing.Sequence[GraphNode]) -> None:
        """
        _RemoveNeuriteSegment

        This function...
        """

        for (From, To) in zip(Segment, Segment[1:]):
            self._RemoveEdge(From, To)

        return

    def _FindShortestNeurite(self: NeuriteGraph, Start: typing.Tuple[int, int], EndContour: np.ndarray, SearchHeuristic: typing.Callable[[typing.Tuple[int, int], np.ndarray], float], NeighbourhoodSize: float) -> Neurite:

        global DebuggingCanvas

        def ReconstructPath(Node: typing.Tuple[int, int], Hierarchy: typing.Dict[typing.Tuple[int, int], typing.Tuple[int, int]]) -> typing.Sequence[typing.Tuple[int, int]]:

            Path: typing.Sequence[typing.Tuple[int, int]] = [Node]
            while ( Node in Hierarchy.keys() ):
                Node = Hierarchy[Node]
                Path.append(Node)
            return Path

        #   Prepare the priority queue used to search nodes of the graph
        NodesToProcess: typing.Sequence[typing.Tuple[float, typing.Tuple[int, int]]] = list()

        #   Prepare a map to track the shortest path through the graph for any
        #   intermediate node.
        From: typing.Dict[typing.Tuple[int, int], typing.Tuple[int, int]] = dict()

        #   Keep Track of the G-Scores to reach each node. Take care to assert
        #   that any missing key is assigned a G-Score if infinity.
        G: typing.Dict[typing.Tuple[int, int], float] = dict()
        G[Start] = 0

        #   Keep Track of the F-Scores assigned to each node. This is the G-score (true cost)
        #   plus the heuristic cost for each node.
        F: typing.Dict[typing.Tuple[int, int], float] = dict()
        F[Start] = G[Start] + SearchHeuristic(Start, EndContour)

        #   Push the starting node into the priority queue.
        heapq.heappush(NodesToProcess, (F[Start], Start))

        #   While there are nodes yet to process...
        CurrentCoordinates: typing.Tuple[int, int] = None
        while ( len(NodesToProcess) > 0 ):

            #   Pop off the highest priority node coordinates, i.e. the one with
            #   the least F score of the nodes under consideration
            _, CurrentCoordinates = heapq.heappop(NodesToProcess)
            CurrentNode: GraphNode = self._Nodes[CurrentCoordinates]

            #   If this point lies sufficiently close to the desired contour, we're done
            if ( cv2.pointPolygonTest(EndContour, CurrentCoordinates, measureDist=True) >= (-NeighbourhoodSize) ):
                return Neurite.FromVertices(ReconstructPath(CurrentCoordinates, From))

            #   Look at all of the "neighbours" to this node
            for NextNode in self._Nodes[CurrentCoordinates]._OutgoingConnections:

                #   Calculate what the G score would be if we moved to this node
                Tentative_G: float = G[CurrentCoordinates] + (DistanceBetween(CurrentNode, NextNode))

                #   Make sure this is better than what we've already found for getting to this node
                if ( Tentative_G < G.get(NextNode.Coordinates, float('inf')) ):

                    #   It's the best path to get to this node
                    From[NextNode.Coordinates] = CurrentCoordinates
                    G[NextNode.Coordinates] = Tentative_G
                    F[NextNode.Coordinates] = Tentative_G + SearchHeuristic(NextNode.Coordinates, EndContour)
                    heapq.heappush(NodesToProcess, (F[NextNode.Coordinates], NextNode.Coordinates))

                    # #   DEBUGGING
                    # #   Visualize the path being taken.
                    # Canvas: np.ndarray = DebuggingCanvas.copy()
                    # Canvas = cv2.drawContours(Canvas, [EndContour], 0, (127, 255, 0), 2)

                    # #   Draw the set of nodes currently under consideration
                    # for (_, Node) in NodesToProcess:
                    #     Canvas = cv2.circle(Canvas, Node, 3, (255, 64, 0), 2)

                    # #   Draw all of the nodes which have been processed, colour-mapped based on their F-scores.
                    # MaxScore: float = np.max(np.array(list(G.values())))
                    # ColourMap = plt.colormaps['RdYlGn']
                    # if ( MaxScore > 0 ):
                    #     for Node in From.values():
                    #         Score: float = G[Node] / MaxScore
                    #         Colour = tuple(int(x * 255) for x in (ColourMap(Score))[:3][::-1])
                    #         Canvas = cv2.circle(Canvas, Node, 3, Colour, -1)

                    # #   ...
                    # #   Finally, display the starting and ending nodes
                    # Canvas = cv2.circle(Canvas, Start, 3, (127, 0, 255), -1)

                    # #   Display the image
                    # Utils.DisplayImage(f"Current A* Progress", Canvas, 0.001, True, True, UpdateWindows=True)
                    # #   DEBUGGING

        #   If the final point doesn't lie on the desired contour, select the point with the least Hueristic-score as the final node
        if ( abs(cv2.pointPolygonTest(EndContour, CurrentCoordinates, measureDist=True)) > NeighbourhoodSize ):
            H: typing.Dict[typing.Tuple[int, int], float] = {x: SearchHeuristic(x, EndContour) for x in F.keys()}
            CurrentCoordinates = min(H, key=H.get)

        return Neurite.FromVertices(ReconstructPath(CurrentCoordinates, From))

class GraphNode():
    """
    GraphNode

    This class...
    """

    Coordinates: typing.Tuple[int, int]

    _IncludeConnections: bool
    _IncomingConnections: typing.Set[GraphNode]
    _OutgoingConnections: typing.Set[GraphNode]

    ### Magic Methods
    def __init__(self: GraphNode, *, TrackConnections: bool = False) -> None:
        """
        Constructor

        This function...

        Return (None):
            ...
        """

        self.Coordinates = ()

        self._IncludeConnections = TrackConnections
        self._IncomingConnections = set()
        self._OutgoingConnections = set()

        return

    def __eq__(self: GraphNode, Other: GraphNode) -> bool:
        if ( self is None ) and ( Other is not None ):
            return False
        elif ( self is not None ) and ( Other is None ):
            return False
        else:
            return np.all(self.Coordinates == Other.Coordinates)

    def __lt__(self: GraphNode, Other: GraphNode) -> bool:
        if ( self is None ) and ( Other is not None ):
            return False
        elif ( self is not None ) and ( Other is None ):
            return False
        else:
            return self.Coordinates < Other.Coordinates

    def __hash__(self: GraphNode) -> int:
        if ( self is None ):
            return hash(None)
        return hash(self.Coordinates)

    def __str__(self: GraphNode) -> str:
        return f"{self.Coordinates}"

    ### Public Methods
    def SetOrigin(self: GraphNode, Origin: np.ndarray) -> GraphNode:
        """
        SetOrigin

        This function..

        Origin:
            ...

        Return (self):
            ...
        """

        self.Coordinates = NumpyCoordinateToTuple(Origin)

        return self

    def SnapToLattice(self: GraphNode, LatticeSpacing: int) -> GraphNode:

        LatticeNode: GraphNode = GraphNode()
        LatticeNode.Coordinates = tuple([round(x / LatticeSpacing) * LatticeSpacing for x in self.Coordinates])
        return LatticeNode

def ConnectionToVector(From: GraphNode, To: GraphNode, *, UnitVector: bool = False) -> np.ndarray:
    """
    ConnectionToVector

    This function...

    From:
        ...
    To:
        ...
    Return (np.ndarray):
        ...
    """

    Vector: np.ndarray = CoordinateTupleToNumpyArray(To.Coordinates) - CoordinateTupleToNumpyArray(From.Coordinates)
    if ( UnitVector ):
        if (( Norm := np.linalg.norm(Vector)) == 0 ):
            Vector = np.array([0, 0], dtype=np.float64)
        else:
            Vector = Vector.astype(np.float64) / Norm

    return Vector

def DistanceBetween(A: GraphNode, B: GraphNode) -> float:
    """
    DistanceBetween

    This function...

    A:
        ...
    B:
        ...

    Return (float):
        ...
    """

    Distance: float = np.linalg.norm(ConnectionToVector(B, A))
    return Distance

def CombineNeuriteOrientationStats(OrientationStats: typing.List[np.ndarray]) -> np.ndarray:
    """
    """

    Epsilon: float = 1e-3

    Angles: np.ndarray = np.array([])
    Weights: np.ndarray = np.array([])

    for Stat in OrientationStats:
        if ( len(Stat) > 0 ):
            Angles = np.append(Angles, Stat[:,0])
            Weights = np.append(Weights, Stat[:,1])

    #   Sort these based on the Angles array
    SortedIndices: np.ndarray = np.argsort(Angles)

    Angles = Angles[SortedIndices]
    Weights = Weights[SortedIndices]

    #   Now, combine values which are "close enough" together into the same group
    FinalAngles: np.ndarray = np.array([])
    FinalWeights: np.ndarray = np.array([])

    while ( len(Angles) > 0 ):
        Mask: np.ndarray = np.abs(Angles - Angles[0]) < Epsilon
        FinalAngles = np.append(FinalAngles, np.mean(Angles[Mask]))
        FinalWeights = np.append(FinalWeights, np.sum(Weights[Mask]))

        Angles = Angles[~Mask]
        Weights = Weights[~Mask]

    return np.array([FinalAngles, FinalWeights])

def VisualizeVertexConnections(Graph: NeuriteGraph, CurrentNode: GraphNode, ReferenceDirection: np.ndarray, IncomingConnections: typing.Set[typing.Tuple[GraphNode, GraphNode]], OutgoingConnections: typing.Set[typing.Tuple[GraphNode, GraphNode]]) -> None:

    NodeSize: int = 2
    #   Visualize this node and its connections...
    I: np.ndarray = Utils.GreyscaleToBGR(Graph._BackgroundImage.copy())

    #   Draw all incoming connections...
    if ( IncomingConnections is not None ):
        for (From, _) in IncomingConnections:
            ConnectionLength: float = np.linalg.norm(ConnectionToVector(From, CurrentNode))
            I = cv2.arrowedLine(I, From.Coordinates, CurrentNode.Coordinates, (0, 255, 255), 1, line_type=cv2.LINE_8, tipLength=5.0 / ConnectionLength)
            I = cv2.circle(I, From.Coordinates, NodeSize, (0, 255, 0), -1)
            DefaultLogWriter.Println(f"Incoming connection from [ {From.Coordinates}->{CurrentNode.Coordinates} ] - ({ConnectionToVector(From, CurrentNode, UnitVector=True)})...")
    else:
        for From in CurrentNode._IncomingConnections:
            ConnectionLength: float = np.linalg.norm(ConnectionToVector(From, CurrentNode))
            I = cv2.arrowedLine(I, From.Coordinates, CurrentNode.Coordinates, (0, 255, 255), 1, line_type=cv2.LINE_8, tipLength=5.0 / ConnectionLength)
            I = cv2.circle(I, From.Coordinates, NodeSize, (0, 255, 0), -1)
            DefaultLogWriter.Println(f"Incoming connection from [ {From.Coordinates}->{CurrentNode.Coordinates} ] - ({ConnectionToVector(From, CurrentNode, UnitVector=True)})...")

    #   Draw all outgoing connections...
    if ( OutgoingConnections is not None ):
        for (_, To) in OutgoingConnections:
            ConnectionLength: float = np.linalg.norm(ConnectionToVector(CurrentNode, To))
            I = cv2.arrowedLine(I, CurrentNode.Coordinates, To.Coordinates, (255, 0, 127), 1, line_type=cv2.LINE_8, tipLength=5.0 / ConnectionLength)
            I = cv2.circle(I, To.Coordinates, NodeSize, (255, 0, 0), -1)
            DefaultLogWriter.Println(f"Outgoing connection to [ {CurrentNode.Coordinates}->{To.Coordinates} ] - ({ConnectionToVector(CurrentNode, To, UnitVector=True)})...")
    else:
        for To in CurrentNode._OutgoingConnections:
            ConnectionLength: float = np.linalg.norm(ConnectionToVector(CurrentNode, To))
            I = cv2.arrowedLine(I, CurrentNode.Coordinates, To.Coordinates, (255, 0, 127), 1, line_type=cv2.LINE_8, tipLength=5.0 / ConnectionLength)
            I = cv2.circle(I, To.Coordinates, NodeSize, (255, 0, 0), -1)
            DefaultLogWriter.Println(f"Outgoing connection to [ {CurrentNode.Coordinates}->{To.Coordinates} ] - ({ConnectionToVector(CurrentNode, To, UnitVector=True)})...")

    #   Draw the current node
    I = cv2.circle(I, CurrentNode.Coordinates, NodeSize, (255, 255, 255), -1)
    #   Draw the reference vector...
    if ( ReferenceDirection is not None ):
        ReferenceDirectionStartingpoint: typing.Tuple[int, int] = NumpyCoordinateToTuple(CoordinateTupleToNumpyArray(CurrentNode.Coordinates) - (15 * ReferenceDirection))
        I = cv2.arrowedLine(I, ReferenceDirectionStartingpoint, CurrentNode.Coordinates, (255, 255, 255), 1, line_type=cv2.LINE_8, tipLength=1/4)
        DefaultLogWriter.Println(f"Current Reference Direction: ({ReferenceDirection})...")

    #   Display the resulting image
    Utils.DisplayImage(f"Current Node being reoriented...", I, 0.01, True, True, UpdateWindows=True)

    return
