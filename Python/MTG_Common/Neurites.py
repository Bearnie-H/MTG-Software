#!/usr/bin/env python3

#   Author: ...
#   Date:   ...

#   Script Purpose: ...
#                       ...

#   Import the necessary standard library modules
from __future__ import annotations
import typing

import collections
import itertools
import math
import random

#   ...

#   Import the necessary third-part modules
import numpy as np
import cv2
#   ...

#   Import the desired locally written modules
from . import Utils
from . import Logger

from . import VideoReadWriter as vwr

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

#   DEMONSTRATION ONLY!
OutputVideo: vwr.VideoReadWriter = vwr.VideoReadWriter()

def NumpyCoordinateToTuple(Coordinate: np.ndarray) -> typing.Tuple[int, int]:
    return tuple(int(x) for x in Coordinate[::-1])

def CoordinateTupleToNumpyArray(Coordinate: typing.Tuple[int, int]) -> np.ndarray:
    return np.array([Coordinate[::-1]]).squeeze()

def GetNeighbours(Space: np.ndarray, Location: np.ndarray, ConnectivityKernel: np.ndarray) -> np.ndarray:

    #   Get the indices of the neighbourhood around this point
    Xs: slice = slice(Location[1]-1, Location[1]+2)
    Ys: slice = slice(Location[0]-1, Location[0]+2)

    #   Search the neighbourhood for pixels to connect to, which have not yet been visited
    CandidateSubset: np.ndarray = Space[Ys, Xs] != 0

    #   Find only those pixels which are novel and valid candidates, and translate to the full indices
    Neighbours = np.argwhere((CandidateSubset * ConnectivityKernel) != 0) + (Location - [1, 1])

    return Neighbours

class Neurite():
    """
    Neurite

    This class...
    """

    _UID: int
    _Points: np.ndarray
    _Children: typing.List[Neurite]

    _Connectivity8Kernel: np.ndarray
    _Connectivity4Kernel: np.ndarray

    #   This contains a list of piece-wise linear end-points, where the neurite tracing neither turns nor branches.
    #   Connecting each segment in series will faithfully reconstruct the path of the neurite through the image.
    _Segments: np.ndarray

    #   When a branch occurs, this array contains the ordered list of points on the trunk neurite from which
    #   the branch occurs.
    _BranchPoints: np.ndarray

    ##  Magic Methods
    def __init__(self, Origin: np.ndarray = None) -> None:
        """
        Constructor:

        ...
        """

        self._UID = random.randint(1,2**32)
        self._BranchPoints = np.empty((0, 2), np.int64)

        if ( Origin is not None ):
            self._Points = np.array([Origin])
        else:
            self._Points = np.empty((0, 2), dtype=np.int64)
        self._Children = list()

        self._Connectivity8Kernel = Connectivity8Kernel.copy()
        self._Connectivity4Kernel = Connectivity4Kernel.copy()

        # DefaultLogWriter.Println(f"{self._UID}: Neurite originating at [ {self._Points[0]} ].")

        return

    #   ...

    ### Static Class Methods
    @staticmethod
    def FromVertices(Vertices: typing.Sequence[typing.Tuple[int, int]]) -> Neurite:
        """
        """

        Points: np.ndarray = np.array([CoordinateTupleToNumpyArray(x) for x in Vertices])
        N: Neurite = Neurite(Points[0])
        N._Points = Points.copy()

        N._Segments = np.empty((0, 2, 2), dtype=np.int64)
        for (Start, End) in zip(Points, Points[1:]):
            N._Segments = np.append(N._Segments, np.array([[Start, End]]), axis=0)

        return N

    ##  Public Methods
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
                        if ( Index == 1 ):
                            CurrentNeurite._BranchPoints = np.append(CurrentNeurite._BranchPoints, [Here], axis=0)
                        NeuritesToExtend.append(Child)

            CurrentNeurite = None

            if ( random.randint(0, 99) == 0 ):
                TotalPixels: int = np.count_nonzero(CandidatePixels)
                VisitedPixels: int = np.count_nonzero(History)
                DefaultLogWriter.Write(f"Neurite Tracing [ {VisitedPixels / TotalPixels * 100:3.3f}% ]...\r")

        # self.Draw(Utils.GreyscaleToBGR(Utils.GammaCorrection(CandidatePixels.copy(), Minimum=0, Maximum=127)), IncludeChildren=True)

        self._PrunePath()
        return self

    def Orientation(self: Neurite, EndToEnd: bool = False) -> float | np.ndarray:

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
            return float(np.arctan2(*self.EndToEndVector(Normalized=True)))

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
        return cv2.arcLength(self._Points, closed=False)

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

    def TipDirection(self: Neurite) -> np.ndarray:

        if ( len(self._Points) <= 1 ):
            return np.array([0, 0])
        else:
            Vector: np.ndarray = self._Points[-1] - self._Points[0]
            return Vector / np.linalg.norm(Vector)

    def Draw(self: Neurite, Image: np.ndarray, *, Colour: int | typing.Tuple[int, int, int] = None, IncludeChildren: bool = True) -> np.ndarray:
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

        for Segment in self._Segments:
            Image = cv2.circle(Image.view(), NumpyCoordinateToTuple(Segment[0]), 3, Colour, -1)
            if ( np.all(Segment[0] == Segment[1]) ):
                continue
            Image = cv2.circle(Image.view(), NumpyCoordinateToTuple(Segment[1]), 3, Colour, -1)
            SegmentLength: float = np.linalg.norm(Segment[1] - Segment[0])
            Image = cv2.arrowedLine(Image.view(), NumpyCoordinateToTuple(Segment[1]), NumpyCoordinateToTuple(Segment[0]), Colour, 1, line_type=cv2.LINE_4, tipLength=5.0 / SegmentLength)

        OutputVideo.WriteFrame(Image)

        if ( IncludeChildren ):
            for Child in self._Children:
                Image = Child.Draw(Image.view(), Colour=Colour)

        Utils.DisplayImage(f"", Image, 0, True, True, UpdateWindows=True)
        return Image

    #   ...

    ##  Private Methods
    def _PrunePath(self: Neurite) -> Neurite:
        """
        _PrunePath

        This function...

        Return (self):
            ...
        """

        StartIndex: int = 0
        EndIndex: int = 1
        BranchIndex: int = 1

        self._Segments: np.ndarray = np.empty((0, 2, 2), dtype=np.int64)

        #   Starting from the beginning, we need to walk the path until we reach the first branch point.
        while ( StartIndex < len(self._Points) ):

            #   If we're not at a branch yet, just increment the counter
            if ( BranchIndex >= self._BranchPoints.shape[0] ):
                EndIndex = len(self._Points)
            elif ( EndIndex < self._Points.shape[0] ) and ( np.any(self._Points[EndIndex] != self._BranchPoints[BranchIndex]) ):
                EndIndex += 1
                continue

            #   We are at a branch, so we need to prune the path we've taken such that
            #   it consists of piecewise linear segments
            ReducedPath: np.ndarray = Utils.PointSequenceToContour(self._Points[StartIndex:EndIndex+1])
            if ( ReducedPath is None ):
                self._Segments = np.append(self._Segments, np.array([self._Points[StartIndex], self._Points[StartIndex]]).reshape(1,2,2), axis=0)
            elif ( len(ReducedPath) == 1 ):
                self._Segments = np.append(self._Segments, np.array([ReducedPath[0], ReducedPath[0]]).reshape(1,2,2), axis=0)
            else:
                for Index, _ in enumerate(ReducedPath[1:]):
                    self._Segments = np.append(self._Segments, np.array([ReducedPath[Index-1], ReducedPath[Index]]).reshape(1, 2, 2), axis=0)

            BranchIndex += 1
            StartIndex = EndIndex
            EndIndex += 1

        for Child in self._Children:
            Child._PrunePath()

        return self

class NeuriteGraph():
    """
    NeuriteGraph

    This class...
    """

    Edges: typing.Set[typing.Tuple[GraphNode, GraphNode]]

    _BackgroundImage: np.ndarray

    def __init__(self: NeuriteGraph) -> None:
        """
        Constructor

        This function...

        Return (None):
            ...
        """

        self.Edges = set()

        self._BackgroundImage = None

        return

    def __len__(self: NeuriteGraph) -> int:
        return len(self.Edges)

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
            for (From, To) in self.Edges:
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

        for (From, To) in self.Edges:
            self._DrawConnection(Canvas, From, To)

        return Canvas

    def AddNeurites(self: NeuriteGraph, Neurites: typing.Sequence[Neurite], *, Simplify: bool = False, LatticeSize: float, NeighbourhoodSize: float, Origins: np.ndarray, Theta: float) -> NeuriteGraph:
        """
        AddNeurites

        This function...

        Neurites:
            ...

        Return (self):
            ...
        """

        for Neurite in Neurites:
            self.AddNeurite(Neurite, Simplify=Simplify, LatticeSize=LatticeSize, NeighbourhoodSize=NeighbourhoodSize, Origins=Origins, Theta=Theta)

        return self

    def AddNeurite(self: NeuriteGraph, Neurite: Neurite, *, Simplify: bool = False, LatticeSize: float, NeighbourhoodSize: float, Origins: np.ndarray, Theta: float) -> NeuriteGraph:
        """
        AddNeurite

        This function...

        Neurite:
            ...

        Return (self):
            ...
        """

        Previous: GraphNode = None
        for Segment in Neurite._Segments:

            Now: GraphNode = GraphNode().SetOrigin(Segment[0])
            Next: GraphNode = GraphNode().SetOrigin(Segment[1])

            self._AddEdge(Previous, Now)
            self._AddEdge(Now, Next)

            Previous = Now

        if ( Simplify ):
            self.Simplify(LatticeSize=LatticeSize, NeighbourhoodSize=NeighbourhoodSize, Origins=Origins, Theta=Theta)

        for Child in Neurite._Children:
            self.AddNeurite(Child, Simplify=Simplify, LatticeSize=LatticeSize, NeighbourhoodSize=NeighbourhoodSize, Origins=Origins, Theta=Theta)

        return self

    def Simplify(self: NeuriteGraph, LatticeSize: float, NeighbourhoodSize: float, Origins: np.ndarray, Theta: float) -> NeuriteGraph:

        DefaultLogWriter.Println(f"Neurite Graph has [ {len(self)} ] connections prior to simplification...")

        #   Collapse "nearby" nodes into the same node, losing a small amount of spatial resolution in order
        #   to have a simpler graph
        self.CollapseConnections(LatticeSize=LatticeSize, NeighbourhoodSize=NeighbourhoodSize)
        DefaultLogWriter.Println(f"Neurite Graph has [ {len(self)} ] connections after collapsing spatially nearby nodes...")

        #   Re-orient the graph connections to be "outward" from the centroid(s)
        self.OrientConnections(Origins=Origins)
        DefaultLogWriter.Println(f"Neurite Graph has [ {len(self)} ] connections after orienting connections...")

        #   Prune any intermediate connections of the graph which neither branch, nor lead to a large enough directional change.
        self.PruneConnections(Epsilon=Theta)
        DefaultLogWriter.Println(f"Neurite Graph has [ {len(self)} ] connections after simplification.")

        Utils.DisplayImage(f"Simplified Neurite Graph...", self.Draw(), 0.1, True, True, UpdateWindows=True)

        return self

    def CollapseConnections(self: NeuriteGraph, LatticeSize: float = 3, NeighbourhoodSize: float = 12) -> NeuriteGraph:
        """
        CollapseConnections

        This function...

        Size:
            ...

        Return (self):
            ...
        """

        self._SnapToLattice(LatticeSize)

        self._CollapseNeighbourhoods(NeighbourhoodSize)

        Utils.DisplayImage(f"Collapsing connections...", self.Draw(), 0.1, True, True)
        return self

    def OrientConnections(self: NeuriteGraph, Origins: np.ndarray = None) -> NeuriteGraph:
        """
        OrientConnections

        This function...

        Origins:
            ...

        Return (self):
            ...
        """

        if ( Origins is None ):
            Origins = np.array([[-1000000, -1000000]])

        #   We want to assert that connections are preferentially directed outward, away from
        #   the cortical explant cores. TODO: describe this more...

        OrientedConnections: typing.Set[typing.Tuple[GraphNode, GraphNode]] = set()

        #   Look at all of the connections in the graph...
        for Index, (From, To) in enumerate(self.Edges):

            DefaultLogWriter.Write(f"Orienting Connections [ {(Index / len(self.Edges)) * 100:03.3f}% ]...\r")

            #   Convert this connection into a vector representation...
            ConnectionVector: np.ndarray = ConnectionToVector(From, To, UnitVector=True)

            #   Identify which centroid origin is closest to the base of this vector
            ClosestCentroid: int = np.argmin(
                np.sum((Origins - CoordinateTupleToNumpyArray(From.Coordinates))**2)
            )

            #   Compute the vector from this centroid to the end of this connection vector
            # RadialExplantVector: np.ndarray = CoordinateTupleToNumpyArray(To.Coordinates) - Origins[ClosestCentroid]
            RadialExplantVector: np.ndarray = ConnectionToVector(GraphNode().SetOrigin(Origins[ClosestCentroid]), To, UnitVector=True)

            #   Is the connection pointing outward?
            if ( np.dot(ConnectionVector, RadialExplantVector) <= 0 ):
                #   No, it's backwards
                OrientedConnections.add((To, From))
            else:
                #   Yes, it's correct
                OrientedConnections.add((From, To))

            # if ( random.randint(0, 500) == 0 ):
            #     IntermediateGraph: NeuriteGraph = NeuriteGraph()
            #     IntermediateGraph.Edges = OrientedConnections.copy()
            #     IntermediateGraph.SetBackgroundImage(self._BackgroundImage)
            #     Utils.DisplayImage(f"Orienting Connections...", IntermediateGraph.Draw(), 0.001, True, True, UpdateWindows=True)

        self.Edges = OrientedConnections
        Utils.DisplayImage(f"Orienting Connections...", self.Draw(), 2, True, True)
        return self

    def PruneConnections(self: NeuriteGraph, Epsilon: float) -> NeuriteGraph:
        """
        PruneConnections

        This function...

        Epsilon:
            ...

        Return (self):
            ...
        """

        Nodes: typing.Set[GraphNode] = set()
        for Edge in self.Edges:
            Nodes.update(Edge)

        NodeCount: int = len(Nodes)
        while ( len(Nodes) > 0 ):

            DefaultLogWriter.Write(f"Pruning Colinear Nodes [ {(len(Nodes) / NodeCount) * 100:03.3f}% remaining ]...\r")

            TestNode: GraphNode = Nodes.pop()

            Sources: typing.Set[GraphNode] = set([Edge[0] for Edge in self.Edges])
            Destinations: typing.Set[GraphNode] = set([Edge[1] for Edge in self.Edges])

            OutgoingConnections: int = sum([1 if TestNode == x else 0 for x in Sources])
            IncomingConnections: int = sum([1 if TestNode == x else 0 for x in Destinations])

            #   If the node has exactly one incoming and outgoing connection, we may be able to prune it
            if ( OutgoingConnections == 1 ) and ( IncomingConnections == 1 ):

                #   Find the specific nodes "on either side" of the test node
                From: GraphNode = [Edge[0] for Edge in self.Edges if Edge[1] == TestNode][0]
                To: GraphNode = [Edge[1] for Edge in self.Edges if Edge[0] == TestNode][0]

                #   Now, check what the angle between the vectors:
                #       From->TestNode
                #       TestNode->To
                #   is, and if it's within the acceptable threshold, then we can remove this middle node and update the corresponding connections!
                IncomingVector: np.ndarray = ConnectionToVector(From, TestNode, UnitVector=True)
                OutgoingVector: np.ndarray = ConnectionToVector(TestNode, To, UnitVector=True)

                #   If the angle between them is sufficiently small, they are "co-linear" enough
                if ( np.arccos(np.dot(IncomingVector, OutgoingVector)) <= Epsilon ):
                    self.Edges.discard((From, TestNode))
                    self.Edges.discard((TestNode, To))
                    self.Edges.add((From, To))
                else:
                    # DefaultLogWriter.Println(f"")
                    pass
            else:
                # DefaultLogWriter.Println(f"")
                pass

            Utils.DisplayImage(f"Pruning Colinear Connections...", self.Draw(), 0.001, True, True, UpdateWindows=True)

        Utils.DisplayImage(f"Pruning Colinear Connections...", self.Draw(), 2, True, True)
        return self

    def PruneConnections_2(self: NeuriteGraph, Epsilon: float) -> NeuriteGraph:
        """
        """



    def ReconstructNeurites(self: NeuriteGraph, ExplantCoreCentroids: np.ndarray = None) -> typing.Sequence[Neurite]:
        """
        ReconstructNeurites

        This function...

        ExplantCoreCentroids:
            ...

        Return (Sequence[Neurite]):
            ...
        """

        if ( ExplantCoreCentroids is None ):
            ExplantCoreCentroids = np.array([[0, 0]])

        Neurites: typing.List = list()
        Connections: typing.Set[typing.Tuple[GraphNode, GraphNode]] = self.Edges.copy()

        #   Starting with the full set of connections within the graph, we want
        #   to select the source node (i.e. one with no incoming connections)
        #   which is closest to one of the centroids. With this node, follow an
        #   outbound path along the connections and subsequent nodes it forms,
        #   constructing a Neurite as the graph is traversed. Continue
        #   traversing until a sink node is reached (i.e. one with no further
        #   outgoing connections). Once a sink is reached and a full Neurite is
        #   constructed, re-examine the Nodes used and prune away any nested
        #   sub-graphs which do not leave orphan nodes. Then repeat the process
        #   until all Nodes of the graph are assigned to at least one Neurite.
        while (( SourceNode := self._FindNearestSourceNode(ExplantCoreCentroids) ) is not None ):

            Vertices: typing.Sequence[GraphNode] = self._TraceFilament(SourceNode)

            self._RemoveFilament(Vertices)

            #   +++ DEBUGGING +++
            Utils.DisplayImage(f"Extracting Neurites...", self.Draw(), 0.1, True, True, UpdateWindows=True)
            #   --- DEBUGGING ---

            Filament: Neurite = Neurite.FromVertices([Vertex.Coordinates for Vertex in Vertices])
            Neurites.append(Filament)

        self.Edges = Connections.copy()
        Utils.DisplayImage(f"Extracting Neurites...", self.Draw(), 10, True, True)
        return Neurites

    ### Private Methods
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
            self.Edges.add((From, To))

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

        #   Walk through the graph once, constructing the locations of the neighbourhoods we will collapse down to
        Neighbourhoods: typing.Set[GraphNode] = set()
        OrderedEdges: typing.List[typing.Tuple[GraphNode, GraphNode]] = list(sorted(self.Edges, key=lambda x: DistanceBetween(x[0], x[1]), reverse=True))
        for Index, Nodes in enumerate(OrderedEdges):
            DefaultLogWriter.Write(f"[ {((Index+1) / len(OrderedEdges))*100:.3f}% ] Creating neighbourhoods around existing nodes - [ {len(Neighbourhoods) } ]...\r")
            for Node in Nodes:
                AddNode: bool = True
                #   Check if there's a neighbourhood "close" to the source node
                for Neighbourhood in Neighbourhoods:
                    if ( DistanceBetween(Node, Neighbourhood) < Size ):
                        AddNode = False
                        break

                if ( AddNode ):
                    Neighbourhoods.add(GraphNode().SetOrigin(CoordinateTupleToNumpyArray(Node.Coordinates)))

                    #   DEBUGGING
                    I: np.ndarray = self._BackgroundImage.copy()
                    for n in Neighbourhoods:
                        I = cv2.circle(I, n.Coordinates, 3, (255, 255, 255), -1)
                    Utils.DisplayImage(f"Creating Neighbourhoods", I, 0.001, True, True, UpdateWindows=True)
                    #   DEBUGGING

        # DefaultLogWriter.Println(f"Finished ")
        I: np.ndarray = self._BackgroundImage.copy()
        for n in Neighbourhoods:
            I = cv2.circle(I, n.Coordinates, 3, (255, 255, 255), -1)
        Utils.DisplayImage(f"Creating Neighbourhoods", I, 2, True, True)

            #   DEBUGGING
            # if ( np.any([DistanceBetween(n1, n2) < Size for (n1, n2) in itertools.product(Neighbourhoods, repeat=2) if n1 != n2]) ):
            #     DefaultLogWriter.Errorln(f"At least two distinct neighbourhoods are too close!")
            #   DEBUGGING

        #   Now, with the set of neighbourhoods identified, we just need to map each connection to the nearest one.
        #   Iterate over the set of connections we are working with, and search for the nearest neighbourhood to each
        #   end of the connection. We replace the original connection with one linking these two neighbourhoods,
        #   so long as they are distinct.
        CollapsedConnections: typing.Set[typing.Tuple[GraphNode, GraphNode]] = set()
        for Index, (From, To) in enumerate(OrderedEdges):

            SourceNeighbourhoodStats = np.array([(DistanceBetween(From, x), x) for x in Neighbourhoods if DistanceBetween(From, x) <= Size])
            SourceNeighbourhood = SourceNeighbourhoodStats[
                np.argmin(SourceNeighbourhoodStats[:,0]),
                1
            ]

            DestinationNeighbourhoodStats = np.array([(DistanceBetween(To, x), x) for x in Neighbourhoods if DistanceBetween(To, x) <= Size])
            DestinationNeighbourhood = DestinationNeighbourhoodStats[
                np.argmin(DestinationNeighbourhoodStats[:,0]),
                1
            ]

            #   Replace this connection with one linking the two neighbourhoods, so long as they're distinct
            if ( SourceNeighbourhood != DestinationNeighbourhood ):
                CollapsedConnections.add((SourceNeighbourhood, DestinationNeighbourhood))

                #   DEBUGGING
                Temp: NeuriteGraph = NeuriteGraph()
                Temp.SetBackgroundImage(self._BackgroundImage)
                Temp.Edges = CollapsedConnections
                Utils.DisplayImage(f"Collapsing connections...", Temp.Draw(), 0.001, True, True, UpdateWindows=True)
                #   DEBUGGING

            DefaultLogWriter.Write(f"Collapsing Edges [ {((Index+1) / len(self.Edges)) * 100:3.3f}% ] - [ {len(CollapsedConnections)} ] remaining...\r")

        DefaultLogWriter.Println(f"Finished collapsing edges of the graph - [ {len(CollapsedConnections)} ] remaining.")
        self.Edges = CollapsedConnections.copy()
        Utils.DisplayImage(f"Collapsing connections...", self.Draw(), 0.001, True, True)
        return self

    def _SnapToLattice(self: NeuriteGraph, Size: float) -> NeuriteGraph:
        """
        """

        CollapsedConnections: typing.Set[typing.Tuple[GraphNode, GraphNode]] = set()
        EdgeCount: int = len(self.Edges)
        for Index, (From, To) in enumerate(self.Edges):
            DefaultLogWriter.Write(f"Snapping connections to lattice [ {(( Index + 1) / EdgeCount ) * 100:.3f}% ] - [ {len(CollapsedConnections)} ]...\r")
            LatticeFrom, LatticeTo = From.SnapToLattice(int(Size)), To.SnapToLattice(int(Size))
            if ( LatticeFrom != LatticeTo ):
                CollapsedConnections.add((LatticeFrom, LatticeTo))

            #   DEBUGGING
            # if ( random.randint(0, len(CollapsedConnections)) == 0 ):
            #     Temp: NeuriteGraph = NeuriteGraph().SetBackgroundImage(self._BackgroundImage)
            #     Temp.Edges = CollapsedConnections
            #     Utils.DisplayImage(f"Collapsing connections...", Temp.Draw(), 0.001, True, True, UpdateWindows=True)
            #   DEBUGGING

        DefaultLogWriter.Println(f"[ {len(CollapsedConnections)} ] connections remaining after snapping to lattice.")
        self.Edges = CollapsedConnections.copy()
        Utils.DisplayImage(f"Collapsing connections...", self.Draw(), 1, True, True)
        return self

    def _FindNearestSourceNode(self: NeuriteGraph, Origins: np.ndarray) -> GraphNode:

        DefaultLogWriter.Println(f"Searching for the closest source node to one of the provided centroids: {[f'{x}, ' for x in Origins]} ")

        Sources: typing.Set[GraphNode] = set([x[0] for x in self.Edges])
        Destinations: typing.Set[GraphNode] = set([x[1] for x in self.Edges])

        Sources -= Destinations

        #   Identify which centroid origin is closest to this node
        SourceNode: GraphNode = None
        MinimumDistance: float = float('inf')
        for Node in Sources:
            Distance: float = np.min(
                [np.linalg.norm(Origin - CoordinateTupleToNumpyArray(Node.Coordinates)) for Origin in Origins]
            )
            if ( Distance < MinimumDistance ):
                MinimumDistance = Distance
                SourceNode = Node

        if ( SourceNode is not None ):
            DefaultLogWriter.Println(f"The closest node to a centroid is located at [ {SourceNode.Coordinates} ]...")

        return SourceNode

    def _TraceFilament(self: NeuriteGraph, Origin: GraphNode) -> typing.Sequence[GraphNode]:

        #   Start the filament at the given Origin node
        Nodes: typing.Sequence[GraphNode] = list()
        CurrentNode: GraphNode = Origin
        Terminated: bool = False

        while ( not Terminated ):
            ConnectionFound: bool = False
            #   Find any outgoing connection from this node...
            for Edge in self.Edges:
                #   If one is found...
                if ( Edge[0] == CurrentNode ):

                    DefaultLogWriter.Println(f"Outgoing connection found: [ {CurrentNode.Coordinates}->{Edge[1].Coordinates} ]")

                    if ( Edge[1] in Nodes ):
                        DefaultLogWriter.Println(f"Potential cycle detected! [ {Edge[1]} ] has already been visited!")
                        continue

                    #   Add the current node to the filament and update which node we're looking for connections from
                    Nodes.append(CurrentNode)
                    CurrentNode = Edge[1]
                    ConnectionFound = True
                    break

            if ( not ConnectionFound ):
                DefaultLogWriter.Println(f"Filament terminates at [ {CurrentNode.Coordinates} ]")
                Nodes.append(CurrentNode)
                Terminated = True

        return Nodes

    def _RemoveFilament(self: NeuriteGraph, Filament: typing.Sequence[GraphNode]) -> None:

        #   Convert the filament from a set of nodes into a set of connections...
        #   Also, reverse this list so that we look at the connections starting
        #   from the known end-point
        FilamentConnections: typing.Sequence[typing.Tuple[GraphNode, GraphNode]] = list(reversed([(x, y) for (x, y) in zip(Filament, Filament[1:])]))

        Source, Destination = FilamentConnections[0]
        DefaultLogWriter.Println(f"Removing terminating connection [ {Source.Coordinates}->{Destination.Coordinates} ]...")
        self.Edges.remove(FilamentConnections[0])

        #   Iterate over the set of connections forming the filament...
        for (Source, Destination) in FilamentConnections[1:]:

            #   If the destination for this segment of the filament appears with
            #   other outgoing connections, we cannot yet prune it and thus we
            #   can immediately return from this function as we can't prune
            #   earlier segments of the filament without potentially leaving
            #   these later nodes orphaned
            Sources: typing.Set[GraphNode] = set([Edge[0] for Edge in self.Edges])

            if ( Destination in Sources ):
                return

            DefaultLogWriter.Println(f"Removing intermediate connection [ {Source.Coordinates}->{Destination.Coordinates} ]...")
            self.Edges.remove((Source, Destination))

        return

class GraphNode():
    """
    GraphNode

    This class...
    """

    Coordinates: typing.Tuple[int, int]

    ### Magic Methods
    def __init__(self: GraphNode) -> None:
        """
        Constructor

        This function...

        Return (None):
            ...
        """

        self.Coordinates = ()
        return

    def __eq__(self: GraphNode, Other: GraphNode) -> bool:
        if ( self is None ) and ( Other is not None ):
            return False
        elif ( self is not None ) and ( Other is None ):
            return False
        else:
            return np.all(self.Coordinates == Other.Coordinates)

    def __hash__(self: GraphNode) -> int:
        if ( self is None ):
            return hash(None)
        return hash(self.Coordinates)

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
        Vector = Vector.astype(np.float64) / np.sqrt(np.inner(Vector, Vector))
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

    Epsilon: float = 1e-5

    Angles: np.ndarray = np.array([])
    Weights: np.ndarray = np.array([])

    for Stat in OrientationStats:
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
