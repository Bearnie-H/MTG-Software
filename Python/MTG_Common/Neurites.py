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

    ##  TESTING NEW PATH TRACKING

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

        DefaultLogWriter.Println(f"{self._UID}: Neurite originating at [ {self._Points[0]} ].")

        return

    #   ...

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
                DefaultLogWriter.Println(f"{CurrentNeurite._UID}: Neurite started at [ {CurrentNeurite._Points[0]} ] terminating at [ {Here} ]. Tree Size={CurrentNeurite.TreeSize()}")
            else:
                for (Index, Neighbour) in enumerate(Neighbours):
                    if ( Index == 0 ):
                        CurrentNeurite._Points = np.append(CurrentNeurite._Points, [Neighbour], axis=0)
                        NeuritesToExtend.appendleft(CurrentNeurite)
                    else:
                        DefaultLogWriter.Println(f"{CurrentNeurite._UID}: Neurite branching at [ {Here} ]. Tree Size={CurrentNeurite.TreeSize()}")
                        Child: Neurite = Neurite(Origin=Here)
                        Child._Points = np.append(Child._Points, [Neighbour], axis=0)
                        CurrentNeurite._Children.append(Child)
                        if ( Index == 1 ):
                            CurrentNeurite._BranchPoints = np.append(CurrentNeurite._BranchPoints, [Here], axis=0)
                        NeuritesToExtend.append(Child)

            CurrentNeurite = None

        # self.Draw(Utils.GreyscaleToBGR(Utils.GammaCorrection(CandidatePixels.copy(), Minimum=0, Maximum=127)), IncludeChildren=True)

        self._PrunePath()
        return self

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
            Image = cv2.line(Image.view(), NumpyCoordinateToTuple(Segment[1]), NumpyCoordinateToTuple(Segment[0]), Colour, 1, lineType=cv2.LINE_4)

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

    def __init__(self: NeuriteGraph) -> None:
        """
        Constructor

        This function...

        Return (None):
            ...
        """

        self.Edges = set()

        return

    def __len__(self: NeuriteGraph) -> int:
        return len(self.Edges)

    ### Public Methods
    def Draw(self: NeuriteGraph) -> np.ndarray:

        MaximumX, MaximumY = 0, 0
        for (From, To) in self.Edges:
            MaximumX, MaximumY = max(MaximumX, From.Coordinates[0], To.Coordinates[0]), max(MaximumY, From.Coordinates[1], To.Coordinates[1])
            MaximumX = math.ceil(MaximumX / 100) * 100
            MaximumY = math.ceil(MaximumY / 100) * 100

        Extent: int = max(MaximumX, MaximumY)

        if ( Extent == 0 ):
            Extent = 500

        Canvas: np.ndarray = np.zeros((Extent, Extent, 3), dtype=np.uint8)

        for (From, To) in self.Edges:
            self._DrawConnection(Canvas, From, To)

        return Canvas

    def AddNeurites(self: NeuriteGraph, Neurites: typing.Sequence[Neurite]) -> NeuriteGraph:
        """
        AddNeurites

        This function...

        Neurites:
            ...

        Return (self):
            ...
        """

        for Neurite in Neurites:
            self.AddNeurite(Neurite)

        return self

    def AddNeurite(self: NeuriteGraph, Neurite: Neurite) -> NeuriteGraph:
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

        for Child in Neurite._Children:
            self.AddNeurite(Child)

        return self

    def CollapseConnections(self: NeuriteGraph, Size: float = 3) -> NeuriteGraph:
        """
        CollapseConnections

        This function...

        Size:
            ...

        Return (self):
            ...
        """

        DefaultLogWriter.Println(f"Collapsing nodes of the adjacency graph which are within [ {Size:.2f} ] units of each other...")

        while ( True ):
            CurrentSize: int = len(self)
            PrunedSize: int = len(self._CollapseConnections(Size))
            # Utils.DisplayImage(f"Collapsing Nearby Nodes...", self.Draw(), 0.1, True, True, UpdateWindows=True)

            if ( PrunedSize == CurrentSize ):
                break

        # Utils.DisplayImage(f"Collapsing Nearby Nodes...", self.Draw(), 0, True, True)
        return self

    def RemoveCycles(self: NeuriteGraph) -> NeuriteGraph:
        """
        RemoveCycles

        This function...

        Return (self):
            ...
        """

        #   ...

        return self

    def OrientConnections(self: NeuriteGraph, Origins: np.ndarray) -> NeuriteGraph:
        """
        OrientConnections

        This function...

        Origins:
            ...

        Return (self):
            ...
        """

        #   We want to assert that connections are preferentially directed outward, away from
        #   the cortical explant cores. TODO: describe this more...

        OrientedConnections: typing.Set[typing.Tuple[GraphNode, GraphNode]] = set()

        #   Look at all of the connections in the graph...
        for (From, To) in self.Edges:

            #   Convert this connection into a vector representation...
            ConnectionVector: np.ndarray = ConnectionToVector(From, To)

            #   Identify which centroid origin is closest to the base of this vector
            ClosestCentroid: int = np.argmin(
                np.sum((Origins - CoordinateTupleToNumpyArray(From.Coordinates))**2)
            )

            #   Compute the vector from this centroid to the end of this connection vector
            RadialExplantVector: np.ndarray = CoordinateTupleToNumpyArray(To.Coordinates) - Origins[ClosestCentroid]

            #   Is the connection pointing outward?
            if ( np.dot(ConnectionVector, RadialExplantVector) < 0 ):
                #   No, it's backwards
                OrientedConnections.add((To, From))
            else:
                #   Yes, it's correct
                OrientedConnections.add((From, To))

        self.Edges = OrientedConnections
        Utils.DisplayImage(f"Oriented Connections", self.Draw(), 1, True, True)
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

        #   TODO: This isn't quite working correctly yet.
        #           If it identifies the two ends of a segment before checking the middle nodes, then it
        #           will add these single-ended connections first, which keeps the "middle" node(s) alive as zombie
        #           nodes.

        #   The point of this function is to identify any nodes which have a
        #   single incoming and outgoing connection, and where these are
        #   "co-linear enough". We could do this by examining the set of
        #   connections directly, but as there's no convenient way to "follow" a
        #   given path through the tree, we need a better way of searching.
        #
        #   We can transform the graph into its dual representation, into nodes
        #   which track their connections, and follow the paths these define as
        #   an alternative.

        #   Get all of the nodes within the graph...
        Nodes: typing.Set[GraphNode] = set()
        for (From, To) in self.Edges:
            Nodes.update([From, To])

        #   Get a copy of the set of connections in the graph we need to examine.
        ConnectionsToCheck: typing.Set[typing.Tuple[GraphNode, GraphNode]] = self.Edges.copy()

        #   The set of pruned connections to use in constructing the final pruned graph
        CheckedConnections: typing.Set[typing.Tuple[GraphNode, GraphNode]] = set()

        #   For each node in the graph, find all of the incoming and outgoing connections for it
        while ( len(Nodes) > 0 ):

            Node = Nodes.pop()

            IncomingConnections: typing.Set[typing.Tuple[GraphNode, GraphNode]] = set()
            OutgoingConnections: typing.Set[typing.Tuple[GraphNode, GraphNode]] = set()

            for (From, To) in ConnectionsToCheck:
                DefaultLogWriter.Println(f"Checking if connection [ {From.Coordinates}->{To.Coordinates} ] includes test node [ {Node.Coordinates} ]...")
                if ( Node == From ):
                    DefaultLogWriter.Println(f"Connection [ {From.Coordinates}->{To.Coordinates} ] is an outgoing connection")
                    OutgoingConnections.add((From, To))
                elif ( Node == To ):
                    DefaultLogWriter.Println(f"Connection [ {From.Coordinates}->{To.Coordinates} ] is an incoming connection.")
                    IncomingConnections.add((From, To))

            #   We can only prune this node if it has a single incoming and outgoing connection
            if not (( len(IncomingConnections) == 1 ) and ( len(OutgoingConnections) == 1 )):
                DefaultLogWriter.Println(f"The node at [ {Node.Coordinates} ] has [ {len(IncomingConnections)} ] incoming and [ {len(OutgoingConnections)} ] outgoing connections, and cannot be pruned.")
                CheckedConnections.update(IncomingConnections)
                CheckedConnections.update(OutgoingConnections)
            else:

                #   Transform these two connections into vectors so we can do vector math on them.
                #   Convert them to unit vectors since we only care about the direction they are pointing in
                (From, _) = IncomingConnections.pop()
                (_, To) = OutgoingConnections.pop()
                IncomingVector: np.ndarray = ConnectionToVector(From, Node, UnitVector=True)
                OutgoingVector: np.ndarray = ConnectionToVector(Node, To, UnitVector=True)

                #   If the angle between them is sufficiently small, they are "co-linear" enough
                if ( np.arccos(np.dot(IncomingVector, OutgoingVector)) <= Epsilon ):
                    DefaultLogWriter.Println(f"Pruning node at [ {Node.Coordinates} ] and forming connection [ {From.Coordinates}->{To.Coordinates} ]...")
                    CheckedConnections.add((From, To))
                else:
                    DefaultLogWriter.Println(f"Unable to prune node at [ {Node.Coordinates} ].")
                    CheckedConnections.add((From, Node))
                    CheckedConnections.add((Node, To))

            #   +++ DEBUGGING +++
            IntermediateGraph: NeuriteGraph = NeuriteGraph()
            IntermediateGraph.Edges = CheckedConnections.copy()
            Utils.DisplayImage("Pruning co-linear connections...", IntermediateGraph.Draw(), 0.5, True, True, UpdateWindows=True)
            #   --- DEBUGGING ---

        self.Edges = CheckedConnections
        Utils.DisplayImage("Pruning co-linear connections...", self.Draw(), 0.5, True, True)
        return self

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
            cv2.arrowedLine(Canvas, From.Coordinates, To.Coordinates, color=(255, 255, 255), thickness=1, tipLength=15.0 / Distance)

        return Canvas

    def _CollapseConnections(self: NeuriteGraph, Size: float) -> NeuriteGraph:
        """
        _CollapseConnections

        This function...

        Size:
            ...

        Return (NeuriteGraph):
            ...
        """

        #   Get the set of current connections we need to check
        ConnectionsToCheck: typing.Set[typing.Tuple[GraphNode, GraphNode]] = self.Edges.copy()

        #   Prepare a set of connections which have had the neighbourhood around both endpoints collapsed down
        CheckedConnections: typing.Set[typing.Tuple[GraphNode, GraphNode]] = set()

        #   Iterate over this set, getting both end-points of the connection
        for (From, To) in list(ConnectionsToCheck):

            #   If the two ends of this connection are close enough to collapse to the same point, skip this connection
            if ( DistanceBetween(From, To) <= Size ):
                continue

            SourceNeighbourhood: typing.Set[GraphNode] = set([From])
            DestinationNeighbourhood: typing.Set[GraphNode] = set([To])

            UpdateToSourceCoordinates: typing.Set[GraphNode] = set()
            UpdateToDestinationCoordinates: typing.Set[GraphNode] = set()

            for (TestFrom, TestTo) in ConnectionsToCheck:

                #   We need to do several things when examining the other
                #   connections in relation to the test connection under
                #   consideration. First:
                #
                #   1)  We need to identify all of the nodes within the
                #   neighbourhood of the Source.
                #   2)  We need to identify all of the nodes within the
                #   neighbourhood of the Destination.
                #
                #   This allows us to define the location of the collapsed
                #   connection entirely. Beyond this, we also need to make sure
                #   that any connections where *either* of their ends will be
                #   affected by this move are also tracked. This gives us a
                #   total of 9 classes of connections we need to track and
                #   update.
                #
                #   1) From the Source Neighbourhood to the Source Neighbourhood
                #   2) From the Source Neighbourhood to the Destination Neighbourhood
                #   3) From the Source Neighbourhood to anywhere else
                #   4) From the Destination Neighbourhood to the Source Neighbourhood
                #   5) From the Destination Neighbourhood to the Destination Neighbourhood
                #   6) From the Destination Neighbourhood to anywhere else
                #   7) From anywhere else to the Source Neighbourhood
                #   8) From anywhere else to the Destination Neighbourhood
                #   9) From anywhere else to anywhere else
                #
                #   Of these cases, we can handle them as follows:
                #
                #   1) Prune
                #   2) Collapse
                #   3) Outgoing: Update Source Coordinates
                #   4) Reversed - Update Source and Destination Coordinates
                #   5) Prune
                #   6) Secondary Outgoing: Update Source Coordinates
                #   7) Incoming: Update Destination Coordinates
                #   8) Secondary Incoming: Update Destination Coordinates
                #   9) Ignore

                #   Prepare flags for defining the orientation and location of this test connection
                StartsInSourceNeighbourhood: bool = ( DistanceBetween(From, TestFrom) <= Size )
                StartsInDestinationNeighbourhood: bool = ( DistanceBetween(To, TestFrom) <= Size )
                StartsElsewhere: bool = not ( StartsInSourceNeighbourhood or StartsInDestinationNeighbourhood )

                EndsInSourceNeighbourhood: bool = ( DistanceBetween(From, TestTo) <= Size )
                EndsInDestinationNeighbourhood: bool = ( DistanceBetween(To, TestTo) <= Size )
                EndsElsewhere: bool = not ( EndsInSourceNeighbourhood or EndsInDestinationNeighbourhood )

                #   First, identify whether either end of this connection lies in the Source neighbourhood
                if ( StartsInSourceNeighbourhood ):
                    SourceNeighbourhood.add(TestFrom)
                if ( EndsInSourceNeighbourhood ):
                    SourceNeighbourhood.add(TestTo)

                #   Second, identify whether either end of this connection lies in the Destination neighbourhood
                if ( StartsInDestinationNeighbourhood ):
                    DestinationNeighbourhood.add(TestFrom)
                if ( EndsInDestinationNeighbourhood ):
                    DestinationNeighbourhood.add(TestTo)

                #   Now, use the flags from above, and the connection type definitions
                #   to determine how to classify the test connection.
                if ( StartsInSourceNeighbourhood ) and ( EndsInSourceNeighbourhood ):
                    #   Class 1: Prune
                    pass
                elif ( StartsInSourceNeighbourhood ) and ( EndsInDestinationNeighbourhood ):
                    #   Class 2: Collapse
                    UpdateToSourceCoordinates.add(TestFrom)
                    UpdateToDestinationCoordinates.add(TestTo)
                    pass
                elif ( StartsInSourceNeighbourhood ) and ( EndsElsewhere ):
                    #   Class 3: Primary Outgoing
                    UpdateToSourceCoordinates.add(TestFrom)
                elif ( StartsInDestinationNeighbourhood ) and ( EndsInSourceNeighbourhood ):
                    #   Class 4: Reversed
                    UpdateToDestinationCoordinates.add(TestFrom)
                    UpdateToSourceCoordinates.add(TestTo)
                elif ( StartsInDestinationNeighbourhood ) and ( EndsInDestinationNeighbourhood ):
                    #   Class 5: Prune
                    pass
                elif ( StartsInDestinationNeighbourhood ) and ( EndsElsewhere ):
                    #   Class 6: Secondary Outgoing
                    UpdateToDestinationCoordinates.add(TestFrom)
                elif ( StartsElsewhere ) and ( EndsInSourceNeighbourhood ):
                    #   Class 7: Primary Incoming
                    UpdateToSourceCoordinates.add(TestTo)
                elif ( StartsElsewhere ) and ( EndsInDestinationNeighbourhood ):
                    #   Class 8: Secondary Incoming
                    UpdateToDestinationCoordinates.add(TestTo)
                elif ( StartsElsewhere ) and ( EndsElsewhere ):
                    #   Class 9: Ignore
                    pass

            #   From the set of nodes identified as being near either end of our test connection,
            #   compute the centroids of these neighbourhoods to place the collapsed end-endpoints at
            SourceCentroid: np.ndarray = np.mean(np.array([CoordinateTupleToNumpyArray(x.Coordinates) for x in SourceNeighbourhood]), axis=0)
            DestinationCentroid: np.ndarray = np.mean(np.array([CoordinateTupleToNumpyArray(x.Coordinates) for x in DestinationNeighbourhood]), axis=0)

            NewSource: GraphNode = GraphNode().SetOrigin(SourceCentroid)
            NewDestination: GraphNode = GraphNode().SetOrigin(DestinationCentroid)

            #   Update the coordinates of the nodes which are affected by collapsing these neighbourhoods
            for Node in UpdateToSourceCoordinates:
                Node.Coordinates = NewSource.Coordinates

            for Node in UpdateToDestinationCoordinates:
                Node.Coordinates = NewDestination.Coordinates

            CheckedConnections.add((NewSource, NewDestination))

            #   DEBUGGING
            # IntermediateGraph: NeuriteGraph = NeuriteGraph()
            # IntermediateGraph.Edges = CheckedConnections.copy()
            # Utils.DisplayImage(f"Collapsing Nearby Nodes...", IntermediateGraph.Draw(), 0.1, True, True, UpdateWindows=True)

        self.Edges = CheckedConnections
        return self

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
        return np.all(self.Coordinates == Other.Coordinates)

    def __hash__(self: GraphNode) -> int:
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
