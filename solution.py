"""
Graph Algorithms and Star System Teleportation Application
Shams Alkhalidy
"""

import heapq
import itertools
import math
import random
import time
import csv
from typing import TypeVar, Callable, Tuple, \
    List, Set, Dict, Any

import numpy as np
from queue import SimpleQueue

T = TypeVar('T')
Matrix = TypeVar('Matrix')  # Adjacency Matrix
Vertex = TypeVar('Vertex')  # Vertex Class Instance
Graph = TypeVar('Graph')  # Graph Class Instance


class Vertex:
    """ Class representing a Vertex object within a Graph """

    __slots__ = ['id', 'adj', 'visited', 'x', 'y']

    def __init__(self, id_init: str, x: float = 0, y: float = 0) -> None:
        """
        DO NOT MODIFY
        Initializes a Vertex
        :param id_init: A unique string identifier used for hashing the vertex
        :param x: The x coordinate of this vertex (used in a_star)
        :param y: The y coordinate of this vertex (used in a_star)
        """
        self.id = id_init
        self.adj = {}  # dictionary {id : weight} of outgoing edges
        self.visited = False  # boolean flag used in search algorithms
        self.x, self.y = x, y  # coordinates for use in metric computations

    def __eq__(self, other: Vertex) -> bool:
        """
        DO NOT MODIFY.
        Equality operator for Graph Vertex class.
        :param other: [Vertex] vertex to compare.
        :return: [bool] True if vertices are equal, else False.
        """
        if self.id != other.id:
            return False
        if self.visited != other.visited:
            print(f"Vertex '{self.id}' not equal")
            print(f"Vertex visited flags not equal: self.visited={self.visited},"
                  f" other.visited={other.visited}")
            return False
        if self.x != other.x:
            print(f"Vertex '{self.id}' not equal")
            print(f"Vertex x coords not equal: self.x={self.x}, other.x={other.x}")
            return False
        if self.y != other.y:
            print(f"Vertex '{self.id}' not equal")
            print(f"Vertex y coords not equal: self.y={self.y}, other.y={other.y}")
            return False
        if set(self.adj.items()) != set(other.adj.items()):
            diff = set(self.adj.items()).symmetric_difference(set(other.adj.items()))
            print(f"Vertex '{self.id}' not equal")
            print(f"Vertex adj dictionaries not equal:"
                  f" symmetric diff of adjacency (k,v) pairs = {str(diff)}")
            return False
        return True

    def __repr__(self) -> str:
        """
        DO NOT MODIFY
        Constructs string representation of Vertex object.
        :return: [str] string representation of Vertex object.
        """
        lst = [f"<id: '{k}', weight: {v}>" for k, v in self.adj.items()]
        return f"<id: '{self.id}'" + ", Adjacencies: " + "".join(lst) + ">"

    __str__ = __repr__

    def __hash__(self) -> int:
        """
        DO NOT MODIFY
        Hashes Vertex into a set; used in unit tests
        :return: hash value of Vertex
        """
        return hash(self.id)

    # ============== Project 9 Vertex Methods Below ==============#
    def deg(self) -> int:
        """
        Returns the degree (number of outgoing edges from) of this Vertex.
        :return: [int] Degree of this vertex.
        """
        return len(self.adj)

    def get_outgoing_edges(self) -> Set[Tuple[str, float]]:
        """
        Returns a set of tuples (end_id, weight) or empty set if no adjacencies.
        :return: [set[tuple[str, float]]] Set of tuples of the form (end_id, weight).
        """
        return set(self.adj.items())

    def euclidean_distance(self, other: Vertex) -> float:
        """
        Computes the Euclidean distance between this Vertex and another Vertex.
        :param other: [Vertex] Other Vertex to which we wish to compute distance.
        :return: [float] Euclidean distance between this Vertex and another Vertex.
        """
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)

    def taxicab_distance(self, other: Vertex) -> float:
        """
        Returns the taxicab distance between this Vertex and another Vertex.
        :param other: [Vertex] Other Vertex to which we wish to compute distance.
        :return: [float] Taxicab distance between this Vertex and another Vertex.
        """
        return abs(self.x - other.x) + abs(self.y - other.y)


class Graph:
    """ Class implementing the Graph ADT using an Adjacency Map structure """

    __slots__ = ['size', 'vertices', 'plot_show', 'plot_delay']

    def __init__(self, plt_show: bool = False, matrix: Matrix = None, csvf: str = "") -> None:
        """
        DO NOT MODIFY
        Instantiates a Graph class instance
        :param: plt_show : if true, render plot when plot() is called; else, ignore calls to plot()
        :param: matrix : optional matrix parameter used for fast construction
        :param: csvf : optional filepath to a csv containing a matrix
        """
        matrix = matrix if matrix else np.loadtxt(csvf, delimiter=',', dtype=str).tolist() if csvf else None
        self.size = 0
        self.vertices = {}

        self.plot_show = plt_show
        self.plot_delay = 0.2

        if matrix is not None:
            for i in range(1, len(matrix)):
                for j in range(1, len(matrix)):
                    if matrix[i][j] == "None" or matrix[i][j] == "":
                        matrix[i][j] = None
                    else:
                        matrix[i][j] = float(matrix[i][j])
            self.matrix2graph(matrix)

    def __eq__(self, other: Graph) -> bool:
        """
        DO NOT MODIFY
        Overloads equality operator for Graph class
        :param other: graph to compare
        """
        if self.size != other.size or len(self.vertices) != len(other.vertices):
            print(f"Graph size not equal: self.size={self.size}, other.size={other.size}")
            return False
        else:
            for vertex_id, vertex in self.vertices.items():
                other_vertex = other.get_vertex_by_id(vertex_id)
                if other_vertex is None:
                    print(f"Vertices not equal: '{vertex_id}' not in other graph")
                    return False

                adj_set = set(vertex.adj.items())
                other_adj_set = set(other_vertex.adj.items())

                if not adj_set == other_adj_set:
                    print(f"Vertices not equal: adjacencies of '{vertex_id}' not equal")
                    print(f"Adjacency symmetric difference = "
                          f"{str(adj_set.symmetric_difference(other_adj_set))}")
                    return False
        return True

    def __repr__(self) -> str:
        """
        DO NOT MODIFY
        :return: String representation of graph for debugging
        """
        return "Size: " + str(self.size) + ", Vertices: " + str(list(self.vertices.items()))

    __str__ = __repr__

    def plot(self) -> None:
        """
        Creates a plot a visual representation of the graph using matplotlib
        """

        if self.plot_show:
            import matplotlib.cm as cm
            import matplotlib.patches as patches
            import matplotlib.pyplot as plt

            # if no x, y coords are specified, place vertices on the unit circle
            for i, vertex in enumerate(self.get_all_vertices()):
                if vertex.x == 0 and vertex.y == 0:
                    vertex.x = math.cos(i * 2 * math.pi / self.size)
                    vertex.y = math.sin(i * 2 * math.pi / self.size)

            # show edges
            num_edges = len(self.get_all_edges())
            max_weight = max([edge[2] for edge in self.get_all_edges()]) if num_edges > 0 else 0
            colormap = cm.get_cmap('cool')
            for i, edge in enumerate(self.get_all_edges()):
                origin = self.get_vertex_by_id(edge[0])
                destination = self.get_vertex_by_id(edge[1])
                weight = edge[2]

                # plot edge
                arrow = patches.FancyArrowPatch((origin.x, origin.y),
                                                (destination.x, destination.y),
                                                connectionstyle="arc3,rad=.2",
                                                color=colormap(weight / max_weight),
                                                zorder=0,
                                                **dict(arrowstyle="Simple,tail_width=0.5,"
                                                                  "head_width=8,head_length=8"))
                plt.gca().add_patch(arrow)

                # label edge
                plt.text(x=(origin.x + destination.x) / 2 - (origin.x - destination.x) / 10,
                         y=(origin.y + destination.y) / 2 - (origin.y - destination.y) / 10,
                         s=weight, color=colormap(weight / max_weight))

            # show vertices
            x = np.array([vertex.x for vertex in self.get_all_vertices()])
            y = np.array([vertex.y for vertex in self.get_all_vertices()])
            labels = np.array([vertex.id for vertex in self.get_all_vertices()])
            colors = np.array(
                ['yellow' if vertex.visited else 'black' for vertex in self.get_all_vertices()])
            plt.scatter(x, y, s=40, c=colors, zorder=1)

            # plot labels
            for j, _ in enumerate(x):
                plt.text(x[j] - 0.03 * max(x), y[j] - 0.03 * max(y), labels[j])

            # show plot
            plt.show()

            # delay execution to enable animation
            time.sleep(self.plot_delay)

    def add_to_graph(self, begin_id: str, end_id: str = None, weight: float = 1) -> None:
        """
        Adds to graph: creates start vertex if necessary,
        an edge if specified,
        and a destination vertex if necessary to create said edge
        If edge already exists, update the weight.
        :param begin_id: unique string id of starting vertex
        :param end_id: unique string id of ending vertex
        :param weight: weight associated with edge from start -> dest
        :return: None
        """
        if self.vertices.get(begin_id) is None:
            self.vertices[begin_id] = Vertex(begin_id)
            self.size += 1
        if end_id is not None:
            if self.vertices.get(end_id) is None:
                self.vertices[end_id] = Vertex(end_id)
                self.size += 1
            self.vertices.get(begin_id).adj[end_id] = weight

    def matrix2graph(self, matrix: Matrix) -> None:
        """
        Given an adjacency matrix, construct a graph
        matrix[i][j] will be the weight of an edge between the vertex_ids
        stored at matrix[i][0] and matrix[0][j]
        Add all vertices referenced in the adjacency matrix, but only add an
        edge if matrix[i][j] is not None
        Guaranteed that matrix will be square
        If matrix is nonempty, matrix[0][0] will be None
        :param matrix: an n x n square matrix (list of lists) representing Graph as adjacency map
        :return: None
        """
        for i in range(1, len(matrix)):  # add all vertices to begin with
            self.add_to_graph(matrix[i][0])
        for i in range(1, len(matrix)):  # go back through and add all edges
            for j in range(1, len(matrix)):
                if matrix[i][j] is not None:
                    self.add_to_graph(matrix[i][0], matrix[j][0], matrix[i][j])

    def graph2matrix(self) -> Matrix:
        """
        given a graph, creates an adjacency matrix of the type described in "construct_from_matrix"
        :return: Matrix
        """
        matrix = [[None] + [v_id for v_id in self.vertices]]
        for v_id, outgoing in self.vertices.items():
            matrix.append([v_id] + [outgoing.adj.get(v) for v in self.vertices])
        return matrix if self.size else None

    def graph2csv(self, filepath: str) -> None:
        """
        given a (non-empty) graph, creates a csv file containing data necessary to reconstruct that graph
        :param filepath: location to save CSV
        :return: None
        """
        if self.size == 0:
            return

        with open(filepath, 'w+') as graph_csv:
            csv.writer(graph_csv, delimiter=',').writerows(self.graph2matrix())

    def unvisit_vertices(self) -> None:
        """
        Resets all visited flags of vertices in Graph to false
        :return: None
        """
        for vertex in self.vertices.values():
            vertex.visited = False

    def get_vertex_by_id(self, v_id: str) -> Vertex:
        """
        Retrieves a Vertex in the Graph by a provided v_id
        :param vertex_id: unique string id of other vertex
        :return: Vertex object if found; else None
        """
        return self.vertices.get(v_id)

    def get_all_vertices(self) -> Set[Vertex]:
        """
        Returns a set of all vertices in the Graph
        :return: set(Vertex)
        """
        return set(self.vertices.values())

    def get_edge_by_ids(self, begin_id: str, end_id: str) -> Tuple[str, str, float]:
        """
        Returns the edge connecting the vertex with id begin_id to the vertex with id end_id
        If edge does not exist, returns None
        :return: tuple(begin_id, end_id, weight) or None if edge does not exist
        """
        if self.vertices.get(begin_id) and self.vertices.get(end_id):
            weight = self.vertices.get(begin_id).adj.get(end_id)
            return (begin_id, end_id, weight) if weight is not None else None

    def get_all_edges(self) -> Set[Tuple[str, str, float]]:
        """
        Returns all edges in the graph
        :return: set(tuple(begin_id, end_id, weight)) or empty list if Graph is empty
        """
        return {(begin_id, end_id, v.adj[end_id])
                for begin_id, v in self.vertices.items() for end_id in v.adj}

    def build_path(self, back_edges: Dict[str, str], begin_id: str, end_id: str) -> Tuple[List[str], float]:
        """
        Given a dictionary of back-edges (a mapping of vertex id to predecessor vertex id),
        reconstruct the path from start_id to end_id and compute the total distance
        :param back_edges: Dictionary of back-edges, i.e., (key=vertex_id, value=predecessor_vertex_id) pairs
        :param start_id: Starting vertex ID string from which to construct path
        :param end_id: Ending vertex ID string to which to construct path
        :return: Tuple where first element is a list of vertex IDs specifying a path from start_id to end_id
                 and the second element is the cost of this path
        """
        path, dist = [end_id], 0
        while path[-1] != begin_id:
            path.append(back_edges.get(path[-1]))  # will construct path in O(V) - no calls to .insert(0)
            dist += self.get_edge_by_ids(path[-1], path[-2])[2]
        return list(reversed(path)), dist

    def bfs(self, begin_id: str, end_id: str) -> Tuple[List[str], float]:
        """
        Perform a breadth-first search beginning at vertex with id begin_id and terminating at vertex with id end_id
        :param begin_id: starting vertex ID string from which to start BFS
        :param end_id: ending vertex ID string from which to end BFS
        :return: Tuple where first element is a list of vertex IDs specifying a path from begin_id to end_id
                 and a second element is the cost of this path
        Time Complexity: O(V+E)
        Space Complexity: O(V)
        """
        # one of the vertices does not exist
        if begin_id not in self.vertices or end_id not in self.vertices:
            return [], 0

        self.unvisit_vertices()     # mark all vertices to be False visited
        queue = SimpleQueue()       # create new queue
        back_edges = {}             # edges dict
        queue.put(begin_id)         # insert first id and set it to be visited
        self.get_vertex_by_id(begin_id).visited = True

        while not queue.empty():
            current_id = queue.get()
            current_vertex = self.get_vertex_by_id(current_id)

            for neighbor_id in current_vertex.adj:
                # if neighbor adj vertex not visited:
                if not self.get_vertex_by_id(neighbor_id).visited:
                    self.get_vertex_by_id(neighbor_id).visited = True
                    # insert an edge item for it into the back_edges dict
                    back_edges[neighbor_id] = current_id
                    queue.put(neighbor_id)      # enqueue neighbor_id into queue

                    if neighbor_id == end_id:
                        # when reaching to end_id build a path using the back_edges
                        return self.build_path(back_edges, begin_id, end_id)

        return [], 0

    def a_star(self, begin_id: str, end_id: str,
               metric: Callable[[Vertex, Vertex], float]) -> Tuple[List[str], float]:
        """
        Perform an A* search beginning at vertex with id start_id and terminating at vertex with id end_id
        :param begin_id: starting vertex ID string from which to start BFS
        :param end_id: ending vertex ID string from which to end BFS
        :param metric: This is a callable parameter and will always be Vertex.euclidean_distance or Vertex.taxicab_distance
        :returns tuple of the form ([path], distance) where
        [path] is a list of vertex id strings beginning with start_id, terminating with end_id, and including the ids of all intermediate vertices connecting the two
        distance is the sum of the weights of the edges along the [path] traveled
        Time Complexity: O((E + V)log(V))
        Space Complexity: O(V)
        """
        if begin_id not in self.vertices or end_id not in self.vertices:
            return [], 0

        self.unvisit_vertices()
        queue = PriorityQueue()
        back_edges = {}
        g_values = {begin_id: 0}
        begin_vertex = self.get_vertex_by_id(begin_id)
        queue.push(0, begin_vertex)

        while not queue.empty():
            current_priority, current_vertex = queue.pop()

            if current_vertex.id == end_id:
                return self.build_path(back_edges, begin_id, end_id)

            for neighbor_id, edge_weight in current_vertex.adj.items():
                neighbor_vertex = self.get_vertex_by_id(neighbor_id)
                tentative_g_value = g_values[current_vertex.id] + edge_weight

                if neighbor_id not in g_values or tentative_g_value < g_values[neighbor_id]:
                    g_values[neighbor_id] = tentative_g_value
                    back_edges[neighbor_id] = current_vertex.id
                    f_value = tentative_g_value + metric(neighbor_vertex, self.get_vertex_by_id(end_id))

                    if neighbor_id in queue.locator:
                        queue.update(f_value, neighbor_vertex)
                    else:
                        queue.push(f_value, neighbor_vertex)
        return [], 0


class PriorityQueue:
    """
    Priority Queue built upon heapq module with support for priority key updates
    Inspired by https://docs.python.org/2/library/heapq.html
    """

    __slots__ = ['data', 'locator', 'counter']

    def __init__(self) -> None:
        """
            Construct an AStarPriorityQueue object
            """
        self.data = []  # underlying data list of priority queue
        self.locator = {}  # dictionary to locate vertices within priority queue
        self.counter = itertools.count()  # used to break ties in prioritization

    def __repr__(self) -> str:
        """
            Represent AStarPriorityQueue as a string
            :return: string representation of AStarPriorityQueue object
            """
        lst = [f"[{priority}, {vertex}], " if vertex is not None else "" for
               priority, count, vertex in self.data]
        return "".join(lst)[:-1]

    def __str__(self) -> str:
        """
            Represent AStarPriorityQueue as a string
            :return: string representation of AStarPriorityQueue object
            """
        return repr(self)

    def empty(self) -> bool:
        """
            Determine whether priority queue is empty
            :return: True if queue is empty, else false
            """
        return len(self.data) == 0

    def push(self, priority: float, vertex: Vertex) -> None:
        """
            Push a vertex onto the priority queue with a given priority
            :param priority: priority key upon which to order vertex
            :param vertex: Vertex object to be stored in the priority queue
            :return: None
            """
        # list is stored by reference, so updating will update all refs
        node = [priority, next(self.counter), vertex]
        self.locator[vertex.id] = node
        heapq.heappush(self.data, node)

    def pop(self) -> Tuple[float, Vertex]:
        """
            Remove and return the (priority, vertex) tuple with lowest priority key
            :return: (priority, vertex) tuple where priority is key,
            and vertex is Vertex object stored in priority queue
            """
        vertex = None
        while vertex is None:
            # keep popping until we have valid entry
            priority, count, vertex = heapq.heappop(self.data)
        del self.locator[vertex.id]  # remove from locator dict
        vertex.visited = True  # indicate that this vertex was visited
        while len(self.data) > 0 and self.data[0][2] is None:
            heapq.heappop(self.data)  # delete trailing Nones
        return priority, vertex

    def update(self, new_priority: float, vertex: Vertex) -> None:
        """
            Update given Vertex object in the priority queue to have new priority
            :param new_priority: new priority on which to order vertex
            :param vertex: Vertex object for which priority is to be updated
            :return: None
            """
        node = self.locator.pop(vertex.id)  # delete from dictionary
        node[-1] = None  # invalidate old node
        self.push(new_priority, vertex)  # push new node


def teleport(system_dict, start, end):
    """
    Teleport function to calculate the shortest path between two points in a network of systems.
    system_dict: A dictionary containing information about each galaxy system.
    start: A tuple containing the name of the start galaxy and the start vertex within that galaxy.
    end: A tuple containing the name of the end galaxy and the end vertex within that galaxy.
    Returns distance (float) between start and end
    """
    start_system, start_vertex = start
    end_system, end_vertex = end
    # When system isn't there
    if start_system not in system_dict or end_system not in system_dict:
        return float('inf')

    start_graph = system_dict[start_system]["graph"]
    end_graph = system_dict[end_system]["graph"]
    # when vertex isn't in the graph
    if start_vertex not in start_graph.vertices or end_vertex not in end_graph.vertices:
        return float('inf')

    # same system distances
    if start_system == end_system:
        # Direct path distance
        direct_path, direct_distance = start_graph.a_star(start_vertex, end_vertex, Vertex.euclidean_distance)
        if not direct_path:
            direct_distance = 1000
        d = 0
        # Distance via teleportation within the same system
        path_to_departure, distance_to_departure = start_graph.a_star(start_vertex, system_dict[start_system]["departure_teleport"], Vertex.euclidean_distance)
        new_system = system_dict[start_system]["departure_destinations"]
        if len(new_system) == 1:
            item = list(new_system)[0]
            int_sys = system_dict[item]
            p,d = int_sys["graph"].a_star(int_sys["arrival_teleport"],int_sys["departure_teleport"], Vertex.euclidean_distance)
        path_from_arrival, distance_from_arrival = start_graph.a_star(system_dict[start_system]["arrival_teleport"], end_vertex, Vertex.euclidean_distance)

        # If both parts of the teleport path exist, calculate the teleport distance
        if distance_from_arrival != 0 and distance_to_departure != 0:
                return min(direct_distance, (distance_to_departure + distance_from_arrival + d))
        # If no direct path or teleport path exists, return infinity
        if direct_distance != 1000:
            return direct_distance
        return float('inf')

    # path to departure
    path_to_departure, distance_to_departure = start_graph.a_star(start_vertex, system_dict[start_system]["departure_teleport"], Vertex.euclidean_distance)
    if not path_to_departure:
        return float('inf')

    # Direct teleportation to the end system
    if end_system in system_dict[start_system]["departure_destinations"]:
        path_from_arrival, distance_from_arrival = end_graph.a_star(system_dict[end_system]["arrival_teleport"], end_vertex, Vertex.euclidean_distance)
        if path_from_arrival:
            return distance_to_departure + distance_from_arrival
        else:
            return float('inf')

    # Teleportation via intermediate systems
    min_additional_distance = float('inf')
    for intermediate_system in system_dict[start_system]["departure_destinations"]:
        if intermediate_system == end_system:
            continue
        #  calculate intermediate system distance
        intermediate_graph = system_dict[intermediate_system]["graph"]
        path_through_intermediate, distance_through_intermediate = intermediate_graph.a_star(system_dict[intermediate_system]["arrival_teleport"], system_dict[intermediate_system]["departure_teleport"], Vertex.euclidean_distance)

        if path_through_intermediate:
            distance_to_end = teleport(system_dict, (intermediate_system, system_dict[intermediate_system]["departure_teleport"]), end)
            if distance_to_end != float('inf'):
                total_distance = distance_to_departure + distance_through_intermediate + distance_to_end
                min_additional_distance = min(min_additional_distance, total_distance)

    return min_additional_distance if min_additional_distance != float('inf') else float('inf')
