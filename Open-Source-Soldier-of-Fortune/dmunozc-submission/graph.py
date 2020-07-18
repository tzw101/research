"""Graph implementation.

Provides an implementation of a Minimum Spanning Tree.

This class accepts a distance dataframe as an input to form a MST using
Kruskal's and Prim's algorithms.
How to use distance matrix to produce a MST is seen in:
Introduction to econophysics: correlations and complexity in finance,
chapter 13.
"""

import heapq


class Graph:
    """This class implements a graph data structure."""

    def __init__(self, vertices):
        """Stores vertices by name and values.

        Examples
        The following vertices
        *    **FROM**
        *    | A | B
        T  A | 1 | 5
        O  B | 3 | -1
        *
        *
        Are stored in a dictionary as
        dict[A] = [(1, A, A), (5, A, B)]
        dict[B] = [(3, B, A), (-1, B, B)]
        Noting that the underlying order is kept.

        Parameters
        ----------
        vertices: pandas dataframe
            dataframe that contains the distances of a graph.
        """
        self.graph_dict = {}
        for vertex in vertices:
            self.graph_dict[vertex] = [
                (edge[dest], vertex, dest)
                for dest, edge in vertices[vertex].groupby(level=0)
                if edge[dest] != 0
            ]

    def __generate_edges(self, directed=True, weights=False):
        """Returns list of edges, with or without weights, directed or
        undirected.

        Returns
        -------
        edges: list of edge tuples
        """
        edges = []
        i = 0
        for vertex in self.graph_dict:
            for j in range(i, len(self.graph_dict[vertex])):
                edge = self.graph_dict[vertex][j]
                # Remove the weights if not required.
                if not weights:
                    edge = edge[1:]
                edges.append(edge)
            # If the graph is undirected, skip the symmetry.
            if not directed:
                i += 1
        return edges

    def __smallest_edge(self, edges, visited, not_visited):
        """Finds the smallest cost edge (u, v) for creating an MST.

        It satisfies the following conditions:
        - origin is in visited.
        - destination is not visited.
        - edge has smallest cost.

        Parameters
        ----------
        edges : list of tuples
            a priority queue list that contains the edges tuple.
            edges tuple consists of (cost, origin, destination).
        visited: set
            a set that contains the currently visited nodes.
        not_visited: set
            a set that contains the currently not visited nodes.

        Returns
        -------
        edge : tuple
            tuple that contains the edge information.
            tuple consists of (cost, origin, destination)
        """
        found = False
        old_edges = edges.copy()
        while not found:
            edge = heapq.heappop(old_edges)
            if any(v in visited for v in edge[1:]) and \
                    any(v in not_visited for v in edge[1:]):
                found = True
        return edge

    def __findset(self, parents, vertex):
        """FindSet implementation using path compression.

        Follows the chain of parent pointers from x up the tree until it
        reaches a root element.

        Parameters
        ----------
        parents: dictionary
            contains the parents of each vertex
        vertex: vertex
            the vertex to find the parent of

        Returns
        -------
        parent: vertex
            root element of vertex.

        References
        ----------
        [1] https://en.wikipedia.org/wiki/Disjoint-set_data_structure
        """
        if parents[vertex] == vertex:
            return vertex
        return self.__findset(parents, parents[vertex])

    def __union(self, parents, ranks, v1, v2):
        """Union implementation.

        Uses findset to determine the roots of the trees x and y belong to.
        It updates the rank list.

        Parameters
        ----------
        parents: dictionary
            contains the parents of each vertex
        rank: dictionary
            contains the ranks of each parent
        v1: vertex
            first vertex
        v2: vertex
            second vertex

        References
        ----------
        [1] https://en.wikipedia.org/wiki/Disjoint-set_data_structure
        """
        v1_root = self.__findset(parents, v1)
        v2_root = self.__findset(parents, v2)
        if v1_root == v2_root:
            return
        # If v1_root and v2_root are not in same set, merge them.
        if ranks[v1_root] < ranks[v2_root]:
            v1_root, v2_root = v2_root, v1_root
        parents[v2_root] = v1_root
        if ranks[v1_root] == ranks[v2_root]:
            ranks[v1_root] += 1

    def vertices(self):
        """Returns the vertices as a list.

        Returns
        -------
        vertices: list
        """
        return list(self.graph_dict.keys())

    def edges(self):
        """Returns list of unweighted edges.

        Returns
        -------
        edges: list of edge tuples
        """
        return self.__generate_edges(directed=True, weights=False)

    def weighted_edges(self):
        """Returns list of weighted edges.

        Returns
        -------
        edges: list of edge tuples
        """
        return self.__generate_edges(directed=True, weights=True)

    def prim_mst(self):
        """Returns minimum spanning tree in a dictionary using Prim's
        algorithm.

        Vanilla Prim's algorithm only works for an undirected graph.
        Therefore this implementation assumes symmetry of adjacency matrix

        Returns
        ----------
        spanning_tree: set
            set that contains the minimum spanning tree edges.
        """
        # Find all distinct edges. Assumed an undirected graph.
        edges = self.__generate_edges(directed=False, weights=True)
        # Presort edges by using a priority queue.
        heapq.heapify(edges)
        # Use the lowest cost edge as the starting point.
        edge = heapq.heappop(edges)
        visited = set(edge[1:])
        not_visited = set([k for k in self.graph_dict])
        not_visited.difference_update(edge[1:])
        spanning_tree = set([edge])
        # Create the mst based on lowest cost edges.
        while not_visited:
            edge = self.__smallest_edge(edges, visited, not_visited)
            visited.update(edge[1:])
            not_visited.difference_update(edge[1:])
            spanning_tree.add(edge)
        return spanning_tree

    def kruskal_mst(self):
        """Returns minimum spanning tree in a dictionary using Kruskal's
        algorithm.

        Vanilla Kruskal's algorithm only works for an undirected graph.
        Therefore this implementation assumes symmetry of adjacency matrix

        Returns
        ----------
        spanning_tree: set
            set that contains the minimum spanning tree edges.
        """
        spanning_tree = set()
        # Find all distinct edges. Assumed an undirected graph.
        edges = self.__generate_edges(directed=False, weights=True)
        vertices = self.vertices()
        # Sort edges by using a priority queue.
        heapq.heapify(edges)
        # Initialize parents and ranks
        parents = {}
        ranks = {}
        for v in vertices:
            parents[v] = v
            ranks[v] = 0
        # Terminate when number of edges denotes a tree.
        while len(spanning_tree) < len(vertices) - 1:
            edge = heapq.heappop(edges)
            from_v = self.__findset(parents, edge[1])
            to_v = self.__findset(parents, edge[2])
            if from_v != to_v:
                spanning_tree.add(edge)
                self.__union(parents, ranks, from_v, to_v)
        return spanning_tree
