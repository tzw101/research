"""Graph implementation.

Provides an implementation of a Minimum Spanning Tree.

This class accepts a distance dataframe as an input to form a MST using Prim's
algorithm.
How to use distance matrix to produce a MST is seen in:
Introduction to econophysics: correlations and complexity in finance,
chapter 13.
"""

import heapq

class Graph:

    def __init__(self, vertices):
        """Stores vertices by name and values.

        Examples
        The following vertices
           | A  |  B
        A  | 1  |  5
        B  | 3  |  -1
        Are stored in a dictionary as
        dict[A] = [(1, A, A), (5, A, B)]
        dict[B] = [(3, B, A), (-1, B, B)]
        Noting that the underlying order is kept.
        """
        self.vertices = {}
        for vertex in vertices:
            self.vertices[vertex] = [(edge[dest], vertex, dest) for dest, edge in vertices[vertex].groupby(level=0)]

    def smallest_edge(self, edges, visited, not_visited):
        """Finds the smallest cost edge (u, v) that satisfies the following
        conditions:
        - u is in visited
        - v is not visited
        - edge has smallest cost
        """
        found = False
        old_edges = edges.copy()
        while not found:
            edge = heapq.heappop(old_edges)
            if edge[1] in visited and edge[2] in not_visited:
                found = True
        return edge




    def mst(self):
        """Returns minimum spanning tree in a dictionary."""
        # Presort edges by using a priority queue.
        edges = set()
        for vertex in self.vertices:
            for edge in self.vertices[vertex]:
                # If to and destination are different.
                if edge[1] != edge[2]:
                    edges.add(edge)
        edges = list(edges)
        heapq.heapify(edges)
        edge = heapq.heappop(edges)
        visited = set([edge[1], edge[2]])
        not_visited = set([k for k in self.vertices])
        not_visited.remove(edge[1])
        not_visited.remove(edge[2])
        spanning_tree = set([edge])
        while not_visited:
            edge = self.smallest_edge(edges, visited, not_visited)
            visited.add(edge[2])
            not_visited.remove(edge[2])
            spanning_tree.add(edge)
        return spanning_tree




