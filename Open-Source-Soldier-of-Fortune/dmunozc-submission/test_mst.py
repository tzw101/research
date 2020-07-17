"""Unit tests for mst.py"""


import pytest
from graph import Graph
import pandas as pd
import itertools


@pytest.fixture
def few_names():
    return ["A", "B", "C"]


@pytest.fixture
def few_distances():
    return [[1, 2, 3], [4, 5, 6], [-7, -8, -9]]


@pytest.fixture
def few_graph(few_distances, few_names):
    df = pd.DataFrame(few_distances, columns=few_names, index=few_names)
    return Graph(df)


@pytest.fixture
def book_names():
    return ["CHV", "GE", "KO", "PG", "TX", "XON"]


@pytest.fixture
def book_distances():
    """Mantenga, Stanley chapter 13 example"""
    return [
        [0, 1.15, 1.18, 1.15, 0.84, 0.89],
        [1.15, 0, 0.86, 0.89, 1.26, 1.16],
        [1.18, 0.86, 0, 0.74, 1.27, 1.11],
        [1.15, 0.89, 0.74, 0, 1.26, 1.10],
        [0.84, 1.26, 1.27, 1.26, 0, 0.94],
        [0.89, 1.16, 1.11, 1.10, 0.94, 0],
    ]


@pytest.fixture
def book_graph(book_distances, book_names):
    """Creates the graph based on the other fixtures."""
    df = pd.DataFrame(book_distances, columns=book_names, index=book_names)
    return Graph(df)


def test_graphs(book_distances, book_names, few_distances, few_names):
    """Tests that the produce graph matches the matrix."""
    for dst, names in [
        (book_distances, book_names),
        (few_distances, few_names),
    ]:
        df = pd.DataFrame(dst, columns=names, index=names)
        graph = Graph(df)
        # For each vertex value in the graph, check it against the matrix.
        # Since distance of 0 is ignored in the graph, need to offset it if
        # present.
        for i in range(len(names)):
            offset = 0
            vertex = names[i]
            for j in range(len(graph.graph_dict[vertex])):
                if dst[j][i] == 0:
                    offset = -1
                else:
                    assert graph.graph_dict[vertex][j + offset][0] == dst[j][i]


def test_vertices(book_graph, book_names, few_graph, few_names):
    """Tests that the created graph contains the vertices it was created
    with."""
    for graph, names in [(book_graph, book_names), (few_graph, few_names)]:
        vertices = graph.vertices()
        assert names == vertices


def test_few_edges(few_graph, few_names):
    """Tests the undirected edges are present from the created graph."""
    all_edges = [p for p in itertools.product(few_names, repeat=2)]
    graph_edges = few_graph.edges()
    assert len(graph_edges) == len(all_edges)
    assert graph_edges == all_edges


def test_book_mst(book_distances, book_names):
    """Tests the MST produces matches Stanley's book.

    Note that MSTs are not unique, so there is always a possibility that
    exact matching answers are not possible.
    """
    # From Stanley chapter 13
    df = pd.DataFrame(book_distances, columns=book_names, index=book_names)
    graph = Graph(df)
    for i in range(2):
        if i == 0:
            mst = graph.kruskal_mst()
        elif i == 1:
            mst = graph.prim_mst()
        mst = sorted(mst)
        assert mst[0][0] == 0.74 and "KO" in mst[0] and "PG" in mst[0]
        assert mst[1][0] == 0.84 and "CHV" in mst[1] and "TX" in mst[1]
        assert mst[2][0] == 0.86 and "KO" in mst[2] and "GE" in mst[2]
        assert mst[3][0] == 0.89 and "CHV" in mst[3] and "XON" in mst[3]
        assert mst[4][0] == 1.10 and "PG" in mst[4] and "XON" in mst[4]
