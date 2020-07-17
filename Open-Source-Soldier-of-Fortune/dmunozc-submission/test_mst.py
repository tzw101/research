"""Unit tests for mst.py"""


import pytest
from graph import Graph
import pandas as pd
import numpy as np

@pytest.fixture
def small_distances():
    """Mantenga, Stanley chapter 13 example"""
    return [[0,    1.15, 1.18, 1.15, 0.84, 0.89],
            [1.15, 0,    0.86, 0.89, 1.26, 1.16],
            [1.18, 0.86, 0,    0.74, 1.27, 1.11],
            [1.15, 0.89, 0.74, 0,    1.26, 1.10],
            [0.84, 1.26, 1.27, 1.26, 0,    0.94],
            [0.89, 1.16, 1.11, 1.10, 0.94, 0   ]]


def test_mst_book(small_distances):
    """Tests the MST produces matches Stanley's book.

    Note that MSTs are not unique, so there is always a possibility that
    exact matching answers are not possible.
    """
    # From Stanley chapter 13
    names = ["CHV", "GE", "KO", "PG", "TX", "XON"]
    df = pd.DataFrame(small_distances, columns=names, index=names)
    graph = Graph(df)
    mst = graph.mst()
    mst = sorted(mst)
    assert(mst[0][0] == 0.74 and "KO"  in mst[0] and "PG"  in mst[0])
    assert(mst[1][0] == 0.84 and "CHV" in mst[1] and "TX"  in mst[1])
    assert(mst[2][0] == 0.86 and "KO"  in mst[2] and "GE"  in mst[2])
    assert(mst[3][0] == 0.89 and "CHV" in mst[3] and "XON" in mst[3])
    assert(mst[4][0] == 1.10 and "PG"  in mst[4] and "XON" in mst[4])





