"""Provides an implementation of a Minimum Spanning Tree.

This class accepts a distance dataframe as an input to form a MST using Prim's
algorithm.
How to use distance matrix to produce a MST is seen in:
Introduction to econophysics: correlations and complexity in finance,
chapter 13.
"""


class MST:

    def __init__(self, distances):
        self.distances = distances



