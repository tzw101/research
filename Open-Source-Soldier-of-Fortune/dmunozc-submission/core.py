"""This file implements MST on a distance metric for a number of stocks.

It follows the methedology as seen in https://arxiv.org/abs/1703.00485
It calculates a distance metric based on multiple assets log returns. Based
on this metric, a minimum spanning tree (MST) is created.
"""

import yfinance as yf
import random
import numpy as np
from graph import Graph
import networkx as nx
import matplotlib.pyplot as plt
import argparse

DEFAULT_ASSETS = [
    "WETF", "BIG", "STNG", "CSCO", "XLY", "LABD", "MRVL", "STZ", "COG", "KO",
    "SGMO", "GOOS", "FDX", "TRGP", "ILMN", "HOG", "SIG", "ET", "MDT", "CHL",
    "S", "MELI", "BSX", "WMT", "MAR", "T", "KRE", "CNC", "EAF", "EQT", "SE",
    "ITCI", "GME", "FAZ", "FIVE", "CARS", "IMMU", "CHWY", "RVLV", "MOMO",
    "SPY", "XLP", "GD", "MU", "LMT", "GOLD", "PPL", "QD", "RSX", "HTHT"
]


def asset_prices(assets, start_date="2015-01-01", end_date="2016-01-01"):
    """Returns asset prices from yahoo finance.

    Parameters
    ----------
    assets: list
        list of assets
    start_date: string
        start date of asset prices
    end_date: astring
        end date of asset prices

    Returns
    -------
    prices: pandas dataframe
    """
    return yf.download(
        tickers=" ".join(assets), start=start_date, end=end_date
    )


def draw_graph(graph, ax, graph_layout):
    """Helper function to draw nx graph.

    Parameters
    ----------
    graph: nx graph
    ax: matplotlib axes
    graph_layout: nx layout
    """
    nx.draw_networkx(graph, graph_layout, node_size=1000, ax=ax)
    labels = nx.get_edge_attributes(graph, "weight")
    nx.draw_networkx_edge_labels(
        graph, graph_layout, edge_labels=labels, font_size=12, ax=ax
    )


def main(assets, start_date, end_date, plot_original=False):
    """Main function to draw MST from assets.

    Parameters
    ----------
    assets: list
        list of assets
    start_date: string
        start date of asset prices
    end_date: astring
        end date of asset prices
    """
    assets = asset_prices(assets, start_date, end_date)
    close_prices = assets["Close"]
    # For reference, log(x) - log(y) == log(x/y)
    log_returns = np.log(close_prices / close_prices.shift(1))[1:]
    correlations = log_returns.corr()
    distances = np.sqrt(2 * (1 - correlations))
    # Build graph.
    graph = Graph(distances)
    edges = graph.weighted_edges()
    nodes = graph.vertices()
    # Initialize nx graph plotting
    graph_layout = None
    if plot_original:
        nxGraph = nx.Graph()
        nxGraph.add_nodes_from(nodes)
        graph_layout = nx.spring_layout(nxGraph)
        # Draw network with all edges.
        # Add weighted edges to nx graph
        for edge in edges:
            nxGraph.add_edge(edge[1], edge[2], weight=round(edge[0], 3))
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(14.5, 10.5)
        draw_graph(nxGraph, ax, graph_layout)
        ax.set_title("Original Network Graph", fontsize=24)
    mst = graph.kruskal_mst()
    # Draw mst network
    nxGraph = nx.Graph()
    nxGraph.add_nodes_from(nodes)
    for edge in mst:
        nxGraph.add_edge(edge[1], edge[2], weight=round(edge[0], 3))
    if not graph_layout:
        graph_layout = nx.spring_layout(nxGraph)
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(10.5, 10.5)
    draw_graph(nxGraph, ax, graph_layout)
    ax.set_title("MST Network Graph", fontsize=24)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-a",
        dest="assets",
        nargs="+",
        help="list of assets, separated by a space",
        default=DEFAULT_ASSETS,
    )
    parser.add_argument(
        "-start",
        dest="start_date",
        help="start date of asset prices",
        default="2019-01-01",
    )
    parser.add_argument(
        "-end",
        dest="end_date",
        help="end date of asset prices",
        default="2020-01-01",
    )
    parser.add_argument(
        "-po",
        dest="plot_original",
        help="plot original full assets graph",
        default=False,
    )
    parser.add_argument(
        "-rs", dest="rseed", help="random seed", default=None,
    )
    args = parser.parse_args()
    if args.rseed:
        random.seed(int(args.rseed))
    main(args.assets, args.start_date, args.end_date, args.plot_original)
