"""This file showcases MST on a distance metric for a number of stocks.

It follows the methodology as seen in https://arxiv.org/abs/1703.00485
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
from scipy.cluster.hierarchy import dendrogram, distance, linkage

DEFAULT_ASSETS = [
    "AAPL", "MSFT", "AMZN", "FB", "GOOG", "JNJ", "V", "PG", "JPM", "UNH",
    "TSLA", "INTC", "NVDA", "NFLX", "ADBE", "PYPL", "CSCO", "PEP", "SPY",
    "HD", "EWA", "MCD", "BA", "MMM", "CAT", "WMT", "IBM", "TRV", "DIS",
    "NKE", "GLD", "SLV", "SPLV", "TLT", "SHY", "SHYD", "IEF", "SHV", "JNK",
    "LQD", "GS", "EWZ", "EWI", "EWJ", "EWG", "EWW", "THD", "EWU", "RSX", "UUP"
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


def draw_graph(vertices, edges, ax, graph_layout=None, title=""):
    """Helper function to draw nx graph.

    Parameters
    ----------
    vertices: list
        contains the vertices of a graph
    edges: list of tuples
        contains edges of a graph
    ax: matplotlib axes
    graph_layout: nx layout
    title: string
        title of plot
    """
    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(vertices)
    for edge in edges:
        nx_graph.add_edge(edge[1], edge[2], weight=round(edge[0], 3))
    if not graph_layout:
        graph_layout = nx.spring_layout
    # Make sure graph_layout is a function type.
    assert hasattr(graph_layout, "__call__")
    graph_layout = graph_layout(nx_graph)
    nx.draw_networkx(nx_graph, graph_layout, node_size=1000, ax=ax, alpha=0.65)
    ax.set_title(title, fontsize=24)


def main(assets, start_date, end_date, plot_original=False):
    """Main function to draw MST from assets.

    Parameters
    ----------
    assets: list
        list of assets
    start_date: string
        start date of asset prices
    end_date: string
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
    # Plot all nodes and edges.
    if plot_original:
        fig, ax = plt.subplots(1, 1)
        fig.set_size_inches(14.5, 10.5)
        draw_graph(
            nodes,
            edges,
            ax,
            graph_layout=nx.spring_layout,
            title="Original Network Graph",
        )
    # Plot MST.
    mst = graph.kruskal_mst()
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(14.5, 10.5)
    draw_graph(
        nodes,
        mst,
        ax,
        graph_layout=nx.spring_layout,
        title="MST Network Graph",
    )
    # Plot dendogram.
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(20.5, 8.5)
    pdist = distance.pdist(distances.values)
    link = linkage(pdist, method="complete")
    dendrogram(link, labels=distances.columns)
    ax.set_title("Dendogram of MST", fontsize=24)
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
        default="2020-01-01",
    )
    parser.add_argument(
        "-end",
        dest="end_date",
        help="end date of asset prices",
        default="2020-02-18",
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
