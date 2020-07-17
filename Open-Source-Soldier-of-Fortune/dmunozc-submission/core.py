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

ASSETS = ["MSFT", "FB", "AAPL", "NFLX", "GOOG", "SPY", "IWM", "GLD", "TLT",
          "TIPS", "SHY"]
random.seed(69)


def asset_prices(N, start_date="2015-01-01", end_date="2016-01-01"):
    assets = random.sample(ASSETS, N)
    return yf.download(
        tickers=" ".join(assets), start=start_date, end=end_date
    )

def draw_graph(G, ax, pos):
     # pos = nx.nx_agraph.graphviz_layout(G)
    nx.draw_networkx(G, pos, node_size=800, ax=ax)
    labels = nx.get_edge_attributes(G, 'weight')
    nx.draw_networkx_edge_labels(G, pos, edge_labels=labels, font_size=12,
                                 ax=ax)

def main():
    assets = asset_prices(len(ASSETS))
    close_prices = assets["Close"]
    # log(5) - log(2) == log(5/2)
    log_returns = np.log(close_prices / close_prices.shift(1))[1:]
    correlations = log_returns.corr()
    distances = np.sqrt(2 * (1 - correlations))
    # Correlation and distance matrices are symmetrical. Can ignore lower
    # diagonal.
    graph = Graph(distances)
    edges = graph.weighted_edges()
    nodes = graph.vertices()
    # mst = graph.mst()
    G = nx.Graph()

    G.add_nodes_from(nodes)
    pos = nx.kamada_kawai_layout(G)
    # Switch needed due to networkx asking weight to be last in tuple.
    new_edges = []
    for edge in edges:
        G.add_edge(edge[1], edge[2], weight=round(edge[0], 3))
    fig, ax = plt.subplots(1, 1)
    draw_graph(G, ax, pos)
    ax.set_title("Original Network")
    mst = graph.kruskal_mst()

    G = nx.Graph()

    G.add_nodes_from(nodes)
    # Switch needed due to networkx asking weight to be last in tuple.
    new_edges = []
    for edge in mst:
        G.add_edge(edge[1], edge[2], weight=round(edge[0], 3))
    fig, ax = plt.subplots(1, 1)
    draw_graph(G, ax, pos)
    ax.set_title("MST Network")



    # plt.savefig("temp.png",bbox_inches='tight')
    plt.show()
main()
