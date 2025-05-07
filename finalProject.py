"""
PageRank & Community Detection on the SNAP email-Eu-core network
===============================================================
Author: Michelle Lawson & Adriana Soldat  
Course: Linear Algebra – Final Project

*Loads the dataset, builds a weighted digraph, computes PageRank, explores
Laplacian communities, and offers live-update simulation tools.*

Place `email-Eu-core.txt` and `email-Eu-core-department-labels.txt` in the
same folder, then run:
    python pagerank_email_eu_core.py

(or import in a notebook for interactive experiments)
"""
from __future__ import annotations
import collections
import math
import pathlib
import sys
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parent
EDGE_FILE = ROOT / "email-Eu-core.txt"
LABEL_FILE = ROOT / "email-Eu-core-department-labels.txt"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(edge_path: pathlib.Path | str = EDGE_FILE,
                 label_path: pathlib.Path | str = LABEL_FILE) -> tuple[nx.DiGraph, Dict[int, int]]:
    """Return a weighted *directed* graph + department labels."""
    edge_path = pathlib.Path(edge_path)
    label_path = pathlib.Path(label_path)
    if not edge_path.exists() or not label_path.exists():
        print("❌  Dataset files missing – download from SNAP and retry.")
        sys.exit(1)

    G = nx.DiGraph()
    with edge_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            u, v = map(int, line.split())
            G.add_edge(u, v, weight=G[u][v]["weight"] + 1 if G.has_edge(u, v) else 1)

    labels: Dict[int, int] = {}
    with label_path.open() as f:
        for line in f:
            if not line.strip():
                continue
            node, dept = map(int, line.split())
            labels[node] = dept
            if node not in G:
                G.add_node(node)
    return G, labels

# ---------------------------------------------------------------------------
# PageRank helpers
# ---------------------------------------------------------------------------

def pagerank(G: nx.DiGraph, alpha: float = 0.85, tol: float = 1e-6,
             max_iter: int = 100) -> Dict[int, float]:
    return nx.pagerank(G, alpha=alpha, tol=tol, max_iter=max_iter, weight="weight")


def display_topk(pr: Dict[int, float], k: int = 10) -> None:
    import pandas as pd
    top = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:k]
    print(pd.DataFrame(top, columns=["node", "PageRank score"]).to_string(index=False))

# ---------------------------------------------------------------------------
# Laplacian zero-eigenvalue multiplicity
# ---------------------------------------------------------------------------

def laplacian_zero_eigenspace_dim(G: nx.Graph) -> int:
    H = G.to_undirected(as_view=False)
    A = nx.to_scipy_sparse_array(H, weight="weight", dtype=float, format="csr")
    degrees = np.ravel(A.sum(axis=1))
    L = np.diag(degrees) - A.toarray()
    eigvals = np.linalg.eigvalsh(L)
    return int(np.sum(np.isclose(eigvals, 0.0, atol=1e-8)))

# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

_cmap = plt.colormaps.get_cmap("coolwarm")  # new-style colormap accessor

def plot_graph(G: nx.DiGraph, pr: Dict[int, float],
               title: str = "PageRank heatmap") -> None:
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    pos = nx.spring_layout(G, seed=42)
    scores = np.array([pr.get(n, 0.0) for n in G.nodes()])
    if scores.max() > 0:
        scores = (scores - scores.min()) / (scores.max() - scores.min())

    nx.draw_networkx_edges(G, pos, alpha=0.3, arrows=True, width=0.5, ax=ax)
    nx.draw_networkx_nodes(G, pos, node_color=scores, cmap=_cmap,
                           node_size=80, ax=ax)
    ax.set_title(title)
    ax.axis("off")

    sm = plt.cm.ScalarMappable(cmap=_cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    plt.colorbar(sm, ax=ax, label="relative PageRank score")
    plt.tight_layout()
    plt.show()

# ---------------------------------------------------------------------------
# Simulation helpers
# ---------------------------------------------------------------------------

def send_email(G: nx.DiGraph, sender: int, recipient: int, n: int = 1) -> None:
    G.add_edge(sender, recipient, weight=G[sender][recipient]["weight"] + n if G.has_edge(sender, recipient) else n)


def simulate_messages(G: nx.DiGraph, interactions: List[Tuple[int, int]], n: int = 1,
                      alpha: float = 0.85) -> Dict[int, float]:
    for s, r in interactions:
        send_email(G, s, r, n)
    return pagerank(G, alpha=alpha)

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    G, labels = load_dataset()
    pr = pagerank(G)

    print("Top-10 influential nodes (initial):")
    display_topk(pr)
    print("\nLaplacian zero-eigenvalue multiplicity (undirected view):",
          laplacian_zero_eigenspace_dim(G))

    plot_graph(G, pr, "Initial PageRank heatmap")

    pair = (0, 42)
    print(f"\nSimulating 100 extra e-mails from {pair[0]} → {pair[1]} …")
    pr2 = simulate_messages(G, [pair], n=100)
    plot_graph(G, pr2, "After simulated messages")

if __name__ == "__main__":
    main()
