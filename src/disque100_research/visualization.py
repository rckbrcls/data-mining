from __future__ import annotations

import textwrap
from typing import Any

import networkx as nx
import numpy as np


ENTITY_COLORS = {
    "report": "#22313A",
    "domain": "#315A72",
    "category": "#4D7F92",
    "subtype": "#79A5AC",
    "detail": "#A9C4C1",
    "age_group": "#B8793E",
    "relationship": "#8C5E58",
    "setting": "#6D7762",
    "month": "#596B8A",
}


def scale_values(values: list[float], minimum: float, maximum: float) -> list[float]:
    numeric = np.asarray(values, dtype=float)
    if numeric.size == 0:
        return []
    if numeric.max() == numeric.min():
        return np.full(numeric.shape, (minimum + maximum) / 2).tolist()
    scaled = (numeric - numeric.min()) / (numeric.max() - numeric.min())
    return (minimum + scaled * (maximum - minimum)).tolist()


def wrapped_labels(graph: nx.Graph, width: int = 22) -> dict[Any, str]:
    return {
        node: textwrap.fill(
            str(graph.nodes[node].get("display_label", node)),
            width=width,
            break_long_words=False,
        )
        for node in graph.nodes
    }


def assert_positive_edge_weights(graph: nx.Graph) -> None:
    assert graph.number_of_nodes() > 0
    assert graph.number_of_edges() > 0
    assert all(
        float(data.get("weight", 1)) > 0
        for _, _, data in graph.edges(data=True)
    )


def deterministic_spring_layout(graph: nx.Graph, seed: int = 42) -> dict[Any, Any]:
    return nx.spring_layout(graph, seed=seed, weight="weight")
