"""
Visualize Duke mice connectomes with Circos:
http://mkweb.bcgsc.ca/tableviewer/visualize/
"""

# %%
from collections import namedtuple
from itertools import combinations_with_replacement

import numpy as np
import pandas as pd
from graspologic.datasets import load_mice

# %%
mice = load_mice()
Community = namedtuple("Community", ["i", "j", "index"])

# %%
def community_index_iterator(blocks):
    indices = []
    for index, row in blocks.iterrows():
        p = Community(row.i, row.j, index)
        indices.append(p)
    for block_1, block_2 in combinations_with_replacement(indices, 2):
        yield block_1, block_2


# %%
def aggregate(graph, blocks):
    """Sum edgeweights in each community."""

    n_communities = len(blocks)
    agg = np.zeros((n_communities, n_communities))

    for block_1, block_2 in community_index_iterator(blocks):
        subgraph = graph[block_1.i : block_1.j, block_2.i : block_2.j]
        agg[block_1.index, block_2.index] = np.sum(subgraph)
        agg[block_2.index, block_1.index] = np.sum(subgraph)

    return agg


# %%
agg_graphs = dict()

for graph, label in zip(mice.graphs, mice.labels):

    agg = aggregate(graph, mice.blocks)
    if label in agg_graphs:
        agg_graphs[label] += agg
    else:
        agg_graphs[label] = agg

agg_graphs = {label: np.round(agg / 8) for label, agg in agg_graphs.items()}

# map_min = min(map(np.min, agg_graphs.values()))
# agg_graphs = agg_graphs = {
#     label: np.round((agg / map_min)) for label, agg in agg_graphs.items()
# }

# %%
def fix_structure_name(name):
    name = name.split("_")
    name = [part.capitalize() for part in name]
    return " ".join(name)


# %%
block_names = []
for _, row in mice.blocks.iterrows():
    # name = fix_structure_name(row.block) + f" ({row.hemisphere.upper()})"
    name = "_".join([row.block, row.hemisphere])
    block_names.append(name)

block_names
# %%
for label, graph in agg_graphs.items():
    df = pd.DataFrame(graph, columns=block_names, index=block_names)
    df.to_csv(f"{label}.csv", sep="\t")

# %%

# %%
