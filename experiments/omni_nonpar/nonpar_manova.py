# %%
import csv

import numpy as np
from graspologic.datasets import load_mice
from graspologic.embed import OmnibusEmbed
from hyppo.ksample import KSample
from tqdm import tqdm

# %%
# Load the full mouse dataset
mice = load_mice()

# Stack all adjacency matrices in a 3D numpy array
graphs = np.array(mice.graphs)

# Get sample parameters
n_subjects = mice.meta["n_subjects"]
n_vertices = mice.meta["n_vertices"]

# %%
# Jointly embed graphs using omnibus embedding
embedder = OmnibusEmbed()
omni_embedding = embedder.fit_transform(graphs)
print(omni_embedding.shape)

# %%
def vertex_pval(vertex, embedding, labels):

    # Get the embedding of the i-th vertex for
    samples = [embedding[labels == group, vertex, :] for group in np.unique(labels)]

    # Calculate the p-value for the i-th vertex
    statistic, pvalue, _ = KSample("MGC").test(*samples, reps=1000000, workers=-1)

    return statistic, pvalue


# %%
out = []
for vertex in tqdm(range(n_vertices)):
    statistic, pvalue = vertex_pval(vertex, omni_embedding, mice.labels)
    out.append([vertex, statistic, pvalue])

# %%
filename = "nonpar_manova.csv"
columns = ["ROI", "stat", "pval"]

with open(filename, "w") as outfile:
    writer = csv.writer(outfile)
    writer.writerow(columns)
