# %%
import csv
from itertools import product

import numpy as np
from hyppo.ksample import KSample
from scipy.stats import truncnorm
from skimage.filters import threshold_otsu
from tqdm import tqdm


# %%
def test(samples, labels, binarize, average):

    if binarize:
        threshold = threshold_otsu(samples)
        samples = samples > threshold

    samples = [samples[labels == group, :] for group in np.unique(labels)]

    if average:
        samples = [np.mean(sample, axis=1) for sample in samples]

    # Run MGC
    try:
        stat, pvalue, *_ = KSample("MGC").test(*samples, reps=10000, workers=-1)
    except ValueError:
        stat, pvalue = np.nan, 1
    return stat, pvalue


# %%
def twin_truncnorm(mu_1, sigma_1, mu_2, sigma_2, n_subjects, n_vertices=10):

    # Initialize distributions
    upper = 1
    lower = -1
    x1 = truncnorm(
        (lower - mu_1) / sigma_1, (upper - mu_1) / sigma_1, loc=mu_1, scale=sigma_1
    )
    x2 = truncnorm(
        (lower - mu_2) / sigma_2, (upper - mu_2) / sigma_2, loc=mu_2, scale=sigma_2
    )

    # Sample distributions
    samples = []
    labels = []
    for _ in range(int(n_subjects)):
        for label in range(2):
            labels.append(label)
            if label == 0:
                samples.append(x1.rvs(n_vertices))
            if label == 1:
                samples.append(x2.rvs(n_vertices))

    return np.array(samples), np.array(labels)


# %%
def data_generator(max_n_subjects, dist_params):
    data = dict()
    for distribution, parameters in dist_params.items():
        samples, labels = twin_truncnorm(n_subjects=max_n_subjects, **parameters)
        data[distribution] = (samples, labels)
    return data


def data_loader(n_subjects, distribution, data):
    n_subjects = int(n_subjects)
    samples, labels = data[distribution]
    return samples[:n_subjects, :], labels[:n_subjects]


# %%
def main(binarize, average, n_subjects, distribution, data):
    samples, labels = data_loader(n_subjects, distribution, data)
    stat, pvalue = test(samples, labels, binarize, average)

    n_groups = len(np.unique(labels))
    sample_size = n_groups * n_subjects

    return (
        sample_size,
        stat,
        pvalue,
    )


# %%
dist_params = {
    "equal": dict(mu_1=0, sigma_1=0.25, mu_2=0, sigma_2=0.25),
    "same_mean": dict(mu_1=0, sigma_1=0.25, mu_2=0, sigma_2=0.5),
    "diff_mean": dict(mu_1=0, sigma_1=0.25, mu_2=0.25, sigma_2=0.25),
}

# %%
binarize = [True, False]
average = [True, False]
n_subjects = np.linspace(10, 100, 10)
distributions = list(dist_params.keys())

n_iterations = range(20)
parameters = product(binarize, average, n_subjects, distributions, n_iterations)

data = data_generator(np.max(n_subjects), dist_params)

out = []
for binarize_, average_, n_subjects_, distribution, _ in tqdm(list(parameters)):
    sample_size, stat, pvalue = main(
        binarize_, average_, n_subjects_, distribution, data
    )
    out.append([binarize_, average_, distribution, sample_size, stat, pvalue])

# %%
filename = "results/block_simulation_4.csv"
columns = ["binarize", "average", "distribution", "sample_size", "stat", "pvalue"]

with open(filename, "w") as outfile:
    writer = csv.writer(outfile)
    writer.writerow(columns)
    writer.writerows(out)
