# %%
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# %%
def get_method_name(row):
    if row.average:
        if row.binarize:
            return "Average Connectivity"
        else:
            return "Average Edge Weight"
    else:
        if row.binarize:
            return "Multivariate Binary"
        else:
            return "Multivariate Weighted"


# %%
df = pd.read_csv("results/block_simulation_2.csv")
df["method"] = df.apply(get_method_name, axis="columns")
df = df.drop(["binarize", "average"], axis="columns")
df["reject"] = df["pvalue"] < 0.05
df.head()

# %%
sns.lineplot(
    data=df.query("distribution == 'same_mean'"),
    x="sample_size",
    y="reject",
    hue="method",
)
plt.show()

# %%
fig, axs = plt.subplots(ncols=3, sharex=True, sharey=False, figsize=(8, 3.5))

sns.lineplot(
    data=df.query("distribution == 'equal'"),
    x="sample_size",
    y="reject",
    hue="method",
    legend=False,
    ax=axs[0],
)
axs[0].set_box_aspect(1)
axs[0].set(
    xlabel="Sample Size",
    ylabel="False Positive Rate",
    title="Same Distribution",
    ylim=(-0.05, 1.05),
)

sns.lineplot(
    data=df.query("distribution == 'same_mean'"),
    x="sample_size",
    y="reject",
    hue="method",
    legend=True,
    ax=axs[1],
)
axs[1].set_box_aspect(1)
axs[1].set(
    xlabel="Sample Size",
    ylabel="True Positive Rate",
    title="Same Mean",
    ylim=(-0.05, 1.05),
)

sns.lineplot(
    data=df.query("distribution == 'diff_mean'"),
    x="sample_size",
    y="reject",
    hue="method",
    legend=False,
    ax=axs[2],
)
axs[2].set_box_aspect(1)
axs[2].set(
    xlabel="Sample Size",
    ylabel="True Positive Rate",
    title="Different Mean",
    ylim=(-0.05, 1.05),
)

plt.tight_layout()
axs[1].legend(
    loc="upper center",
    bbox_to_anchor=(0.5, -0.25),
    fancybox=True,
    shadow=True,
    ncol=2,
)
plt.savefig("community_sim.pdf", bbox_inches="tight")

# %%
