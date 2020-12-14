# %%
import pandas as pd
from graspologic.datasets import load_mice

# %%
mice = load_mice()

# %%
# Replace ROI indices with actual names
def lookup_roi_name(index):
    hemisphere = "R" if index // 166 else "L"
    index = index % 166
    roi_name = mice.atlas.query(f"ROI == {index+1}")["Structure"].item()
    return f"{roi_name} ({hemisphere})"


# %%
# Read nonpar results
df = pd.read_csv("nonpar_manova.csv")
df.columns = ["ROI", "statistic", "pvalue"]

# Sort dataframe
df.sort_values(by="pvalue", inplace=True, ignore_index=True)
pvalue_rank = df["pvalue"].rank(ascending=False, method="max")
df["holm_pvalue"] = df["pvalue"].multiply(pvalue_rank)
df["holm_pvalue"] = df["holm_pvalue"].apply(lambda pval: 1 if pval > 1 else pval)

# Rename ROIs
df["ROI"] = df["ROI"].apply(lookup_roi_name)

# Print significant vertices
nonpar_sig = df.query("holm_pvalue <= 0.05")

# %%
# Read parametric results
df1 = pd.read_csv("parametric_manova.csv")
df1.drop(columns="order.p", inplace=True)
df1.columns = ["ROI", "pvalue"]

# Sort dataframe
df1.sort_values(by="pvalue", inplace=True, ignore_index=True)
pvalue_rank = df1["pvalue"].rank(ascending=False, method="max")
df1["holm_pvalue"] = df1["pvalue"].multiply(pvalue_rank)
df1["holm_pvalue"] = df1["holm_pvalue"].apply(lambda pval: 1 if pval > 1 else pval)

# Rename ROIs
df1["ROI"] -= 1
df1["ROI"] = df1["ROI"].apply(lookup_roi_name)

# Print significant vertices
par_sig = df1.query("holm_pvalue <= 0.05")

# %%
nonpar_rois = set(nonpar_sig["ROI"])
par_rois = set(par_sig["ROI"])

len(nonpar_rois.intersection(par_rois))
# %%
