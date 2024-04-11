import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

res_dir = "PosFalse3/2024_03_07"

dirs = [f for f in os.listdir(res_dir) if os.path.isdir(os.path.join(res_dir, f))]
os.chdir(res_dir)

df = pd.DataFrame()
n = 1
for i in dirs:
    if df.empty:
        df = pd.read_csv(
            i + "/predictions.csv",
            names=("dimension", "ground truth", f"predictions {n}"),
        )
    else:
        preds = pd.read_csv(
            i + "/predictions.csv",
            names=("dimension", "ground truth", f"predictions {n}"),
        )
        df = pd.merge(df, preds)
    n += 1

df.set_index("dimension", inplace=True)
df = df.T
df.to_csv("Predictions.csv")

dims = df.columns.values
df = df.to_numpy()

df[0], df[1] = df[1], df[0].copy()

avgs = np.mean(df[1:], axis=0)
stds = np.std(df[1:], axis=0)

plt.figure(dpi=300)
plt.errorbar(
    dims,
    avgs,
    yerr=stds,
    marker="x",
    capsize=2,
    linewidth=1,
    elinewidth=0.5,
    color="red",
    ecolor="blue",
)
plt.plot(dims, df[0], linewidth=1, color="black")
plt.ylim(-10, 90)
plt.xlim(-5, 210)
plt.xlabel("Dimensions")
plt.ylabel(r"$\log (V \times \mathcal{Z})$")
plt.savefig("Avg NS Estimates")
