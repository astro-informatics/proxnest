import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

res_dir = "mywork/Gaussian Benchmark/Residuals2/2024_03_06"

dirs = [f for f in os.listdir(res_dir) if os.path.isdir(os.path.join(res_dir,f))]
os.chdir(res_dir)

df = pd.DataFrame()
n = 1
for i in dirs:
    if df.empty:
        df = pd.read_csv(i+"/residuals.csv", names=("dimension", f"run {n}"))
    else:
        res = pd.read_csv(i+"/residuals.csv", names=("dimension", f"run {n}"))
        df = pd.merge(df,res)
    n+=1

df.set_index('dimension', inplace=True)
df = df.T
df.to_csv("dataframe.csv")

avgs = np.mean(df.to_numpy(), axis=0)
stds = np.std(df.to_numpy(), axis=0)

plt.figure(dpi=300)
plt.errorbar(df.columns.values, avgs, yerr=stds, marker="x", capsize=2, 
             linewidth=0.5, elinewidth=0.5, color="black", ecolor="red")
plt.xlabel("Dimensions")
plt.ylabel("Residual")
plt.savefig("plot")
