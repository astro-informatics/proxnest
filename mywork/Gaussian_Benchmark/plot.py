import numpy as np
import matplotlib as plt

save_dir = "/Users/henry/Work/PhD/Project/Repositories/proxnest/mywork/Gaussian_Benchmark/test_original/2024_04_15/17_28_10"

mean_predictions = np.mean(array_predictions, axis=0)
std_dev_predictions = np.std(array_predictions, axis=0)

plt.rcParams["mathtext.fontset"] = "stix"
plt.figure(dpi=200)
plt.errorbar(
    x=dimensions,
    y=mean_predictions[:, 2],
    yerr=std_dev_predictions[:, 2],
    color="tomato",
    marker="x",
    linewidth=0.5,
    markersize=2,
    label="ProxNest",
)
plt.plot(
    dimensions,
    mean_predictions[:, 1],
    color="black",
    marker="o",
    linewidth=0.5,
    markersize=2,
    label="Ground truth",
)
plt.ylim(0, np.max(mean_predictions[:, 1:]) + 10)
plt.xlim(0, 1000)
plt.xlabel("Dimensions")
plt.ylabel(r"$\log (V \times \mathcal{Z})$")
plt.title(hd, fontsize=7)
plt.savefig(save_dir + "plot1.pdf")
plt.close()