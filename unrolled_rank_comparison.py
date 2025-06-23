import pickle
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['axes.linewidth'] = 1.2  # All subplot borders (left, bottom, top, right)
rcParams['lines.linewidth'] = 2.0

# Load data
logit_data = pickle.load(open("cot_weights/arithmetic_rank_results_16.pkl", "rb"))
coda_data = pickle.load(open("cot_weights/coda_arithmetic_rank_16_with_prelude.pkl", "rb"))

# Compute averages
def compute_average(data, length):
    results = [0 for _ in range(length)]
    for row in data:
        for i in range(length):
            results[i] += row[i]
    return [result / len(data) for result in results]

logit_avg = compute_average(logit_data, 68)
print(len(coda_data[0]))
coda_avg = compute_average(coda_data, 66)

# Plot side by side
fig, axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)  # 1 row, 2 columns

# Plot Logit Lens
axes[0].plot(range(1, 69), logit_avg, marker='o')
axes[0].set_title("Logit Lens from $s_1$ to $s_{68}$", fontsize=24)
axes[0].set_xlabel("Block Number", fontsize=21)
axes[0].set_ylabel("Rank", fontsize=21)
axes[0].set_yscale("log")
axes[0].tick_params(axis='both', labelsize=20)

# Plot Coda Lens
axes[1].plot(range(1, 67), coda_avg, marker='o')
axes[1].set_title("Coda Lens from $s_1$ to $s_{66}$", fontsize=24)
axes[1].set_xlabel("Block Number", fontsize=21)
axes[1].set_yscale("log")
axes[1].tick_params(axis='both', labelsize=20)
print(coda_avg)

plt.tight_layout()
plt.savefig("graphs/lens_comparison_unrolled_rank_with_prelude.png", bbox_inches="tight")
