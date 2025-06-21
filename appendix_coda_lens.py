import pickle
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['axes.linewidth'] = 1.2  # All subplot borders (left, bottom, top, right)
rcParams['lines.linewidth'] = 2.3

# --- Load Coda Lens data ---
coda_inter = pickle.load(open("cot_weights/coda_arithmetic_inter_rank_16.pkl", "rb"))
coda_correct = pickle.load(open("cot_weights/coda_arithmetic_correct_rank_16.pkl", "rb"))
coda_the = pickle.load(open("cot_weights/coda_arithmetic_the_rank_16.pkl", "rb"))

def compute_average_ranks(data, L):
    results = [0 for _ in range(L)]
    for row in data:
        for i in range(L):
            results[i] += row[i]
    return [r / len(data) for r in results]

# Coda lens: only block R4 (i%4 == 3)
coda_len = 64
coda_inter_avg = compute_average_ranks(coda_inter, coda_len)
coda_correct_avg = compute_average_ranks(coda_correct, coda_len)
coda_the_avg = compute_average_ranks(coda_the, coda_len)
coda_block1 = [coda_inter_avg[i] for i in range(64) if i % 4 == 0]
coda_block1_correct = [coda_correct_avg[i] for i in range(64) if i % 4 == 0]
coda_block1_the = [coda_the_avg[i] for i in range(64) if i % 4 == 0]

coda_block2 = [coda_inter_avg[i] for i in range(64) if i % 4 == 1]
coda_block2_correct = [coda_correct_avg[i] for i in range(64) if i % 4 == 1]
coda_block2_the = [coda_the_avg[i] for i in range(64) if i % 4 == 1]

coda_block3 = [coda_inter_avg[i] for i in range(64) if i % 4 == 2]
coda_block3_correct = [coda_correct_avg[i] for i in range(64) if i % 4 == 2]
coda_block3_the = [coda_the_avg[i] for i in range(64) if i % 4 == 2]

# --- Plotting side-by-side ---
fig, axes = plt.subplots(1, 3, figsize=(15, 5.5), sharey=True)

# Plot Coda Lens
axes[0].plot(range(1, 17), coda_block1, label="Intermediate Token")
axes[0].plot(range(1, 17), coda_block1_correct, label="Final Token")
axes[0].plot(range(1, 17), coda_block1_the, label="Random Token: 'the'")
axes[0].set_title("Coda Lens at $R_1$", fontsize=24)
axes[0].set_xlabel("Recurrent Steps", fontsize=21)
axes[0].set_ylabel("Rank", fontsize=21)
axes[0].set_yscale("log")
axes[0].tick_params(labelsize=20)

# Plot Coda Lens
axes[1].plot(range(1, 17), coda_block2, label="Intermediate Token")
axes[1].plot(range(1, 17), coda_block2_correct, label="Final Token")
axes[1].plot(range(1, 17), coda_block2_the, label="Random Token: 'the'")
axes[1].set_title("Coda Lens at $R_2$", fontsize=24)
axes[1].set_xlabel("Recurrent Steps", fontsize=21)
axes[1].set_ylabel("Rank", fontsize=21)
axes[1].set_yscale("log")
axes[1].tick_params(labelsize=20)

axes[2].plot(range(1, 17), coda_block3, label="Intermediate Token")
axes[2].plot(range(1, 17), coda_block3_correct, label="Final Token")
axes[2].plot(range(1, 17), coda_block3_the, label="Random Token: 'the'")
axes[2].set_title("Coda Lens at $R_3$", fontsize=24)
axes[2].set_xlabel("Recurrent Steps", fontsize=21)
axes[2].set_ylabel("Rank", fontsize=21)
axes[2].set_yscale("log")
axes[2].tick_params(labelsize=20)
import matplotlib.ticker as ticker

for ax in axes:
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
# Shared Legend
handles, labels = axes[0].get_legend_handles_labels()
legend = fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=21, frameon=True, handlelength=1.0, borderpad=0.3, columnspacing=0.6)
legend.get_frame().set_linewidth(2.5)
plt.tight_layout(rect=[0, 0, 1, 0.90])  # leave space for top legend
plt.savefig("graphs/appendix_coda.png")
