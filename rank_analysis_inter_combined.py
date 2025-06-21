import pickle
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['axes.linewidth'] = 1.2  # All subplot borders (left, bottom, top, right)
rcParams['lines.linewidth'] = 2.5

# --- Load Logit Lens data ---
logit_inter = pickle.load(open("cot_weights/arithmetic_inter_rank_results_16.pkl", "rb"))
logit_correct = pickle.load(open("cot_weights/arithmetic_correct_rank_results_16.pkl", "rb"))
logit_the = pickle.load(open("cot_weights/arithmetic_the_rank_results_16.pkl", "rb"))

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

# Logit lens: only block R3 (i%4 == 2) and +2 offset
logit_len = 68
logit_inter_avg = compute_average_ranks(logit_inter, logit_len)
logit_correct_avg = compute_average_ranks(logit_correct, logit_len)
logit_the_avg = compute_average_ranks(logit_the, logit_len)
logit_block3 = [logit_inter_avg[i + 2] for i in range(64) if i % 4 == 2]
logit_block3_correct = [logit_correct_avg[i + 2] for i in range(64) if i % 4 == 2]
logit_block3_the = [logit_the_avg[i + 2] for i in range(64) if i % 4 == 2]

# Coda lens: only block R4 (i%4 == 3)
coda_len = 64
coda_inter_avg = compute_average_ranks(coda_inter, coda_len)
coda_correct_avg = compute_average_ranks(coda_correct, coda_len)
coda_the_avg = compute_average_ranks(coda_the, coda_len)
coda_block4 = [coda_inter_avg[i] for i in range(64) if i % 4 == 3]
coda_block4_correct = [coda_correct_avg[i] for i in range(64) if i % 4 == 3]
coda_block4_the = [coda_the_avg[i] for i in range(64) if i % 4 == 3]

# --- Plotting side-by-side ---
fig, axes = plt.subplots(1, 2, figsize=(10, 5.5), sharey=True)

# Plot Logit Lens
axes[0].plot(range(1, 17), logit_block3, label="Intermediate Token")
axes[0].plot(range(1, 17), logit_block3_correct, label="Final Token")
axes[0].plot(range(1, 17), logit_block3_the, label="Random Token: 'the'")
axes[0].set_title("Logit Lens at $R_3$", fontsize=24)
axes[0].set_xlabel("Recurrent Steps", fontsize=21)
axes[0].set_ylabel("Rank", fontsize=21)
axes[0].set_yscale("log")
axes[0].tick_params(labelsize=20)

# Plot Coda Lens
axes[1].plot(range(1, 17), coda_block4, label="Intermediate Token")
axes[1].plot(range(1, 17), coda_block4_correct, label="Final Token")
axes[1].plot(range(1, 17), coda_block4_the, label="Random Token: 'the'")
axes[1].set_title("Coda Lens at $R_4$", fontsize=24)
axes[1].set_xlabel("Recurrent Steps", fontsize=21)
axes[1].set_yscale("log")
axes[1].tick_params(labelsize=20)
import matplotlib.ticker as ticker

for ax in axes:
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
# Shared Legend
handles, labels = axes[0].get_legend_handles_labels()
legend = fig.legend(handles, labels, loc='upper center', ncol=3, fontsize=21, frameon=True, handlelength=1.0, borderpad=0.3, columnspacing=0.6)
legend.get_frame().set_linewidth(2.5)
plt.tight_layout(rect=[0, 0, 1, 0.90])  # leave space for top legend
plt.savefig("graphs/inter_block.png")
