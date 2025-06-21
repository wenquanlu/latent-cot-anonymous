import matplotlib.pyplot as plt
import pickle
import re

from matplotlib import rcParams

rcParams['axes.linewidth'] = 1.2  # All subplot borders (left, bottom, top, right)
rcParams['lines.linewidth'] = 2.0

def is_signed_numeric(s):
    return bool(re.fullmatch(r'(-\d+|\d+|-)', s))

# --- Load and process CODA lens data ---
coda_data = pickle.load(open("coda_arithmetic_results_top5_16.pkl", "rb"))
freq_coda = [0 for _ in range(64)]
for row in coda_data:
    for i in range(64):
        tokens = row[i]
        count = sum(1 for token in tokens if is_signed_numeric(token.strip()))
        freq_coda[i] += count
avg_coda = [f / 500 for f in freq_coda]

blocks_coda = [[] for _ in range(4)]
for i in range(64):
    blocks_coda[i % 4].append(avg_coda[i])

# --- Load and process Logit lens data ---
logit_data = pickle.load(open("arithmetic_results_top5_16.pkl", "rb"))
freq_logit = [0 for _ in range(68)]
for row in logit_data:
    for i in range(68):
        tokens = row[i]
        count = sum(1 for token in tokens if is_signed_numeric(token.strip()))
        freq_logit[i] += count
avg_logit = [f / 500 for f in freq_logit]

blocks_logit = [[] for _ in range(4)]
for i in range(64):
    blocks_logit[i % 4].append(avg_logit[i + 2])

# --- Create the shared figure with two subplots ---
fig, axes = plt.subplots(1, 2, figsize=(10, 5.5), sharey=True)

# Shared legend labels
labels = ["$R_1$ Block", "$R_2$ Block", "$R_3$ Block", "$R_4$ Block"]
colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']

# CODA subplot
for b, color, label in zip(blocks_coda, colors, labels):
    axes[1].plot(range(1, 17), b, label=label, linewidth=2.5, color=color)
axes[1].set_title("Coda Lens", fontsize=24)
axes[1].set_xlabel("Recurrence Step", fontsize=21)
axes[1].tick_params(labelsize=20)

# Logit subplot
for b, color, label in zip(blocks_logit, colors, labels):
    axes[0].plot(range(1, 17), b, linewidth=2.5, color=color)
axes[0].set_title("Logit Lens", fontsize=24)
axes[0].set_xlabel("Recurrence Step", fontsize=21)
axes[0].set_ylabel("Proportion", fontsize=21)
axes[0].tick_params(labelsize=20)
import matplotlib.ticker as ticker

for ax in axes:
    ax.xaxis.set_major_locator(ticker.MultipleLocator(2))
# --- Shared legend above plots ---
legend = fig.legend(labels, loc='upper center', ncol=4, fontsize=21, frameon=True, handlelength=1.5, borderpad=0.3)
legend.get_frame().set_linewidth(2.5)
# Adjust layout
plt.tight_layout(rect=[0, 0, 1, 0.88])  # Leave space for legend
plt.savefig("graphs/arithmetic_numeric_combined.png")