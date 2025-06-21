import json
import pickle

import matplotlib.pyplot as plot

arithmetic_rank = pickle.load(open("coda_arithmetic_rank_16.pkl", "rb"))

results = [0 for i in range(64)]

for row in arithmetic_rank:
    for i in range(64):
        results[i] += row[i]
    pass

average_ranks = [result/100 for result in results]

plot.plot(range(3, 67), average_ranks)
#plot.title("Coda Lens Rank Trajectory of Final \nPredicted Token over Unrolled Layers", fontsize=17)
plot.xlabel("Block Number", fontsize=20)
plot.ylabel("Rank", fontsize=20)
plot.xticks(fontsize=17)
plot.yticks(fontsize=17)
plot.yscale("log")
plot.title("Coda Lens", fontsize=20)
plot.tight_layout()
plot.savefig("graphs/coda_lens_unrolled_rank.png", bbox_inches = "tight")