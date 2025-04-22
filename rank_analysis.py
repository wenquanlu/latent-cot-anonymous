import pickle
import matplotlib.pyplot as plot

arithmetic_rank = pickle.load(open("arithmetic_rank_results_16.pkl", "rb"))

results = [0 for i in range(68)]

for row in arithmetic_rank:
    for i in range(68):
        results[i] += row[i]
    pass

average_ranks = [result/100 for result in results]

block1_recurrences = []
block2_recurrences = []
block3_recurrences = []
block4_recurrences = []

for i in range(64):
    if i % 4 == 0:
        block1_recurrences.append(average_ranks[i + 2])
    elif i%4 == 1:
        block2_recurrences.append(average_ranks[i + 2])
    elif i%4 == 2:
        block3_recurrences.append(average_ranks[i + 2])
    elif i%4 == 3:
        block4_recurrences.append(average_ranks[i + 2])

plot.plot(range(1, 17), block1_recurrences, label="Recurrent Block 1")
plot.plot(range(1, 17), block2_recurrences, label="Recurrent Block 2")
plot.plot(range(1, 17), block3_recurrences, label="Recurrent Block 3")
plot.plot(range(1, 17), block4_recurrences, label="Recurrent Block 4")
plot.yscale("log")
plot.legend()
plot.xlabel("Recurrence")
plot.ylabel("Rank")
plot.title("Logit Lens Rank Trajectory of Final Predicted Token Over Recurrences")
plot.savefig("graphs/arithmetic_rank_logit.png")


# import pickle
# import matplotlib.pyplot as plot

# arithmetic_rank = pickle.load(open("coda_arithmetic_rank_16.pkl", "rb"))

# results = [0 for i in range(64)]

# for row in arithmetic_rank:
#     for i in range(64):
#         results[i] += row[i]
#     pass

# average_ranks = [result/100 for result in results]

# block1_recurrences = []
# block2_recurrences = []
# block3_recurrences = []
# block4_recurrences = []

# for i in range(64):
#     if i % 4 == 0:
#         block1_recurrences.append(average_ranks[i])
#     elif i%4 == 1:
#         block2_recurrences.append(average_ranks[i])
#     elif i%4 == 2:
#         block3_recurrences.append(average_ranks[i])
#     elif i%4 == 3:
#         block4_recurrences.append(average_ranks[i])

# plot.plot(range(1, 17), block1_recurrences, label="Recurrent Block 1")
# plot.plot(range(1, 17), block2_recurrences, label="Recurrent Block 2")
# plot.plot(range(1, 17), block3_recurrences, label="Recurrent Block 3")
# plot.plot(range(1, 17), block4_recurrences, label="Recurrent Block 4")
# plot.yscale("log")
# plot.legend()
# plot.xlabel("Recurrence")
# plot.ylabel("Rank")
# plot.title("Coda Lens Rank Trajectory of Final Predicted Token Over Recurrences")
# plot.savefig("graphs/arithmetic_rank_coda.png")