import pickle
import matplotlib.pyplot as plot

inter_rank = pickle.load(open("cot_weights/arithmetic_inter_rank_results_16.pkl", "rb"))
correct_rank = pickle.load(open("cot_weights/arithmetic_correct_rank_results_16.pkl", "rb"))
print(correct_rank)
inter_results = [0 for i in range(68)]
correct_results = [0 for i in range(68)]

for row in inter_rank:
    for i in range(68):
        inter_results[i] += row[i]
    pass

for row in correct_rank:
    for i in range(68):
        correct_results[i] += row[i]

inter_average_ranks = [result/100 for result in inter_results]
correct_average_ranks = [result/100 for result in correct_results]

# block1_recurrences = []
# block2_recurrences = []
inter_block3_recurrences = []
correct_block3_recurrences = []
# block4_recurrences = []

for i in range(64):
    if i%4 == 2:
        inter_block3_recurrences.append(inter_average_ranks[i + 2])
for i in range(64):
    if i%4 == 2:
        correct_block3_recurrences.append(correct_average_ranks[i + 2])


# plot.plot(range(1, 17), block1_recurrences, label="Recurrent Block 1")
# plot.plot(range(1, 17), block2_recurrences, label="Recurrent Block 2")
plot.plot(range(1, 17), inter_block3_recurrences, label="Inter")
plot.plot(range(1, 17), correct_block3_recurrences, label="Correct")
plot.yscale("log")
plot.legend()
plot.savefig("graphs/arithmetic_inter.png")
# plot.plot(range(1, 17), block4_recurrences, label="Recurrent Block 4")

# plot.legend()
# plot.xlabel("Recurrence")
# plot.ylabel("Rank")
# plot.title("Logit Lens Rank Trajectory of Intermediate Token Over Recurrences")
# plot.savefig("graphs/arithmetic_intermediate_rank_logit.png")


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