import re
import pickle
import matplotlib.pyplot as plot

def is_signed_numeric(s):
    return bool(re.fullmatch(r'(-\d+|\d+|-)', s))


arithmetic_data = pickle.load(open("coda_arithmetic_results_top5_16.pkl", "rb"))


freq = [0 for i in range(64)]

for row in arithmetic_data:
    for i in range(64):
        tokens = row[i]
        count = 0
        for token in tokens:
            if is_signed_numeric(token.strip()):
                count += 1
        freq[i] += count

average_freq = [i/500 for i in freq]


block1_recurrences = []
block2_recurrences = []
block3_recurrences = []
block4_recurrences = []

for i in range(64):
    if i % 4 == 0:
        block1_recurrences.append(average_freq[i])
    elif i%4 == 1:
        block2_recurrences.append(average_freq[i])
    elif i%4 == 2:
        block3_recurrences.append(average_freq[i])
    elif i%4 == 3:
        block4_recurrences.append(average_freq[i])

plot.plot(range(1, 17), block1_recurrences, label="Recurrent Block 1")
plot.plot(range(1, 17), block2_recurrences, label="Recurrent Block 2")
plot.plot(range(1, 17), block3_recurrences, label="Recurrent Block 3")
plot.plot(range(1, 17), block4_recurrences, label="Recurrent Block 4")
plot.legend()
plot.xlabel("Recurrence")
plot.ylabel("Percent")
plot.title("Proportion of Signed Integers in Coda Lens Top-5 Predictions vs. Recurrence \nStep")
plot.savefig("graphs/arithmetic_numeric_coda.png")


# arithmetic_data = pickle.load(open("arithmetic_results_top5_16.pkl", "rb"))


# freq = [0 for i in range(68)]

# for row in arithmetic_data:
#     for i in range(68):
#         tokens = row[i]
#         count = 0
#         for token in tokens:
#             if is_signed_numeric(token.strip()):
#                 count += 1
#         freq[i] += count

# average_freq = [i/500 for i in freq]


# block1_recurrences = []
# block2_recurrences = []
# block3_recurrences = []
# block4_recurrences = []

# for i in range(64):
#     if i % 4 == 0:
#         block1_recurrences.append(average_freq[i + 2])
#     elif i%4 == 1:
#         block2_recurrences.append(average_freq[i + 2])
#     elif i%4 == 2:
#         block3_recurrences.append(average_freq[i + 2])
#     elif i%4 == 3:
#         block4_recurrences.append(average_freq[i + 2])

# plot.plot(range(1, 17), block1_recurrences, label="Recurrent Block 1")
# plot.plot(range(1, 17), block2_recurrences, label="Recurrent Block 2")
# plot.plot(range(1, 17), block3_recurrences, label="Recurrent Block 3")
# plot.plot(range(1, 17), block4_recurrences, label="Recurrent Block 4")
# plot.legend()
# plot.xlabel("Recurrence")
# plot.ylabel("Percent")
# plot.title("Proportion of Signed Integers in Logit Lens Top-5 Predictions vs. Recurrence \nStep")
# plot.savefig("graphs/arithmetic_numeric_logit.png")
