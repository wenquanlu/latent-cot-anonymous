import pickle

#arithmetic_data = pickle.load(open("arithmetic_results_16.pkl", "rb"))

#boolean_data = pickle.load(open("boolean_results_16.pkl", "rb"))

# arithmetic_data = pickle.load(open("coda_last_arithmetic_results_16.pkl", "rb"))

# arithmetic_agreement = [0 for i in range(16)]
# for row in arithmetic_data:
#     for i in range(len(arithmetic_agreement)):
#         if row[i] == row[-1]:
#             arithmetic_agreement[i] += 1

# print(arithmetic_agreement)

# import matplotlib.pyplot as plot

# plot.plot(range(1, 17), arithmetic_agreement)

# plot.savefig("arithmetic_coda_last_agreement.png")

# boolean_data = pickle.load(open("coda_last_boolean_results_16.pkl", "rb"))

# arithmetic_agreement = [0 for i in range(16)]
# for row in boolean_data:
#     for i in range(len(arithmetic_agreement)):
#         if row[i] == row[-1]:
#             arithmetic_agreement[i] += 1

# print(arithmetic_agreement)

# import matplotlib.pyplot as plot

# plot.plot(range(1, 17), arithmetic_agreement)

# plot.savefig("boolean_coda_last_agreement.png")


# boolean_data = pickle.load(open("coda_boolean_results_16.pkl", "rb"))

# arithmetic_agreement = [0 for i in range(64)]
# for row in boolean_data:
#     for i in range(len(arithmetic_agreement)):
#         if row[i] == row[-1]:
#             arithmetic_agreement[i] += 1

# print(arithmetic_agreement)

# import matplotlib.pyplot as plot

# plot.plot(range(2, 2 + 64), arithmetic_agreement)

# plot.savefig("boolean_coda_agreement.png")























# arithmetic_data = pickle.load(open("coda_arithmetic_results_16.pkl", "rb"))

# arithmetic_agreement = [0 for i in range(64)]
# for row in arithmetic_data:
#     for i in range(len(arithmetic_agreement)):
#         if row[i] == row[-1]:
#             arithmetic_agreement[i] += 1

# print(arithmetic_agreement)

# import matplotlib.pyplot as plot

# # plot.plot(range(2, 2 + 64), arithmetic_agreement)

# # plot.savefig("aritmetic_coda_agreement.png")

# block1_recurrences = []
# block2_recurrences = []
# block3_recurrences = []
# block4_recurrences = []

# for i in range(64):
#     if i % 4 == 0:
#         block1_recurrences.append(arithmetic_agreement[i])
#     elif i%4 == 1:
#         block2_recurrences.append(arithmetic_agreement[i])
#     elif i%4 == 2:
#         block3_recurrences.append(arithmetic_agreement[i])
#     elif i%4 == 3:
#         block4_recurrences.append(arithmetic_agreement[i])

# plot.plot(range(1, 17), block1_recurrences, label="Recurrent Block 1")
# plot.plot(range(1, 17), block2_recurrences, label="Recurrent Block 2")
# plot.plot(range(1, 17), block3_recurrences, label="Recurrent Block 3")
# plot.plot(range(1, 17), block4_recurrences, label="Recurrent Block 4")

# plot.legend()
# plot.xlabel("Recurrence")
# plot.ylabel("Percent")
# plot.title("Match Rate of Coda Lens Top Token with Final Prediction vs Recurrences")
# plot.savefig("graphs/arithmetic_agreement_coda.png")





arithmetic_data = pickle.load(open("arithmetic_results_16.pkl", "rb"))

print(arithmetic_data)
arithmetic_agreement = [0 for i in range(68)]
for row in arithmetic_data:
    for i in range(len(arithmetic_agreement)):
        if row[i] == row[-1]:
            arithmetic_agreement[i] += 1

print(arithmetic_agreement)

import matplotlib.pyplot as plot

# plot.plot(range(2, 2 + 64), arithmetic_agreement)

# plot.savefig("aritmetic_coda_agreement.png")

block1_recurrences = []
block2_recurrences = []
block3_recurrences = []
block4_recurrences = []

for i in range(64):
    if i % 4 == 0:
        block1_recurrences.append(arithmetic_agreement[i + 2])
    elif i%4 == 1:
        block2_recurrences.append(arithmetic_agreement[i + 2])
    elif i%4 == 2:
        block3_recurrences.append(arithmetic_agreement[i + 2])
    elif i%4 == 3:
        block4_recurrences.append(arithmetic_agreement[i + 2])

plot.plot(range(1, 17), block1_recurrences, label="Recurrent Block 1")
plot.plot(range(1, 17), block2_recurrences, label="Recurrent Block 2")
plot.plot(range(1, 17), block3_recurrences, label="Recurrent Block 3")
plot.plot(range(1, 17), block4_recurrences, label="Recurrent Block 4")

plot.legend()
plot.xlabel("Recurrence")
plot.ylabel("Percent")
plot.title("Match Rate of Logit Lens Top Token with Final Prediction vs Recurrences")
plot.savefig("graphs/arithmetic_agreement_logit.png")





















# arithmetic_agreement = [0 for i in range(68)]
# for row in arithmetic_data:
#     for i in range(len(arithmetic_agreement)):
#         if row[i] == row[-1]:
#             arithmetic_agreement[i] += 1

# print(arithmetic_agreement)

# import matplotlib.pyplot as plot

# plot.plot(arithmetic_agreement)

# plot.savefig("arithmetic_agreement.png")

# modulo_agreement = [0, 0, 0, 0]
# for row in arithmetic_data:
#     for i in range(2, 2 + 64):
#         j = (i - 2)%4
#         if row[i] == row[-1]:
#             modulo_agreement[j] += 1

# print(modulo_agreement)

# boolean_agreement = [0 for i in range(68)]
# for row in boolean_data:
#     for i in range(len(boolean_agreement)):
#         if row[i] == row[-1]:
#             boolean_agreement[i] += 1

# print(boolean_agreement)

# import matplotlib.pyplot as plot

# plot.plot(boolean_agreement)

# plot.savefig("boolean_agreement.png")

# modulo_agreement = [0, 0, 0, 0]
# for row in arithmetic_data:
#     for i in range(2, 2 + 64):
#         j = (i - 2)%4
#         if row[i] == row[-1]:
#             modulo_agreement[j] += 1

# print(modulo_agreement)


# boolean_agreement = [0, 0, 0, 0]
# for row in boolean_data:
#     for i in range(len(boolean_agreement)):
#         if row[i] == row[-1]:
#             boolean_agreement[i] += 1
# print(boolean_agreement)
