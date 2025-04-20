import pickle

arithmetic_data = pickle.load(open("arithmetic_results.pkl", "rb"))

boolean_data = pickle.load(open("boolean_results.pkl", "rb"))

arithmetic_agreement = [0, 0, 0, 0]
for row in arithmetic_data:
    for i in range(len(arithmetic_agreement)):
        if row[i] == row[-1]:
            arithmetic_agreement[i] += 1

print(arithmetic_agreement)

boolean_agreement = [0, 0, 0, 0]
for row in boolean_data:
    for i in range(len(boolean_agreement)):
        if row[i] == row[-1]:
            boolean_agreement[i] += 1
print(boolean_agreement)
