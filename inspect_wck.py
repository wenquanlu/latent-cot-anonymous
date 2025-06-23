import pickle

w = pickle.load(open("cot_weights/arithmetic_the_rank_results_16_test.pkl","rb"))

print(w[:3])