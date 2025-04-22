from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import random, numpy as np
from tqdm import tqdm
import pickle

tokenizer = AutoTokenizer.from_pretrained("tomg-group-umd/huginn-0125")
for i in range(10):
    print(tokenizer.convert_tokens_to_ids(f"Ġ{i}"))
    # print(tokenizer.convert_tokens_to_ids(f"Ċ{i}"))
    # print(tokenizer.convert_tokens_to_ids(f"{i}Ċ"))

# print(tokenizer.convert_tokens_to_ids('Ġ5'))
# print(tokenizer.convert_tokens_to_ids('1Ċ'))
#print(pickle.load(open("arithmetic_results_top5_16.pkl", "rb")))