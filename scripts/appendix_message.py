
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch

import random, numpy as np, torch
import pickle

device = "cuda:0"

model = AutoModelForCausalLM.from_pretrained("./huginn-0125-local", torch_dtype=torch.bfloat16, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("./huginn-0125-local")
model.eval().to(device)
from datasets import load_dataset
from tqdm import tqdm
import copy
import pickle

ds = load_dataset("EleutherAI/arithmetic", "arithmetic_1dc")

num_example_context = 4
messages = [
    {"role": "system", "content": "You are a concise and helpful assistant. Always return only the final answer straightway."},
]

for i in range(num_example_context):
    messages.append({"role": "user", "content": ds["validation"][i]["context"]}) #[18:-9] + " = "})
    messages.append({"role": "Huginn", "content": ds["validation"][i]["completion"].strip()})

