from datasets import load_dataset
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import random, numpy as np
from tqdm import tqdm

ds = load_dataset("EleutherAI/arithmetic", "arithmetic_1dc")

def operator(s, num1, num2):
    if s == "+":
        return num1 + num2
    elif s == "-":
        return num1 - num2
    elif s == "*":
        return num1 * num2
    else:
        raise ValueError("Invalid operator")
    
def single_digit_test(s):
    num1, operator1, num2, oprator2, num3 = int(s[0]), s[1], int(s[2]), s[3], int(s[4])

    result1 = operator(operator1, num1, num2)
    if len(str(result1)) > 1:
        return False
    result2 = operator(oprator2, result1, num3)
    if len(str(result2)) > 1:
        return False
    if num1 == result1 or num2 == result1 or result1 == result2:
        return False
    return True

def get_intermediate(s):
    num1, operator1, num2, oprator2, num3 = int(s[0]), s[1], int(s[2]), s[3], int(s[4])

    result1 = operator(operator1, num1, num2)
    return str(result1)

device = "cuda:0"
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def trim_output(out):
    return out.split("\n")[-1]

def get_answer_for_manual(model, tokenizer, messages, num_steps):
    set_seed()

    # Step 1: Use the chat template
    chat_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Step 2: Encode WITHOUT adding special tokens again (they're already in the template)
    input_ids = tokenizer.encode(chat_input, return_tensors="pt", add_special_tokens=False).to(device)

    # Step 3: Define a custom generation config (same as before)
    config = GenerationConfig(max_length=256, stop_strings=["<|end_text|>", "<|end_turn|>"], 
                            use_cache=True,
                            do_sample=False, temperature=None, top_k=None, top_p=None, min_p=None, 
                            return_dict_in_generate=True,
                            eos_token_id=65505,bos_token_id=65504,pad_token_id=65509, num_return_sequences=1)

    # Step 4: Generate
    with torch.no_grad():
        outputs = model.generate(input_ids, config, tokenizer=tokenizer, num_steps=num_steps)

    output = outputs.sequences[0]
    # print(outputs)
    decoded_output = tokenizer.decode(output, skip_special_tokens=True)

    return trim_output(decoded_output)

commit = "2a364bd96e3eaa831be324f7c1f9e74892e4e594"
model = AutoModelForCausalLM.from_pretrained("tomg-group-umd/huginn-0125", torch_dtype=torch.bfloat16, trust_remote_code=True, revision=commit)
tokenizer = AutoTokenizer.from_pretrained("tomg-group-umd/huginn-0125", revision=commit)
model.eval().to(device)
from datasets import load_dataset
from tqdm import tqdm
import copy
import re


single_digit_rows = []
for row in ds["validation"]:
    if len(row["completion"].strip()) == 1:
        if single_digit_test(re.findall(r'[\d]+|[+\-*]', row['context'])) and len(row['completion'].strip()) == 1:
            single_digit_rows.append(row)


print(len(single_digit_rows))

num_example_context = 4
messages = [
    {"role": "system", "content": "You are a concise and helpful assistant. Always return only the final answer straightway."},
]

for i in range(num_example_context):
    messages.append({"role": "user", "content": single_digit_rows[i]["context"]})
    messages.append({"role": "Huginn", "content": single_digit_rows[i]["completion"].strip()})

results = []
acc = 0
filtered_dataset = single_digit_rows[:4]
# load in dataset
for i in tqdm(range(num_example_context, len(single_digit_rows))): # , 100 + num_example_context)): #
    test_message = copy.deepcopy(messages)
    test_message.append({"role": "user", "content": single_digit_rows[i]["context"]})
    result = get_answer_for_manual(model, tokenizer, test_message, 16)
    print("correct", single_digit_rows[i]["completion"])
    print("predict", result)
    
    if result == single_digit_rows[i]["completion"].strip():
        single_digit_rows[i]["intermediate"] = get_intermediate(re.findall(r'[\d]+|[+\-*]', single_digit_rows[i]['context']))
        filtered_dataset.append(single_digit_rows[i])
        acc += 1

import json
with open("filtered_arithmetic_dataset.json", "w") as f:
    json.dump(filtered_dataset, f, indent=4) 
print("acc:", acc)