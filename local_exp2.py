
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch

import random, numpy as np, torch
results = []

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def find_sublist_index(lst, sub):
    for i in range(len(lst) - len(sub) + 1):
        if lst[i:i + len(sub)] == sub:
            return i
    return -1  # Not found


def trim_output(out):
    return out.split("\n")[-1]
device = "cuda:0"

import re

def is_signed_numeric(s):
    return bool(re.fullmatch(r'(-\d+|\d+|-)', s))



from shared_store import intermediate_coda_token


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

    print(trim_output(decoded_output))



if __name__ == "__main__":
    num_steps = 64
    # messages = [
    #     {"role": "system", "content": "You are a concise and helpful assistant. Always return only the final answer straightway."},
    #     {"role": "user", "content": "3 + 2 - 1 = "},
    #     {"role": "Huginn", "content": "4"},
    #     {"role": "user", "content": "2 - 1 + 5 = "},
    #     {"role": "Huginn", "content": "6"}
    # ]
    # question = [{"role": "user", "content": "1 + 2 + 3 = "}]

    # messages = [
    #     {"role": "system", "content": "You are a concise and helpful assistant. Always return only the final answer straightway."},
    #     {"role": "user", "content": "not ( True ) and ( True ) is "},
    #     {"role": "Huginn", "content": "False"},
    #     {"role": "user", "content": "True and not not ( not False ) is "},
    #     {"role": "Huginn", "content": "True"},
    # ]
    # question = [{"role": "user", "content": "False or ( False ) and not False is "}]
    #get_answer_for_manual(question, num_steps)

    # messages = [
    #     {"role": "system", "content": "You are a concise and helpful assistant. Always return only the final answer straightway."},
    #     {"role": "user", "content": "Question: What is (7 + 5) - 6? Answer: "},
    #     {"role": "Huginn", "content": "6"},
    #     {"role": "user", "content": "Question: What is (4 + 8) - 9? Answer: "},
    #     {"role": "Huginn", "content": "3"},
    # ]
    # question = [{"role": "user", "content": "Question: What is (7 - 4) + 1? Answer: "}]
    model = AutoModelForCausalLM.from_pretrained("./huginn-0125-local", torch_dtype=torch.bfloat16, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("./huginn-0125-local")
    model.eval().to(device)

    # get_answer_for_manual(model, tokenizer, messages + question, num_steps=64)
    # logit_lens(model, tokenizer, messages + question, num_steps)
    # logit_coda_lens(model, tokenizer, messages + question, num_steps)
    # coda_lens(model, tokenizer, question, num_steps)

    from datasets import load_dataset
    from tqdm import tqdm
    import copy
    import pickle

    ds = load_dataset("maveriq/bigbenchhard", "boolean_expressions")

    num_example_context = 4
    messages = [
        {"role": "system", "content": "You are a concise and helpful assistant. Always return only the final answer straightway."},
    ]

    for i in range(num_example_context):
        messages.append({"role": "user", "content": ds["train"][i]["input"]})
        messages.append({"role": "Huginn", "content": ds["train"][i]["target"].strip()})
    
    results = []
    signed_numeric = [0 for i in range(68)]
    # load in dataset
    for i in tqdm(range(num_example_context, 100 + num_example_context)): #len(ds['validation']))):
        test_message = copy.deepcopy(messages)
        test_message.append({"role": "user", "content": ds["train"][i]["input"]})
        print(test_message)
        #print(test_message)
        get_answer_for_manual(model, tokenizer, test_message, 16)

        #coda_lens(model, tokenizer, test_message, 16)
        for i, row in enumerate(intermediate_coda_token):
            intermediate_coda_token[i] = tokenizer.decode(row)
        results.append([intermediate_coda_token[i] for i in range(16 * 4)])
        intermediate_coda_token.clear()

        
    
    #print(results)
    with open("coda_boolean_results_16.pkl", "wb") as f:
        pickle.dump(results, f)
    
        #get_answer_for_manual(model, tokenizer, test_message, 64)
    # A for loop

    # call logit_lens on last recurrent layer's output


    # call logit_coda_lens on first coda layer's output


    # call logit_coda_lens on second coda layer's output


    # add results to a list

    # aggregate data

    print(messages)
