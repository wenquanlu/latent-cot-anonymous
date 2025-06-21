
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch

import random, numpy as np
device = "cuda:0"
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



# def coda_lens(model, tokenizer, messages, num_steps, topk = 1):
#     ret = []
#     set_seed()
#     chat_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

#     # Step 2: Encode WITHOUT adding special tokens again (they're already in the template)
#     input_ids = tokenizer.encode(chat_input, return_tensors="pt", add_special_tokens=False).to(device)

#     # Step 3: Define a custom generation config (same as before)
#     config = GenerationConfig(max_length=256, stop_strings=["<|end_text|>", "<|end_turn|>"], 
#                             use_cache=True,
#                             do_sample=False, temperature=None, top_k=None, top_p=None, min_p=None, 
#                             return_dict_in_generate=True,
#                             output_scores=True,
#                             eos_token_id=65505,bos_token_id=65504,pad_token_id=65509, num_return_sequences=1)



#     outputs = model.generate(input_ids, config, tokenizer=tokenizer, num_steps=16)
#     # output = outputs.sequences[0]
#     # # print(outputs)
#     # decoded_output = tokenizer.decode(output, skip_special_tokens=True)

#     # print(f"step {i} prediction", trim_output(decoded_output))
#     # if len(outputs.scores) < 2:
#     #     continue
#     logits = outputs.scores[0]  # get the first answer token
#     assert len(outputs.scores[0]) == 1

#     topk_ids = logits.topk(k=1).indices[0].item()

#     return topk_ids

def coda_lens(model, tokenizer, messages, num_steps):
    set_seed()
    chat_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer.encode(chat_input, return_tensors="pt", add_special_tokens=False).to(device)


    # Step 3: Define a custom generation config (same as before)
    config = GenerationConfig(max_length=256, stop_strings=["<|end_text|>", "<|end_turn|>"], 
                            use_cache=True,
                            do_sample=False, temperature=None, top_k=None, top_p=None, min_p=None, 
                            return_dict_in_generate=True,
                            output_scores=True,
                            eos_token_id=65505,bos_token_id=65504,pad_token_id=65509, num_return_sequences=1)


    # Step 4: Generate
    with torch.no_grad():
        for i in range(1, num_steps + 1):
            set_seed()
            outputs = model.generate(input_ids, config, tokenizer=tokenizer, num_steps=i)
            # output = outputs.sequences[0]
            # # print(outputs)
            # decoded_output = tokenizer.decode(output, skip_special_tokens=True)

            # print(f"step {i} prediction", trim_output(decoded_output))
            if len(outputs.scores) < 2:
                continue
            logits = outputs.scores[-2]  # this is shape (1, vocab_size) for the most recent token

            topk_ids = logits.topk(k=5).indices[0]
            topk_tokens = tokenizer.batch_decode(topk_ids.tolist())
            print(f"Step {i} top-5: {topk_tokens}")

if __name__ == "__main__":
    num_steps = 64
    model = AutoModelForCausalLM.from_pretrained("tomg-group-umd/huginn-0125", torch_dtype=torch.bfloat16, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("tomg-group-umd/huginn-0125")
    model.eval().to(device)
    from datasets import load_dataset
    from tqdm import tqdm
    import copy
    import pickle
    import json
    ds = json.load(open("filtered_arithmetic_dataset.json", "r"))

    num_example_context = 4
    messages = [
        {"role": "system", "content": "You are a concise and helpful assistant. Always return only the final answer straightway."},
    ]

    for i in range(num_example_context):
        messages.append({"role": "user", "content": ds[i]["context"]})
        messages.append({"role": "Huginn", "content": ds[i]["completion"].strip()})
    
    results = []
    # signed_numeric = [0 for i in range(68)]
    # # load in dataset
    # for i in tqdm(range(num_example_context, 100 + num_example_context)): #len(ds['validation']))):
    #     test_message = copy.deepcopy(messages)
    #     test_message.append({"role": "user", "content": ds["validation"][i]["context"]})
    #     #print(test_message)
    #     #get_answer_for_manual(model, tokenizer, test_message, 64)
    #     #result = coda_lens(model, tokenizer, test_message, 16)
    #     result = logit_coda_rank_lens(model, tokenizer, test_message, 16)
    #     results.append([i + 1 for i in result])
    # print(final_token_ids)
    #print(results)
    #with open("arithmetic_rank_results_16.pkl", "wb") as f:
    #    pickle.dump(results, f)


    rank_inters = []
    correct_ids = []
    for i in tqdm(range(num_example_context, len(ds))):
        test_message = copy.deepcopy(messages)
        test_message.append({"role": "user", "content": ds[i]["context"]})
        print(ds[i]['context'])
        coda_lens(model, tokenizer, test_message, 16)
    #     topid = coda_lens(model, tokenizer, test_message, 16)
    #     correct_ids.append(topid)
    #     print("correct", ds[i]["completion"].strip())
    #     print("predict", tokenizer.decode(topid))
    #     print(get_answer_for_manual(model, tokenizer, test_message, 16))
    # print(correct_ids)
