
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch

import random, numpy as np, torch

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

def is_boolean(s):
    bools = {"true", "false", "True", "False", "TRUE", "FALSE"}
    return s in bools



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


# answer: single token
def logit_lens(model, tokenizer, messages, num_steps, topk=1):
    set_seed()
    def capture_last_block_output(module, inp, out):
    # SandwichBlock returns (hidden_state, attn_map); grab the hidden state tensor
        hidden_states_per_step.append(out[0].detach()) 


    hidden_states_per_step = []  # will collect the hidden state after each recurrence
    #print(tokenizer.chat_template)

    last_core_layer = model.transformer.core_block[-1]
    hook_handle = last_core_layer.register_forward_hook(capture_last_block_output)

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

    # print(outputs)
    print(f"Captured {len(hidden_states_per_step)} intermediate states")

    # only select first output token
    selected_hidden = hidden_states_per_step[0: num_steps]

    for t, hidden in enumerate(selected_hidden, start=1):
        hidden_norm = model.transformer.ln_f(hidden)
        logits = model.lm_head(hidden_norm)         # shape: (batch, seq_len, vocab_size)
        print(logits.shape)
        top_tokens = tokenizer.batch_decode(logits.topk(k = topk, dim=-1)[1][0, -1].tolist())
        print(f"Step {t} top prediction(s): {top_tokens}")

def logit_coda_lens(model, tokenizer, messages, num_steps, select_recurr_steps = [32, 64], topk = 1):
    first_coda_states_per_step = []  # will collect the hidden state after each recurrence
    second_coda_states_per_step = []
    recurrent_states = []
    def capture_last_block_output(module, inp, out):
    # SandwichBlock returns (hidden_state, attn_map); grab the hidden state tensor
        recurrent_states.append(out[0].detach()) 

    def capture_first_coda_layer_output(module, inp, out):
    # SandwichBlock returns (hidden_state, attn_map); grab the hidden state tensor
        first_coda_states_per_step.append(out[0].detach()) 

    def capture_second_coda_layer_output(module, inp, out):
        second_coda_states_per_step.append(out[0].detach())
    set_seed()

    first_coda_layer = model.transformer.coda[0]
    second_coda_layer = model.transformer.coda[1]
    last_core_layer = model.transformer.core_block[-1]
    hook_handle = last_core_layer.register_forward_hook(capture_last_block_output)
    first_hook_handle = first_coda_layer.register_forward_hook(capture_first_coda_layer_output)
    second_hook_handle = second_coda_layer.register_forward_hook(capture_second_coda_layer_output)

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


    # only select first output token
    selected_hidden = [recurrent_states[i-1] for i in select_recurr_steps]
    # only select first output token
    selected_hidden += [first_coda_states_per_step[0], second_coda_states_per_step[0]]

    ret = []
    
    for t, hidden in enumerate(selected_hidden, start=1):
        hidden_norm = model.transformer.ln_f(hidden)
        logits = model.lm_head(hidden_norm)         # shape: (batch, seq_len, vocab_size)
        top_tokens = tokenizer.batch_decode(logits.topk(k = topk, dim=-1)[1][0, -1].tolist())
        ret += top_tokens
    hook_handle.remove()
    first_hook_handle.remove()
    second_hook_handle.remove()
    return ret
    
    # pass
    # output = outputs.sequences[0]
    # # print(outputs)
    # decoded_output = tokenizer.decode(output, skip_special_tokens=True)

    # print(trim_output(decoded_output))

def coda_lens(model, tokenizer, messages, num_steps, topk = 1):
    set_seed()

    chat_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # Step 2: Encode WITHOUT adding special tokens again (they're already in the template)
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
            outputs = model.generate(input_ids, config, tokenizer=tokenizer, num_steps=i)
            # output = outputs.sequences[0]
            # # print(outputs)
            # decoded_output = tokenizer.decode(output, skip_special_tokens=True)

            # print(f"step {i} prediction", trim_output(decoded_output))
            if len(outputs.scores) < 2:
                continue
            logits = outputs.scores[-2]  # this is shape (1, vocab_size) for the most recent token

            topk_ids = logits.topk(k=topk).indices[0]
            topk_tokens = tokenizer.batch_decode(topk_ids.tolist())
            print(f"Step {i} top-5: {topk_tokens}")
    


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
    model = AutoModelForCausalLM.from_pretrained("tomg-group-umd/huginn-0125", torch_dtype=torch.bfloat16, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("tomg-group-umd/huginn-0125")
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
    bools = [0, 0, 0, 0]
    # load in dataset
    for i in tqdm(range(num_example_context, len(ds['train']))):
        test_message = copy.deepcopy(messages)
        test_message.append({"role": "user", "content": ds["train"][i]["input"]})
        #get_answer_for_manual(model, tokenizer, test_message, 64)
        result = logit_coda_lens(model, tokenizer, test_message, 64)
        print(result)
        for i in range(4):
            if is_boolean(result[i].strip()):
                bools[i] += 1

        results.append(result)

    with open("boolean_results.pkl", "wb") as f:
        pickle.dump(results, f)
    print(bools)
    

    
        #get_answer_for_manual(model, tokenizer, test_message, 64)
    # A for loop

    # call logit_lens on last recurrent layer's output


    # call logit_coda_lens on first coda layer's output


    # call logit_coda_lens on second coda layer's output


    # add results to a list

    # aggregate data

    print(messages)
