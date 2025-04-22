
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch
import json
import random, numpy as np, torch

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

num_steps = 64

def find_sublist_index(lst, sub):
    for i in range(len(lst) - len(sub) + 1):
        if lst[i:i + len(sub)] == sub:
            return i
    return -1  # Not found

def trim_output(out):
    return out.split("\n")[-1]
device = "cuda:0"



def get_answer_for_manual(messages, num_steps):
    model = AutoModelForCausalLM.from_pretrained("tomg-group-umd/huginn-0125", torch_dtype=torch.bfloat16, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("tomg-group-umd/huginn-0125")
    model.eval().to(device)

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

    print(decoded_output)


# answer: single token
def logit_lens(messages, num_steps):
    def capture_last_block_output(module, inp, out):
    # SandwichBlock returns (hidden_state, attn_map); grab the hidden state tensor
        hidden_states_per_step.append(out[0].detach()) 


    hidden_states_per_step = []  # will collect the hidden state after each recurrence
    model = AutoModelForCausalLM.from_pretrained("tomg-group-umd/huginn-0125", torch_dtype=torch.bfloat16, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("tomg-group-umd/huginn-0125")
    #print(tokenizer.chat_template)

    last_core_layer = model.transformer.core_block[-1]
    hook_handle = last_core_layer.register_forward_hook(capture_last_block_output)


    model.eval().to(device)

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
        top_tokens = tokenizer.batch_decode(logits.topk(k = 5, dim=-1)[1][0, -1].tolist())
        print(f"Step {t} top prediction(s): {top_tokens}")

def logit_coda_lens(messages, num_steps, coda_layer = 0):
    def capture_last_block_output(module, inp, out):
    # SandwichBlock returns (hidden_state, attn_map); grab the hidden state tensor
        print("hook called!")
        hidden_states_per_step.append(out[0].detach()) 

    set_seed()
    hidden_states_per_step = []  # will collect the hidden state after each recurrence
    model = AutoModelForCausalLM.from_pretrained("tomg-group-umd/huginn-0125", torch_dtype=torch.bfloat16, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("tomg-group-umd/huginn-0125")

    first_coda_layer = model.transformer.coda[coda_layer]
    hook_handle = first_coda_layer.register_forward_hook(capture_last_block_output)


    model.eval().to(device)

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
        top_tokens = tokenizer.batch_decode(logits.topk(k = 5, dim=-1)[1][0, -1].tolist())
        print(f"token {t} top prediction(s): {top_tokens}")
    
    pass
    output = outputs.sequences[0]
    # print(outputs)
    decoded_output = tokenizer.decode(output, skip_special_tokens=True)

    print(trim_output(decoded_output))

def coda_lens(messages, num_steps):
    set_seed()
    model = AutoModelForCausalLM.from_pretrained("tomg-group-umd/huginn-0125", torch_dtype=torch.bfloat16, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("tomg-group-umd/huginn-0125")

    model.eval().to(device)

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
            set_seed(42)
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
    with open('filtered_arithmetic_dataset.json', 'r') as f:
        data = json.load(f)
    ## 71 datasets in total
    msgs = data[:4]
    question = [data[6]]
    # print(msgs)
    # print(question)
    
    num_example_context = 4
    messages = [
        {"role": "system", "content": "You are a concise and helpful assistant. Always return only the final answer straightway."},
    ]

    for i in range(len(msgs)):
        messages.append({"role": "user", "content": data[i]["context"]})
        messages.append({"role": "Huginn", "content": data[i]["completion"].strip()})
    
    question[0]["role"] = "user"
    question[0]["content"] = question[0]["context"]
        
    # print(messages)
    print(question)
    get_answer_for_manual(messages + question, num_steps=16)
    coda_lens(question, num_steps=16)

    