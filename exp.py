
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

num_steps = 64

def find_sublist_index(lst, sub):
    for i in range(len(lst) - len(sub) + 1):
        if lst[i:i + len(sub)] == sub:
            return i
    return -1  # Not found

def trim_output(out):
    return out.split("\n")[-1]
device = "cuda:0"



def get_answer_for_manual(question):
    model = AutoModelForCausalLM.from_pretrained("tomg-group-umd/huginn-0125", torch_dtype=torch.bfloat16, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("tomg-group-umd/huginn-0125")
    model.eval().to(device)

    # Step 1: Use the chat template
    messages = [
        {"role": "system", "content": "You are a concise assistant. Always return only the final answer straightway."},
        #{"role": "user", "content": "Q: What is 2 + 2? Final answer: 4"},
        {"role": "user", "content": question}
    ]
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
def logit_lens(question, num_steps):
    def capture_last_block_output(module, inp, out):
    # SandwichBlock returns (hidden_state, attn_map); grab the hidden state tensor
        hidden_states_per_step.append(out[0].detach()) 


    hidden_states_per_step = []  # will collect the hidden state after each recurrence
    model = AutoModelForCausalLM.from_pretrained("tomg-group-umd/huginn-0125", torch_dtype=torch.bfloat16, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("tomg-group-umd/huginn-0125")

    last_core_layer = model.transformer.core_block[-1]
    hook_handle = last_core_layer.register_forward_hook(capture_last_block_output)


    model.eval().to(device)

    # Step 1: Use the chat template
    messages = [
        {"role": "system", "content": "You are a concise assistant. Always return only the final answer straightway."},
        {"role": "user", "content": "What is 2 + 2? Final answer: 4"},
        {"role": "user", "content": "What is 2 - 1 + 5? Final answer: 6"},
        {"role": "user", "content": question}
    ]
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
    selected_hidden = hidden_states_per_step[0: num_steps]

    for t, hidden in enumerate(selected_hidden, start=1):
        hidden_norm = model.transformer.ln_f(hidden)
        logits = model.lm_head(hidden_norm)         # shape: (batch, seq_len, vocab_size)
        print(logits.shape)
        top_tokens = tokenizer.batch_decode(logits.topk(k = 5, dim=-1)[1][0, -1].tolist())
        print(f"Step {t} top prediction(s): {top_tokens}")

def logit_coda_lens(question, num_steps, coda_layer = 1):
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
    messages = [
        {"role": "system", "content": "You are a concise assistant. Always return only the final answer straightway."},
        {"role": "user", "content": "What is 2 + 2? Final answer: 4"},
        {"role": "user", "content": "What is 2 - 1 + 5? Final answer: 6"},
        {"role": "user", "content": question}
    ]
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
    selected_hidden = hidden_states_per_step[0: num_steps]

    for t, hidden in enumerate(selected_hidden, start=1):
        hidden_norm = model.transformer.ln_f(hidden)
        logits = model.lm_head(hidden_norm)         # shape: (batch, seq_len, vocab_size)
        print(logits.shape)
        top_tokens = tokenizer.batch_decode(logits.topk(k = 5, dim=-1)[1][0, -1].tolist())
        print(f"Step {t} top prediction(s): {top_tokens}")
    
    pass
    output = outputs.sequences[0]
    # print(outputs)
    decoded_output = tokenizer.decode(output, skip_special_tokens=True)

    print(trim_output(decoded_output))

def coda_lens(question, num_steps):
    def capture_last_block_output(module, inp, out):
    # SandwichBlock returns (hidden_state, attn_map); grab the hidden state tensor
        hidden_states_per_step.append(out[0].detach()) 


    hidden_states_per_step = []  # will collect the hidden state after each recurrence
    model = AutoModelForCausalLM.from_pretrained("tomg-group-umd/huginn-0125", torch_dtype=torch.bfloat16, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("tomg-group-umd/huginn-0125")

    last_core_layer = model.transformer.core_block[-1]
    hook_handle = last_core_layer.register_forward_hook(capture_last_block_output)


    model.eval().to(device)

    # Step 1: Use the chat template
    messages = [
        {"role": "system", "content": "You are a concise assistant. Always return only the final answer straightway."},
        {"role": "user", "content": "What is 2 + 2? Final answer: 4"},
        {"role": "user", "content": "What is 2 - 1 + 5? Final answer: 6"},
        {"role": "user", "content": question}
    ]
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
        for i in range(1, num_steps + 1):
            outputs = model.generate(input_ids, config, tokenizer=tokenizer, num_steps=i)
            output = outputs.sequences[0]
            # print(outputs)
            decoded_output = tokenizer.decode(output, skip_special_tokens=True)

            print(f"step {i} prediction", trim_output(decoded_output))


question = "What is 2 + 4 + 2 + 1? Final answer: "
#get_answer_for_manual(question)
logit_lens(question, num_steps)
logit_coda_lens(question, num_steps)
#coda_lens(question, num_steps)