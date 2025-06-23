
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch

import random, numpy as np, torch
import pickle
from shared_store import current_pred_id, top_token_rank, intermediate_result_token, intermediate_token_rank, the_token, the_token_rank
results = []

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def trim_output(out):
    return out.split("\n")[-1]
device = "cuda:0"

import re

from shared_store import intermediate_coda_token

def get_final_predicted_logit(model, hidden):
    hidden_norm = model.transformer.ln_f(hidden)
    logits = model.lm_head(hidden_norm)         # shape: (batch, seq_len, vocab_size)
    top_token = logits.topk(k = 1, dim=-1)[1][0, -1].item()
    return top_token


def forward_generate(model, tokenizer, messages, final_token_ids):
    second_coda_states_per_step = []
    def capture_second_coda_layer_output(module, inp, out):
        second_coda_states_per_step.append(out[0].detach())
    
    set_seed()
    second_coda_layer = model.transformer.coda[1]
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
        outputs = model.generate(input_ids, config, tokenizer=tokenizer, num_steps=16)
    
    final_token_id = get_final_predicted_logit(model, second_coda_states_per_step[0])
    final_token_ids.append(final_token_id)




def get_final_token_ids():
    commit = "2a364bd96e3eaa831be324f7c1f9e74892e4e594"
    model = AutoModelForCausalLM.from_pretrained("tomg-group-umd/huginn-0125", torch_dtype=torch.bfloat16, trust_remote_code=True, revision=commit)
    tokenizer = AutoTokenizer.from_pretrained("tomg-group-umd/huginn-0125", revision=commit)
    model.eval().to(device)
    
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
    
    final_token_ids = []
    for i in tqdm(range(num_example_context, len(ds))):
        test_message = copy.deepcopy(messages)
        test_message.append({"role": "user", "content": ds[i]["context"]})
        forward_generate(model, tokenizer, test_message, final_token_ids)
    return final_token_ids

    pass
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
    arithmetic_token_ids = get_final_token_ids()

    model = AutoModelForCausalLM.from_pretrained("./huginn-trace", torch_dtype=torch.bfloat16, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("./huginn-trace")
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

    rank_results = []
    intermediate_rank_results = []
    random_the_results = []
    signed_numeric = [0 for i in range(68)]
    # load in dataset
    for i in tqdm(range(num_example_context, len(ds))):
        current_pred_id.clear()
        intermediate_result_token.clear()
        the_token.clear()
        
        current_pred_id.append(arithmetic_token_ids[i - num_example_context])
        test_message = copy.deepcopy(messages)
        test_message.append({"role": "user", "content": ds[i]["context"]}) #[18:-9] + " = "})
        intermediate_token = ds[i]["intermediate"].strip()
        intermediate_result_token.append(tokenizer.convert_tokens_to_ids(intermediate_token))
        intermediate_result_token.append(tokenizer.convert_tokens_to_ids(f"Ġ{intermediate_token}"))
        the_token.append(tokenizer.convert_tokens_to_ids(f"the"))

        get_answer_for_manual(model, tokenizer, test_message, 16)

        rank_results.append([top_token_rank[i] for i in range(16*4)])
        intermediate_rank_results.append([intermediate_token_rank[i] for i in range(16 * 4)])
        random_the_results.append([the_token_rank[i] for i in range(16 * 4)])
        #print("top rank", rank_results)
        #print("intermediate rank", intermediate_rank_results)
        intermediate_coda_token.clear()
        top_token_rank.clear()
        intermediate_token_rank.clear()
        the_token_rank.clear()

    # with open("cot_weights/coda_arithmetic_correct_rank_16.pkl", "wb") as f:
    #     pickle.dump(rank_results, f)

    # with open("cot_weights/coda_arithmetic_intern_rank_16.pkl", "wb") as f:
    #     pickle.dump(intermediate_rank_results, f)    
    
    with open("cot_weights/coda_arithmetic_the_rank_16_test.pkl", "wb") as f:
        pickle.dump(random_the_results, f)
