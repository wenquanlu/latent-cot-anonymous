
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
import torch

import random, numpy as np, torch
import pickle

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = "cuda:0"


final_token_ids = []

    
def get_final_predicted_logit(model, hidden):
    hidden_norm = model.transformer.ln_f(hidden)
    logits = model.lm_head(hidden_norm)         # shape: (batch, seq_len, vocab_size)
    top_token = logits.topk(k = 1, dim=-1)[1][0, -1].item()
    return top_token

def logit_coda_rank_lens(model, tokenizer, messages, num_steps, select_recurr_steps = [32, 64]):
    first_coda_states_per_step = []  # will collect the hidden state after each recurrence
    second_coda_states_per_step = []
    recurrent_states = []
    first_prelude_states = []
    second_prelude_states = []

    #print(len(model.transformer.prelude))

    def capture_first_prelude(module, inp, out):
    # SandwichBlock returns (hidden_state, attn_map); grab the hidden state tensor
        first_prelude_states.append(out[0].detach()) 
    def capture_second_prelude(module, inp, out):
        second_prelude_states.append(out[0].detach()) 
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
    #print(len(model.transformer.core_block))
    first_prelude_handle = model.transformer.prelude[0].register_forward_hook(capture_first_prelude)
    second_prelude_handle = model.transformer.prelude[1].register_forward_hook(capture_second_prelude)
    core1_hook_handle = model.transformer.core_block[0].register_forward_hook(capture_last_block_output)
    core2_hook_handle = model.transformer.core_block[1].register_forward_hook(capture_last_block_output)
    core3_hook_handle = model.transformer.core_block[2].register_forward_hook(capture_last_block_output)
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
    
    final_token_id = get_final_predicted_logit(model, second_coda_states_per_step[0])
    final_token_ids.append(final_token_id)

    # only select first output token
    selected_hidden = [first_prelude_states[0], second_prelude_states[0]]
    selected_hidden += recurrent_states[:num_steps * 4]
    # only select first output token
    selected_hidden += [first_coda_states_per_step[0], second_coda_states_per_step[0]]
    # print("selected hidden length", len(selected_hidden))
    ranks = []
    top5 = []
    
    for t, hidden in enumerate(selected_hidden, start=1):
        hidden_norm = model.transformer.ln_f(hidden)
        logits = model.lm_head(hidden_norm)         # shape: (batch, seq_len, vocab_size)
        top5_tokens = tokenizer.batch_decode(logits.topk(k = 5, dim=-1)[1][0, -1].tolist())
        top5.append(top5_tokens)
        logits_at_last_token = logits[0, -1]
        sorted_logits, sorted_indices = torch.sort(logits_at_last_token, descending=True)
        rank = (sorted_indices == final_token_id).nonzero(as_tuple=False).item()
        ranks.append(rank)
    
    first_prelude_handle.remove()
    second_prelude_handle.remove()
    core1_hook_handle.remove()
    core2_hook_handle.remove()
    core3_hook_handle.remove()
    hook_handle.remove()
    first_hook_handle.remove()
    second_hook_handle.remove()
    return ranks, top5

    


if __name__ == "__main__":

    commit = "2a364bd96e3eaa831be324f7c1f9e74892e4e594"
    model = AutoModelForCausalLM.from_pretrained("tomg-group-umd/huginn-0125", torch_dtype=torch.bfloat16, trust_remote_code=True, revision=commit)
    tokenizer = AutoTokenizer.from_pretrained("tomg-group-umd/huginn-0125", revision=commit)
    model.eval().to(device)

    from datasets import load_dataset
    from tqdm import tqdm
    import copy

    ds = load_dataset("EleutherAI/arithmetic", "arithmetic_1dc")

    num_example_context = 4
    messages = [
        {"role": "system", "content": "You are a concise and helpful assistant. Always return only the final answer straightway."},
    ]

    for i in range(num_example_context):
        messages.append({"role": "user", "content": ds["validation"][i]["context"]})
        messages.append({"role": "Huginn", "content": ds["validation"][i]["completion"].strip()})
    
    ranks = []

    # load in dataset
    for i in tqdm(range(num_example_context, 100 + num_example_context)): #len(ds['validation']))):
        test_message = copy.deepcopy(messages)
        test_message.append({"role": "user", "content": ds["validation"][i]["context"]})
        rank, top5 = logit_coda_rank_lens(model, tokenizer, test_message, 16)
        ranks.append([i + 1 for i in rank])

    with open("arithmetic_rank_results_16.pkl", "wb") as f:
       pickle.dump(ranks, f)

    with open("arithmetic_results_top5_16.pkl", "wb") as f:
        pickle.dump(top5, f)

