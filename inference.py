import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
device = "cuda:0"

model = AutoModelForCausalLM.from_pretrained("tomg-group-umd/huginn-0125", torch_dtype=torch.bfloat16, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("tomg-group-umd/huginn-0125")

input_ids = tokenizer.encode("The capital of Westphalia is", return_tensors="pt", add_special_tokens=True).to(device)
model.eval()
model.to(device)

model(input_ids, num_steps=32)
model.eval()
config = GenerationConfig(max_length=256, stop_strings=["<|end_text|>", "<|end_turn|>"], 
                          use_cache=True,
                          do_sample=False, temperature=None, top_k=None, top_p=None, min_p=None, 
                          return_dict_in_generate=True,
                          eos_token_id=65505,bos_token_id=65504,pad_token_id=65509, num_return_sequences=1)


input_ids = tokenizer.encode("1 + 2 + 3 + 4 - 2 = ", return_tensors="pt", add_special_tokens=True).to(device)
outputs = model.generate(input_ids, config, tokenizer=tokenizer, num_steps=64)

generated_ids = outputs.sequences  # tensor of shape (batch_size, sequence_length)
print(generated_ids)
decoded_output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
print(decoded_output[0])