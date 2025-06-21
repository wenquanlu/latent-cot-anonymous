import pickle

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

tokenizer = AutoTokenizer.from_pretrained("./huginn-0125-local")
print(tokenizer.convert_tokens_to_ids("the"))
#print(tokenizer.convert_tokens_to_ids("Ġthe"))