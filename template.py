messages = []
messages.append({"role": "system", "content" : "You are a helpful assistant."})
messages.append({"role": "user", "content" : "What do you think of Goethe's Faust?"})
chat_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(chat_input)
input_ids = tokenizer.encode(chat_input, return_tensors="pt", add_special_tokens=False).to(device)

model.generate(input_ids, config, num_steps=64, tokenizer=tokenizer)
