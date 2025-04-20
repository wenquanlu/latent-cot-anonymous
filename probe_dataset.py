from datasets import load_dataset

ds = load_dataset("maveriq/bigbenchhard", "boolean_expressions")
print(ds['train'][0])