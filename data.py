from datasets import load_dataset
import json

# Load the dataset
dataset = load_dataset("maveriq/bigbenchhard", split="train")

# Take first 50 examples
subset = dataset.select(range(50))

# Save to JSON file
output_file = "bigbenchhard_50_examples.json"
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(subset.to_list(), f, indent=2, ensure_ascii=False)

print(f"Saved 50 examples to {output_file}")
