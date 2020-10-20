
from datasets import load_dataset

dataset = load_dataset("conll2003")

for p in dataset["train"]:
    print(p)
