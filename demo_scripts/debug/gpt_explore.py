# hide_output
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

device = "cuda:1" if torch.cuda.is_available() else "cpu"
model_name = "gpt2-xl"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)


input_txt = "Transformers are the"
input_ids = tokenizer(input_txt, return_tensors="pt")["input_ids"].to(device)
iterations = []
n_steps = 8

generated = []
with torch.no_grad():
    past_key_values = None
    context = input_ids
    for _ in range(n_steps):
        output = model(input_ids=context, past_key_values=past_key_values)
        # Select logits of the first batch and the last token and apply softmax
        next_token_logits = output.logits[0, -1, :]
        next_token_probs = torch.softmax(next_token_logits, dim=-1)
        sorted_ids = torch.argsort(next_token_probs, dim=-1, descending=True)
        token_id = sorted_ids[0]

        # update context and past key values
        context = torch.tensor(token_id).reshape(1, -1)
        past_key_values = output.past_key_values

        # generated
        generated.append(token_id)

print(tokenizer.decode(generated))
