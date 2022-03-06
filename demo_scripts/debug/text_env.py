from nlp_gym.envs.text_generation.env import TextGenEnv
from transformers import AutoTokenizer, AutoModelForCausalLM
from nlp_gym.data_pools.text_generation_pool import Sample
import torch


def get_last(obs):
    # get context length/index
    # find the first entry where the mask is 0
    index = obs["context_attention_mask_pt"].flatten().tolist().index(0)-1

    if index == -1:
        index = 0

    return obs["context_encoded_pt"][:, index]


# tokenizer
model_name = "gpt2-xl"
tokenizer = AutoTokenizer.from_pretrained(model_name)
env = TextGenEnv(tokenizer, reward_function=None,
                 max_steps=10, max_text_length=24)

# model
device = "cuda:0" if torch.cuda.is_available() else "cpu"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

# sample
sample = Sample(1, "Transformers are the", ["Hello, who is this"])

input_ids = tokenizer("Transformers are the", return_tensors="pt")[
    "input_ids"].to(device)

# play an episode
obs = env.reset(sample)
done = False
generated = []
while not done:

    # given the obs, get the next action using Gpt-2
    output = model(input_ids=obs["input_encoded_pt"].to(device),
                   attention_mask=obs["input_attention_mask_pt"].to(device))

    # pick the logits corresponding to the right most token (AR)
    # alternativelty one can just concanate the prompt and context tokens
    masks = obs["input_attention_mask_pt"].flatten().tolist()
    masks.reverse()
    index = masks.index(1)
    index = 34 - index - 1

    # choose the action
    next_token_logits = output.logits[0, index-1, :]
    next_token_probs = torch.softmax(next_token_logits, dim=-1)
    sorted_ids = torch.argsort(next_token_probs, dim=-1, descending=True)
    action = sorted_ids[0]
    print(index, tokenizer.decode(action))

    # execute that action
    obs, _, done, info = env.step(action)
print(info)
