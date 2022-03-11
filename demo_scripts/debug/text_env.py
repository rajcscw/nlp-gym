from nlp_gym.data_pools.custom_text_generation_pools import CommonGen
from nlp_gym.data_pools.text_generation_pool import Sample
from nlp_gym.envs.text_generation.env import TextGenEnv
from nlp_gym.envs.text_generation.reward import MeteorRewardFunction
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


# reward function
reward_fn = MeteorRewardFunction()

# tokenizer
model_name = "gpt2-xl"
tokenizer = AutoTokenizer.from_pretrained(model_name)
env = TextGenEnv(tokenizer, reward_function=reward_fn,
                 max_steps=10, max_text_length=24)

# model
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model.config.pad_token_id = tokenizer.pad_token_id

# Sample
sample = Sample(1, prompt_or_input_text="Transformers are the",
                references=["best in the world"])

# play an episode
obs = env.reset(sample)
done = False
generated = []
while not done:

    input_ids = torch.from_numpy(
        obs["input_encoded_pt"]).reshape(1, -1).to(device)
    model_inputs = model.prepare_inputs_for_generation(
        input_ids, attention_mask=torch.from_numpy(obs["input_attention_mask_pt"]).reshape(1, -1).to(device))
    output = model(**model_inputs)

    next_token_logits = output.logits[0, -1, :]
    next_token_probs = torch.softmax(next_token_logits, dim=-1)
    sorted_ids = torch.argsort(next_token_probs, dim=-1, descending=True)
    action = sorted_ids[0]
    print(tokenizer.decode(action))

    # execute that action
    obs, reward, done, info = env.step(action)
print(reward)
print(info)
