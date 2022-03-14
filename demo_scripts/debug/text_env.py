from nlp_gym.data_pools.custom_text_generation_pools import CommonGen
from nlp_gym.envs.text_generation.env import TextGenEnv
from nlp_gym.envs.text_generation.reward import MeteorRewardFunction
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# data pool
data_pool = CommonGen.prepare(split="train")

# reward function
reward_fn = MeteorRewardFunction()

# tokenizer
model_name = "gpt2-xl"
tokenizer = AutoTokenizer.from_pretrained(model_name)
env = TextGenEnv(tokenizer, reward_function=reward_fn,
                 max_steps=10, max_text_length=24)
for sample, weight in data_pool:
    env.add_sample(sample, weight)

# model
device = "cuda:0" if torch.cuda.is_available() else "cpu"
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
model.config.pad_token_id = tokenizer.pad_token_id

# sample
sample = data_pool.sample()

print(sample)

# play an episode
obs = env.reset(sample)
done = False
generated = []
past = None

model_kwargs = {
    "attention_mask":  torch.from_numpy(
        obs["input_attention_mask_pt"]).reshape(1, -1).to(device)
}
while not done:
    input_ids = torch.from_numpy(
        obs["input_encoded_pt"]).reshape(1, -1).to(device)
    model_inputs = model.prepare_inputs_for_generation(
        input_ids, **model_kwargs)
    output = model(**model_inputs)

    next_token_logits = output.logits[0, -1, :]
    next_token_probs = torch.softmax(next_token_logits, dim=-1)
    sorted_ids = torch.argsort(next_token_probs, dim=-1, descending=True)
    action = sorted_ids[0]
    print(tokenizer.decode(action))

    # execute that action
    obs, reward, done, info = env.step(action)

    # update the past
    past = output.past_key_values

    # TBD - fix past bug - refer generate_using_gpt2
    model_kwargs = model._update_model_kwargs_for_generation(
        output, model_kwargs, is_encoder_decoder=model.config.is_encoder_decoder
    )

print(reward)
print(info)
