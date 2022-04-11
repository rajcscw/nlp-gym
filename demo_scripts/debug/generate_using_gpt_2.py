from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')


# when generating, we will use the logits of right-most token to predict the next token
# so the padding should be on the left
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token  # to avoid an error

sentences = [
    # use different length sentences to test batching
    "I enjoy walking with my cute dog",
    "Transformers are"
]
inputs = tokenizer(sentences, return_tensors="pt",
                   padding="max_length", max_length=10)

model_kwargs = {
    "attention_mask": inputs["attention_mask"]
}

batch_size = len(sentences)
input_ids = inputs["input_ids"]
generated_actions = [[] for _ in sentences]
for i in range(20):
    model_inputs = model.prepare_inputs_for_generation(
        input_ids, **model_kwargs)
    outputs = model(**model_inputs)
    next_token_logits = outputs["logits"][:, -1, :].reshape(batch_size, -1)
    next_token_probs = torch.softmax(next_token_logits, dim=1)
    sorted_ids = torch.argsort(next_token_probs, dim=1, descending=True)

    for batch_ix in range(batch_size):
        gen_action = sorted_ids[batch_ix, 0]
        generated_actions[batch_ix].append(tokenizer.decode(gen_action))

    actions = sorted_ids[:, 0].reshape(batch_size, -1)
    input_ids = torch.cat(
        [input_ids, actions.clone()], dim=1)
    model_kwargs = model._update_model_kwargs_for_generation(
        outputs, model_kwargs, is_encoder_decoder=model.config.is_encoder_decoder
    )
print(generated_actions)
