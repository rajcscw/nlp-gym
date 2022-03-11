from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')


# when generating, we will use the logits of right-most token to predict the next token
# so the padding should be on the left
tokenizer.padding_side = "left"
tokenizer.pad_token = tokenizer.eos_token  # to avoid an error

sentences = [
    "Hello, my dog is",  # use different length sentences to test batching
]
inputs = tokenizer(sentences, return_tensors="pt",
                   padding="max_length", max_length=10)

model_kwargs = {
    "attention_mask": inputs["attention_mask"]
}

input_ids = inputs["input_ids"]
for i in range(10):
    model_inputs = model.prepare_inputs_for_generation(
        input_ids, **model_kwargs)
    outputs = model(**model_inputs)
    next_token_logits = outputs["logits"][0, -1, :]
    next_token_probs = torch.softmax(next_token_logits, dim=-1)
    sorted_ids = torch.argsort(next_token_probs, dim=-1, descending=True)
    action = sorted_ids[0]
    print(tokenizer.decode(action))

    input_ids = torch.cat(
        [input_ids, torch.tensor(action.clone()).reshape(1, -1)], dim=1)
    model_kwargs = model._update_model_kwargs_for_generation(
        outputs, model_kwargs, is_encoder_decoder=model.config.is_encoder_decoder
    )
