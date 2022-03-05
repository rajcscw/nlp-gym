from dataclasses import dataclass
from typing import Dict, List

import torch
from transformers import AutoTokenizer
from nlp_gym.data_pools.text_generation_pool import Sample
from copy import deepcopy


@dataclass
class Observation:
    # encoded input
    prompt_or_input_encoded_pt: torch.tensor
    # attention mask for the input
    prompt_or_input_attention_mask_pt: torch.tensor
    # input text
    prompt_or_input_text: str
    # encoded context
    context_encoded_pt: torch.tensor
    # attention mask for the context
    context_attention_mask_pt: torch.tensor
    # context text
    context_text: str
    # reference texts
    target_or_reference_texts: List[str]

    def to_dict(self) -> Dict[str, torch.tensor]:
        """
        For stable baselines (only return tensor items)
        """
        dict_obs = {
            "prompt_or_input_encoded_pt": self.prompt_or_input_encoded_pt,
            "prompt_or_input_attention_mask_pt": self.prompt_or_input_attention_mask_pt,
            "context_encoded_pt": self.context_encoded_pt,
            "context_attention_mask_pt": self.context_attention_mask_pt

        }
        return dict_obs

    def update(self, action: int, tokenizer: AutoTokenizer) -> "Observation":
        """
        Updates the observation using the given action
        """

        # get the current context
        current_context = deepcopy(self.context_encoded_pt)
        current_context_attention_mask = deepcopy(
            self.context_attention_mask_pt)

        if torch.sum(self.context_attention_mask_pt) > 0:
            context_length = self.context_attention_mask_pt.flatten().tolist().index(1)+1
        else:
            context_length = 0

        # update the context
        current_context[:, context_length] = action
        current_context_attention_mask[:, context_length] = 1

        # decode the context
        context_text = tokenizer.decode(
            current_context.flatten(), skip_special_tokens=True)

        # and create a new observation
        obs = Observation(self.prompt_or_input_encoded_pt,
                          self.prompt_or_input_attention_mask_pt,
                          self.prompt_or_input_text,
                          current_context,
                          current_context_attention_mask,
                          context_text,
                          self.target_or_reference_texts)

        return obs

    @ classmethod
    def init_from_sample(cls, sample: Sample,
                         tokenizer: AutoTokenizer,
                         max_length: int):
        # encode the prompt text
        prompt_outputs = tokenizer(sample.prompt_or_input_text,
                                   padding="max_length",
                                   max_length=max_length,
                                   return_tensors="pt",
                                   return_attention_mask=True)

        # encode the context text
        context_outputs = tokenizer("",
                                    padding="max_length",
                                    max_length=max_length,
                                    return_tensors="pt",
                                    return_attention_mask=True)

        obs = Observation(prompt_or_input_encoded_pt=prompt_outputs.input_ids,
                          prompt_or_input_attention_mask_pt=prompt_outputs.attention_mask,
                          prompt_or_input_text=sample.prompt_or_input_text,
                          context_encoded_pt=context_outputs.input_ids,
                          context_attention_mask_pt=context_outputs.attention_mask,
                          context_text="",
                          target_or_reference_texts=sample.references)

        return obs


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    sample = Sample("1", "Hello, this is cool", ["it is good", "going well"])

    obs = Observation.init_from_sample(
        sample=sample,
        tokenizer=tokenizer,
        max_length=256
    )
    updated_obs = obs.update(10, tokenizer)
    updated_obs = updated_obs.update(11, tokenizer)
