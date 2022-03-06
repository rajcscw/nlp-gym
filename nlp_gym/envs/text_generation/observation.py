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

    # concatenated input
    input_encoded_pt: torch.tensor
    input_attention_mask_pt: torch.tensor

    def to_dict(self) -> Dict[str, torch.tensor]:
        """
        For stable baselines (only return tensor items)
        """
        dict_obs = {
            "prompt_or_input_encoded_pt": self.prompt_or_input_encoded_pt,
            "prompt_or_input_attention_mask_pt": self.prompt_or_input_attention_mask_pt,
            "context_encoded_pt": self.context_encoded_pt,
            "context_attention_mask_pt": self.context_attention_mask_pt,
            "input_encoded_pt": self.input_encoded_pt,
            "input_attention_mask_pt": self.input_attention_mask_pt

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

        # get context length/index
        # find the first entry where the mask is 0
        next_index = self.context_attention_mask_pt.flatten().tolist().index(0)

        # update the context
        current_context[:, next_index] = action
        current_context_attention_mask[:, next_index] = 1

        # decode the context
        context_text = tokenizer.decode(
            current_context.flatten(), skip_special_tokens=True)

        # concatenate
        input_encoded_pt = torch.cat(
            (self.prompt_or_input_encoded_pt, current_context), dim=1)
        input_attention_mask_pt = torch.cat(
            (self.prompt_or_input_attention_mask_pt, current_context_attention_mask), dim=1)

        # and create a new observation
        obs = Observation(self.prompt_or_input_encoded_pt,
                          self.prompt_or_input_attention_mask_pt,
                          self.prompt_or_input_text,
                          current_context,
                          current_context_attention_mask,
                          context_text,
                          self.target_or_reference_texts,
                          input_encoded_pt,
                          input_attention_mask_pt)

        return obs

    @ classmethod
    def init_from_sample(cls, sample: Sample,
                         tokenizer: AutoTokenizer,
                         max_input_length: int,
                         max_context_length: int):
        # encode the prompt text
        prompt_outputs = tokenizer(sample.prompt_or_input_text,
                                   padding="max_length",
                                   max_length=max_input_length,
                                   return_tensors="pt",
                                   return_attention_mask=True,
                                   truncation=True)

        # encode the context text
        context_outputs = tokenizer("",
                                    padding="max_length",
                                    max_length=max_context_length,
                                    return_tensors="pt",
                                    return_attention_mask=True)

        # concatenate
        input_encoded_pt = torch.cat(
            (prompt_outputs.input_ids, context_outputs.input_ids), dim=1)
        input_attention_mask_pt = torch.cat(
            (prompt_outputs.attention_mask, context_outputs.attention_mask), dim=1)

        obs = Observation(prompt_or_input_encoded_pt=prompt_outputs.input_ids,
                          prompt_or_input_attention_mask_pt=prompt_outputs.attention_mask,
                          prompt_or_input_text=sample.prompt_or_input_text,
                          context_encoded_pt=context_outputs.input_ids,
                          context_attention_mask_pt=context_outputs.attention_mask,
                          input_encoded_pt=input_encoded_pt,
                          input_attention_mask_pt=input_attention_mask_pt,
                          context_text="",
                          target_or_reference_texts=sample.references)

        return obs


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    sample = Sample("1", "Hello, this is cool", ["it is good", "going well"])

    obs = Observation.init_from_sample(
        sample=sample,
        tokenizer=tokenizer,
        max_input_length=256,
        max_context_length=256
    )
    updated_obs = obs.update(10, tokenizer)
    updated_obs = updated_obs.update(11, tokenizer)
