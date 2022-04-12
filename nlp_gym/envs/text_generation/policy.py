from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import torch
from gym.spaces import Discrete
from gym.spaces.dict import Dict as DictSpace
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import Schedule
from torch import nn
from transformers import AdamW, AutoModelForCausalLM
from stable_baselines3.common.distributions import CategoricalDistribution
from copy import deepcopy
from transformers.generation_utils import top_k_top_p_filtering


class LMActorCriticPolicy(BasePolicy):
    def __init__(self, observation_space: DictSpace,
                 action_space: Discrete,
                 lr_schedule: Schedule,
                 model_name: str,
                 optimizer_kwargs: Dict[str, Any] = {},
                 weight_decay: float = 1e-6,
                 use_sde: bool = None,
                 apply_model_parallel: bool = True,
                 sample_during_rollout: bool = True,
                 logits_filtering_args: dict = {}):
        super().__init__(observation_space, action_space)
        self._action_space = action_space
        self._apply_model_parallel = apply_model_parallel
        self._build_model_heads(model_name)
        self._setup_optimizer(optimizer_kwargs, weight_decay)
        self._action_dist = CategoricalDistribution(
            self._action_space.n)
        self._sample_during_rollout = sample_during_rollout
        self._logits_filtering_args = logits_filtering_args

    def _build_model_heads(self,
                           model_name: str):
        self._policy_model = AutoModelForCausalLM.from_pretrained(
            model_name)
        self._value_model = AutoModelForCausalLM.from_pretrained(
            model_name)
        self._ref_model = deepcopy(self._policy_model)

        # apply model parallel
        if torch.cuda.is_available() and self._apply_model_parallel:
            if self._policy_model.is_parallelizable:
                self._policy_model.parallelize()
                self._ref_model.parallelize()
            if self._value_model.is_parallelizable:
                self._value_model.parallelize()

        self._value_head = nn.Linear(
            self._value_model.config.hidden_size, 1)

    def _setup_optimizer(self, optimizer_kwargs: Dict[str, Any],
                         weight_decay: float):
        params = list(self.named_parameters())

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in params if not any(
                nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in params if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        self.optimizer = AdamW(
            optimizer_grouped_parameters, **optimizer_kwargs)

    def _prepare_inputs_for_model(self, model: AutoModelForCausalLM,
                                  input_ids: torch.tensor,
                                  model_kwargs: Optional[Dict[str, torch.tensor]] = None):
        model_inputs = model.prepare_inputs_for_generation(
            input_ids, **model_kwargs)

        if self._apply_model_parallel:
            # if model is in parallel mode, move the tensors to the first device
            model_inputs = {key: value.to(model.transformer.first_device) if isinstance(
                value, torch.Tensor) else value for key, value in model_inputs.items()}
        return model_inputs

    def _sample_actions(self, next_token_logits: torch.tensor):
        if self._logits_filtering_args:
            next_token_logits = top_k_top_p_filtering(
                next_token_logits, **self._logits_filtering_args)

        dist = self._action_dist.proba_distribution(
            action_logits=next_token_logits)
        actions = dist.get_actions(not self._sample_during_rollout)
        return actions

    def _forward_policy(self, input_ids: torch.tensor,
                        attention_mask: torch.tensor,
                        model_kwargs: Optional[Dict[str, torch.tensor]] = None,
                        greedy: bool = False,
                        actions: torch.tensor = None):
        # prepare inputs
        if not model_kwargs:
            model_kwargs = {
                "attention_mask": attention_mask,
            }
        model_inputs = self._prepare_inputs_for_model(self._policy_model,
                                                      input_ids,
                                                      model_kwargs)

        # forward pass to transformers
        output = self._policy_model(
            output_hidden_states=True, **model_inputs)

        # compute action probs - policy head
        next_token_logits = output.logits[:, -1, :]
        dist = self._action_dist.proba_distribution(
            action_logits=next_token_logits)
        entropy = dist.entropy()

        # sample actions or evaluate if provided already
        if actions is None:
            actions = self._sample_actions(next_token_logits)
        log_prob = dist.log_prob(actions)

        return actions, log_prob, entropy, output

    def _forward_value(self, input_ids: torch.tensor,
                       attention_mask: torch.tensor,
                       model_kwargs: Optional[Dict[str, torch.tensor]] = None):
        # prepare inputs
        if not model_kwargs:
            model_kwargs = {
                "attention_mask": attention_mask,
            }
        model_inputs = self._prepare_inputs_for_model(self._value_model,
                                                      input_ids,
                                                      model_kwargs)

        # forward pass to transformers
        output = self._value_model(
            output_hidden_states=True, **model_inputs)

        last_tokens_hidden = output.hidden_states[-1][:, -1, :].to(self.device)
        values = self._value_head.forward(last_tokens_hidden)
        return values

    def forward(self, obs: Dict[str, torch.tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass to both policy and critic networks

        Args:
            obs (Dict[str, torch.tensor]): observation

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: action, value and log probability of the action
        """
        input_ids = obs["input_encoded_pt"].int()
        attention_mask = obs["input_attention_mask_pt"]
        actions, log_prob, _, _ = self._forward_policy(
            input_ids, attention_mask)
        values = self._forward_value(input_ids, attention_mask)
        return actions, values, log_prob

    @staticmethod
    def _predict(self, observation: Dict[str, torch.tensor],
                 deterministic: bool = False) -> torch.Tensor:
        # dummy just to comply with base policy
        pass

    def predict(
            self,
            observation: Union[np.ndarray, Dict[str, np.ndarray]],
            state: Optional[Tuple[np.ndarray, ...]] = None,
            episode_start: Optional[np.ndarray] = None,
            deterministic: bool = False) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:

        with torch.no_grad():

            # input ids and mask
            input_ids = torch.from_numpy(
                observation["input_encoded_pt"]).reshape(1, -1).to(self.device).int()
            attention_mask = torch.from_numpy(
                observation["input_attention_mask_pt"]).reshape(1, -1).to(self.device)

            # use the past
            if state is None:
                model_kwargs = {
                    "attention_mask":  torch.from_numpy(
                        observation["input_attention_mask_pt"]).reshape(1, -1).to(self.device)
                }
            else:
                model_kwargs = state

            # forward heads
            action, _, _, output = self._forward_policy(
                input_ids, attention_mask, model_kwargs, True)

            # update the model kwargs for further generation
            model_kwargs = self._policy_model._update_model_kwargs_for_generation(
                output, model_kwargs, is_encoder_decoder=self._policy_model.config.is_encoder_decoder
            )
            return action, model_kwargs

    def predict_values(self, obs: Dict[str, torch.tensor]):
        input_ids = obs["input_encoded_pt"].int()
        attention_mask = obs["input_attention_mask_pt"]
        values = self._forward_value(input_ids, attention_mask)
        return values

    def evaluate_actions(self, obs: torch.Tensor,
                         actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        input_ids = obs["input_encoded_pt"].int()
        attention_mask = obs["input_attention_mask_pt"]

        _, log_prob, entropy, _ = self._forward_policy(input_ids=input_ids,
                                                       attention_mask=attention_mask,
                                                       actions=actions)
        values = self._forward_value(input_ids, attention_mask)

        return values, log_prob, entropy

    def _forward_ref_model(self, input_ids: torch.tensor,
                           attention_mask: torch.tensor,
                           action: int):
        model_kwargs = {
            "attention_mask": attention_mask,
        }
        model_inputs = self._prepare_inputs_for_model(self._ref_model,
                                                      input_ids,
                                                      model_kwargs)
        output = self._ref_model(
            output_hidden_states=True, **model_inputs)
        next_token_logits = output.logits[:, -1, :]
        dist = self._action_dist.proba_distribution(
            action_logits=next_token_logits)
        log_prob = dist.log_prob(action)
        return log_prob

    def compute_ref_divergence(self, observation: Dict[str, torch.tensor], action: int):
        with torch.no_grad():
            action = torch.tensor(action).to(self.device)
            input_ids = torch.from_numpy(
                observation["input_encoded_pt"]).reshape(1, -1).to(self.device).int()
            attention_mask = torch.from_numpy(
                observation["input_attention_mask_pt"]).reshape(1, -1).to(self.device)

            # target
            target_probs = self._forward_ref_model(
                input_ids, attention_mask, action=action)

            # current policy
            _, log_prob, _, output = self._forward_policy(
                input_ids, attention_mask, actions=action)

            # KL divergence
            log_ratio = log_prob - target_probs
            approx_kl_div = ((torch.exp(log_ratio) - 1) -
                             log_ratio).cpu().item()
            return approx_kl_div

    def to(self, device):
        if self._apply_model_parallel:
            self._value_head = self._value_head.to(device)
            return self
        else:
            return super().to(device)
