from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import obs_as_tensor
from copy import deepcopy
import torch


class KLRewardCallback(BaseCallback):
    """
    Callback to shape with rewards with KL divergence penalty

    This is a workaround because:
    - to compute KL, we need need current policy (and also reference LM) and we can't attach the 
      policy to the reward function due to the fact that instantiation of policy is done
      after the env. Since we also use VecEnv, it is tricky to communicate the policy to all envs
    """

    def __init__(self, kl_coeff: float,
                 batch_size: int,
                 verbose: int = 0):
        super().__init__(verbose)
        self._kl_coeff = kl_coeff
        self._batch_size = batch_size

    def _on_rollout_end(self) -> None:
        # get whole batch of rollout data
        input_ids = self.model.rollout_buffer.observations["input_encoded_pt"].copy(
        )
        attention_mask = self.model.rollout_buffer.observations["input_attention_mask_pt"].copy(
        )
        actions = self.model.rollout_buffer.actions.copy()
        log_probs = self.model.rollout_buffer.log_probs.copy()
        rewards = self.model.rollout_buffer.rewards.copy()

        # re-shape them to (n_steps * n_envs, dim)
        n_steps, n_envs, _ = input_ids.shape
        total_size = n_steps * n_envs
        input_ids = torch.from_numpy(input_ids.reshape(total_size, -1)).int()
        attention_mask = torch.from_numpy(
            attention_mask.reshape(total_size, -1))
        actions = torch.from_numpy(actions.reshape(total_size, -1)).long()
        log_probs = torch.from_numpy(log_probs.reshape(total_size, -1))
        rewards = torch.from_numpy(rewards.reshape(total_size))

        # here we compute KL rewards to the observations in the rollout buffer
        kl_div = torch.zeros_like(rewards)
        with torch.no_grad():
            current_ix = 0
            while current_ix < total_size:
                input_ids_batch = input_ids[current_ix:current_ix +
                                            self._batch_size, :].to(self.model.policy.device)
                attention_mask_batch = attention_mask[current_ix:current_ix +
                                                      self._batch_size, :].to(self.model.policy.device)
                actions_batch = actions[current_ix:current_ix +
                                        self._batch_size, :].to(self.model.policy.device).flatten()
                log_probs_batch = log_probs[current_ix:current_ix +
                                            self._batch_size, :].to(self.model.policy.device).flatten()

                ref_log_probs = self.model.policy.get_log_probs_ref_model(input_ids_batch,
                                                                          attention_mask_batch,
                                                                          actions_batch)

                log_ratio = log_probs_batch - ref_log_probs
                kl_div[current_ix:current_ix +
                       self._batch_size] = log_ratio
                current_ix += self._batch_size

        # compute KL reward from divergence (TBD: Verfiy this)
        kl_div = kl_div.reshape(n_steps, n_envs).numpy()
        kl_reward = -1 * self._kl_coeff * kl_div

        # add KL reward to task rewards
        task_rewards = rewards.reshape(n_steps, n_envs).numpy()
        total_rewards = task_rewards + kl_reward
        assert total_rewards.shape == self.model.rollout_buffer.rewards.shape
        self.model.rollout_buffer.rewards = total_rewards

        # Compute value for the last timestep
        with torch.no_grad():
            last_values = self.model.policy.predict_values(
                obs_as_tensor(self.locals["new_obs"], self.model.device))

        # and re-compute the advantages and return again
        self.model.rollout_buffer.compute_returns_and_advantage(
            last_values=last_values, dones=self.locals["dones"])

    def _on_step(self):
        super()._on_step()
