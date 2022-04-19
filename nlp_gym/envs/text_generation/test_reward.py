from nlp_gym.envs.text_generation.observation import Observation
from nlp_gym.envs.text_generation.reward import RewardFunction


class RewardIncreasingNumbers(RewardFunction):
    def __init__(self, eos_token: str,
                 min_tokens: int,
                 include_prompt: bool = False) -> None:
        super().__init__()
        self.eos_token = eos_token
        self.min_tokens = min_tokens
        self.include_prompt = include_prompt

    def is_number(self, text):
        try:
            float(text)
            return True
        except ValueError:
            return False

    def __call__(self, prev_observation: Observation,
                 action: int,
                 current_observation: Observation,
                 done: bool) -> float:
        if done:
            gen_tokens = [
                current_observation.prompt_or_input_text] if self.include_prompt else []
            gen_tokens.extend(current_observation.action_history)
            if self.eos_token in gen_tokens:
                gen_tokens.remove(self.eos_token)
            number_tokens = [float(token)
                             for token in gen_tokens if self.is_number(token)]
            if len(number_tokens) > 0:
                # then we check how many numbers are in the sorted order
                sorted_count = 1
                previous_token = number_tokens[0]
                for token in number_tokens[1:]:
                    if token > previous_token:
                        sorted_count += 1
                        previous_token = token
                    else:
                        break
                return (sorted_count/max(len(gen_tokens), self.min_tokens))
        return 0.0
