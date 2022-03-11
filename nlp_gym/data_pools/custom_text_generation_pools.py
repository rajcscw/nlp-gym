from nlp_gym.data_pools.text_generation_pool import TextGenPool, Sample
from abc import abstractclassmethod
from typing import List
from datasets import load_dataset


class CommonGen(TextGenPool):
    @classmethod
    def prepare(cls, split: str) -> 'TextGenPool':
        ds = load_dataset("gem", "common_gen")

        samples = []
        for ix, item in enumerate(ds[split]):
            concepts = " ".join(item["concepts"])
            targets = [item["target"]]
            sample = Sample(id=f"{split}_{ix}",
                            prompt_or_input_text=concepts,
                            references=targets
                            )
            samples.append(sample)
        pool_instance = cls(samples)
        return pool_instance


if __name__ == "__main__":
    ds = CommonGen.prepare("validation")
    for sample, _ in ds:
        print(sample)
