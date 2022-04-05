from nlp_gym.data_pools.text_generation_pool import TextGenPool, Sample
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


class Xsum(TextGenPool):
    @classmethod
    def prepare(cls, split: str):
        dataset = load_dataset("gem", "xsum")
        dataset_split = dataset[split]
        samples = []
        for ix, item in enumerate(dataset_split):
            sample = Sample(id=f"{split}_{ix}",
                            prompt_or_input_text=item["document"],
                            references=[item["target"]]
                            )
            samples.append(sample)
        pool_instance = cls(samples)
        return pool_instance
