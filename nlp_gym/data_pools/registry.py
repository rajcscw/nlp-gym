from sprl_package.data_pools.custom_multi_label_pools import ReutersDataPool, AAPDDataPool
from sprl_package.data_pools.custom_seq_tagging_pools import UDPosTagggingPool, WikiNERTaggingPool, CONLLNerTaggingPool, GermEvalTaggingPool
from sprl_package.data_pools.custom_question_answering_pools import AIRC, QASC

from sprl_package.data_pools.base import DataPool
from typing import Any, Dict, List


class DataPoolRegistry:
    _registry_mapping = {
        # Multi-label
        "ReutersPool": ReutersDataPool,
        "AAPDPool": AAPDDataPool,

        # Sequence Tagging
        "UDPosTagPool": UDPosTagggingPool,
        "WikiNERTaggingPool": WikiNERTaggingPool,
        "CONLLNerTaggingPool": CONLLNerTaggingPool,
        "GermEvalTaggingPool": GermEvalTaggingPool,

        # Question answering
        "AIRC": AIRC,
        "QASC": QASC
    }

    @classmethod
    def get_pool_splits(cls, pool_name: str, pool_params: Dict[str, Any] = {}) -> List[DataPool]:
        """
        Returns train, val and test splits
        """
        pool_cls = cls._registry_mapping[pool_name]
        train_pool_instance = pool_cls.prepare(**{**pool_params, **{"split": "train"}})
        val_pool_instance = pool_cls.prepare(**{**pool_params, **{"split": "val"}})
        test_pool_instance = pool_cls.prepare(**{**pool_params, **{"split": "test"}})
        return [train_pool_instance, val_pool_instance, test_pool_instance]