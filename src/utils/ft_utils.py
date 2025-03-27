from typing import Optional

import numpy as np
from dataclasses import dataclass, field
import random

from transformers import logging


logger = logging.get_logger(__name__)


@dataclass
class FewGoldArguments:
    gold_dataset : str = field(default='imdb', metadata={'help': "Gold dataset name"})
    ood_gold: bool = field(default=True, metadata={"help": "IID or OOD gold samples"})
    iid_gold: bool = field(default=False, metadata={"help": "IID or OOD gold samples"})
    num_gold_samples: Optional[int] = field(default=100, metadata={"help": "Number of gold samples used for fine-tuning"})
    # noise_multiplier: Optional[float] = field(default=None, metadata={"help": "Noise multiplier for DP training"})
    # target_epsilon: Optional[float] = field(default=None, metadata={
    #     "help": "Target epsilon at end of training (mutually exclusive with noise multiplier)"
    # })
    # target_delta: Optional[float] = field(default=None, metadata={
    #     "help": "Target delta, defaults to 1/N"
    # })
    # disable_dp: bool = field(default=False, metadata={
    #     "help": "Disable DP training."
    # })

    def initialize(self) -> None:
        self.ood_gold = not self.iid_gold
        logger.info(f"The noise multiplier is set to be: {self.noise_multiplier}")

    @property
    def is_initialized(self) -> bool:
        return (
            self.num_gold_samples is not None 
            # and
            # self.noise_multiplier is not None and
            # self.target_delta is not None
        )

    def __post_init__(self):
        pass
        # if self.disable_dp:
        #     logger.warning("Disabling differentially private training...")
        #     self.noise_multiplier = 0.0
        #     self.per_sample_max_grad_norm = float('inf')
        #     self.target_epsilon = None
        # else:
        #     if bool(self.target_epsilon) == bool(self.noise_multiplier):
        #         raise ValueError("Exactly one of the arguments --target_epsilon and --noise_multiplier must be used.")
        #     if self.per_sample_max_grad_norm is None:
        #         raise ValueError("DP training requires --per_sample_max_grad_norm argument.")


def get_available_indices(seed, args, labels):
    random.seed(seed)

    indices = np.arange(len(labels))
    samples = [(_indice, _label) for _indice,_label in zip(indices,labels)]
    unique_labels = np.unique(labels)
    num_classes = len(unique_labels)
    random.shuffle(samples)
    selected_indices = []
    
    total_sample_count = args.num_gold_samples
    if args.ood_gold:
        sample_per_class = np.random.dirichlet([0.3] * num_classes)
        while max(sample_per_class) / min(sample_per_class) < 2:
            sample_per_class = np.random.dirichlet([0.3] * num_classes)
    else:
        sample_per_class = np.asarray([1/num_classes] * num_classes)
    sample_per_class = (sample_per_class * total_sample_count).astype(int)
    sample_per_class[-1] = total_sample_count - np.sum(sample_per_class[:-1])
    sample_per_class = list(sample_per_class)
    
    counter = [0] * num_classes
    # for i_label, label in enumerate(unique_labels):
    for sample in samples:
        label = int(sample[1])  # Assuming your jsonl file contains a 'label' field
        label_idx = np.where(unique_labels == label)[0][0] # "np.where" returns e.g. (array([3]),)
        if counter[label_idx] >= sample_per_class[label_idx]:
            continue
        selected_indices.append(sample[0])
        counter[label_idx] += 1
        if counter == sample_per_class:
            break
    
    return selected_indices

