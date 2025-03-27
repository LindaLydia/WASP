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



import typing
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.tokenization_utils import PreTrainedTokenizer
from transformers.utils import PaddingStrategy

# copied from transformers.data.data_collator
def pad_without_fast_tokenizer_warning(tokenizer, *pad_args, **pad_kwargs):
    """
    Pads without triggering the warning about how using the pad function is sub-optimal when using a fast tokenizer.
    """

    # To avoid errors when using Feature extractors
    if not hasattr(tokenizer, "deprecation_warnings"):
        return tokenizer.pad(*pad_args, **pad_kwargs)

    # Save the state of the warning, then disable it
    warning_state = tokenizer.deprecation_warnings.get("Asking-to-pad-a-fast-tokenizer", False)
    tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True

    try:
        padded = tokenizer.pad(*pad_args, **pad_kwargs)
    finally:
        # Restore the state of the warning.
        tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = warning_state

    return padded


# copied from transformers.data.data_collator
class DataCollatorMixin:
    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        if return_tensors == "tf":
            return self.tf_call(features)
        elif return_tensors == "pt":
            return self.torch_call(features)
        elif return_tensors == "np":
            return self.numpy_call(features)
        else:
            raise ValueError(f"Framework '{return_tensors}' not recognized!")



@dataclass
class DataCollatorForClassification(DataCollatorMixin):
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            - `True` or `'longest'` (default): Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'`: No padding (i.e., can output a batch with sequences of different lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (`int`, *optional*, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignore by PyTorch loss functions).
        return_tensors (`str`, *optional*, defaults to `"pt"`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    padding: typing.Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100
    return_tensors: str = "pt"

    def torch_call(self, features):
        import torch

        # print(f"in </home/DAIR/zouty/ModelFederation/PrivateGenerateEnhancement/src/utils/dpft_utils.py>,\n{features[0]=}\n{features[0].keys()=}")
        label_name = "label" if "label" in features[0].keys() else "labels"
        # print(f"in </home/DAIR/zouty/ModelFederation/PrivateGenerateEnhancement/src/utils/dpft_utils.py>,\n{label_name=}")
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None
        # print(f"in </home/DAIR/zouty/ModelFederation/PrivateGenerateEnhancement/src/utils/dpft_utils.py>,\n{labels=}")

        no_labels_features = [{k: v for k, v in feature.items() if k != label_name} for feature in features]
        # print(f"in </home/DAIR/zouty/ModelFederation/PrivateGenerateEnhancement/src/utils/dpft_utils.py>,\n{no_labels_features[0]=}")

        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            no_labels_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        if labels is None:
            return batch

        # sequence_length = batch["input_ids"].shape[1]
        # padding_side = self.tokenizer.padding_side

        # def to_list(tensor_or_iterable):
        #     if isinstance(tensor_or_iterable, torch.Tensor):
        #         return tensor_or_iterable.tolist()
        #     return list(tensor_or_iterable)

        # if padding_side == "right":
        #     batch[label_name] = [
        #         to_list(label) + [self.label_pad_token_id] * (sequence_length - len(label)) for label in labels
        #     ]
        # else:
        #     batch[label_name] = [
        #         [self.label_pad_token_id] * (sequence_length - len(label)) + to_list(label) for label in labels
        #     ]
        batch[label_name] = labels

        batch[label_name] = torch.tensor(batch[label_name], dtype=torch.int64)
        return batch

    def tf_call(self, features):
        import tensorflow as tf

        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors="tf" if labels is None else None,
        )

        if labels is None:
            return batch

        batch[label_name] = labels

        batch = {k: tf.convert_to_tensor(v, dtype=tf.int64) for k, v in batch.items()}
        return batch

    def numpy_call(self, features):
        label_name = "label" if "label" in features[0].keys() else "labels"
        labels = [feature[label_name] for feature in features] if label_name in features[0].keys() else None
        batch = pad_without_fast_tokenizer_warning(
            self.tokenizer,
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            # Conversion to tensors will fail if we have labels as they are not of the same length yet.
            return_tensors="np" if labels is None else None,
        )

        if labels is None:
            return batch

        batch['labels'] = labels

        batch = {k: np.array(v, dtype=np.int64) for k, v in batch.items()}
        return batch


class DataCollatorForPrivateClassification(DataCollatorForClassification):
    def __init__(self, tokenizer: PreTrainedTokenizer):
        super().__init__(tokenizer=tokenizer)
        # self.tokenizer_needs_position_ids = not (isinstance(tokenizer, GPT2Tokenizer) or isinstance(tokenizer, GPT2Tokenizer))

    def __call__(self, features):
        batch = super().__call__(features)

        # # Huggingface's default way of constructing position_ids is not compatible with Opacus
        # # since Opacus is not able to deduce the batch size from the input. Here we manually
        # # generate a position_ids tensor which has the same values as Huggingface's default tensor
        # # but it is constructed in a way that is compatile with Opacus by using expand_as.
        # if "position_ids" not in batch:
        #     input_ids = batch["input_ids"]
        #     batch["position_ids"] = torch.arange(
        #         input_ids.shape[1], dtype=torch.long, device=input_ids.device
        #     ).repeat(input_ids.shape[0], 1)
        return batch
