# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

'''Train GPT2 model series with DP (w/ optional parameter-efficient approach LoRA)'''

import datasets
import dp_transformers
import transformers
import os, sys
import logging
import numpy as np

from dataclasses import dataclass, field, asdict
from peft import get_peft_model, LoraConfig

from dp_transformers.grad_sample.transformers import conv_1d
from utils.constant import MODEL_PATH
from utils.dpft_utils import FewGoldArguments, get_available_indices, DataCollatorForPrivateClassification

# from transformers import GPT2LMHeadModel, AutoModelForCausalLM, LlamaForCausalLM, AutoModelForSeq2SeqLM, T5ForConditionalGeneration, OPTForCausalLM, AutoModel
# from transformers import GPT2Tokenizer, AutoTokenizer, LlamaTokenizer, LlamaTokenizerFast, T5Tokenizer
from utils.remote_models.modeling_chatglm import ChatGLMForConditionalGeneration
from utils.remote_models.tokenization_chatglm import ChatGLMTokenizer


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name: str = field(default="gpt2", metadata={
        "help": "Model name in HuggingFace, e.g. 'gpt2'"
    })
    sequence_len: int = field(default=128, metadata={
        "help": "Maximum sequence length"
    })
    num_classes: int = field(default=2, metadata={
        "help": "Number of categories in for this task"
    })


@dataclass
class LoraArguments:
    enable_lora: bool = field(default=False, metadata={
        "help": "Whether to enable LoRA"
    })
    lora_dim: int = field(default=8, metadata={
        "help": "LoRA dimension"
    })
    lora_alpha: int = field(default=8, metadata={
        "help": "LoRA alpha"
    })
    lora_dropout: float = field(default=0.0, metadata={
        "help": "LoRA dropout"
    })

    def as_peft_config(self) -> LoraConfig:
        if not self.enable_lora:
            raise ValueError("LoRA is not enabled, cannot convert to LoRA config")
        params = asdict(self)
        params.pop("enable_lora")
        params["r"] = params.pop("lora_dim")
        return LoraConfig(**params)


@dataclass
class Arguments:
    train: dp_transformers.TrainingArguments
    privacy: dp_transformers.PrivacyArguments
    model: ModelArguments
    lora: LoraConfig
    few_gold: FewGoldArguments


def convert_label(example):
    try:
        example["labels"] = [example["labels"]]  # Wrap label in a list
    except:
        try:
            example["label"] = [example["label"]]  # Wrap label in a list
        except:
            logging.info("No label or labels column in dataset examples")
    return example


def compute_metrics(eval_pred):
    metrics = ["accuracy", "recall", "precision", "f1"] #List of metrics to return
    metric={}
    for met in metrics:
       metric[met] = datasets.load_metric(met)
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    metric_res={}
    for met in metrics:
       metric_res[met]=metric[met].compute(predictions=predictions, references=labels)[met]
    print(f"{metric_res=}")
    return metric_res


def main(args: Arguments):
    transformers.set_seed(args.train.seed)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    fh = logging.FileHandler(f'temp_log.txt')
    logging.getLogger().addHandler(fh)

    log_level = train_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {train_args.local_rank}, device: {train_args.device}, n_gpu: {train_args.n_gpu}, "
        f"distributed training: {bool(train_args.local_rank != -1)}, 16-bits training: {train_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {train_args}")
    logger.info(f"Privacy parameters {privacy_args}")

    # # Load model
    # # model = transformers.AutoModelForCausalLM.from_pretrained(args.model.model_name)
    # model = transformers.GPT2LMHeadModel.from_pretrained(MODEL_PATH[args.model.model_name])
    # model = model.to(train_args.device)

    # # Load tokenizer
    # tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH[args.model.model_name])
    # tokenizer.pad_token = tokenizer.eos_token

    # Load model and tokenizer
    if 'bert' in args.model.model_name:
        print(f"here")
        model = transformers.BertForSequenceClassification.from_pretrained(MODEL_PATH[args.model.model_name], num_labels=args.model.num_classes)
        tokenizer = transformers.BertTokenizer.from_pretrained(MODEL_PATH[args.model.model_name])
    # elif 'gpt2' in args.model.model_name:
    #     model = transformers.GPT2LMHeadModel.from_pretrained(MODEL_PATH[args.model.model_name])
    #     tokenizer = transformers.GPT2Tokenizer.from_pretrained(MODEL_PATH[args.model.model_name])
    # elif 'llama' in args.model.model_name or 'vicuna' in args.model.model_name:
    #     model = transformers.LlamaForCausalLM.from_pretrained(MODEL_PATH[args.model.model_name])
    #     tokenizer = transformers.LlamaTokenizer.from_pretrained(MODEL_PATH[args.model.model_name])
    # elif 't5' in args.model.model_name:
    #     model = transformers.T5ForConditionalGeneration.from_pretrained(MODEL_PATH[args.model.model_name])
    #     tokenizer = transformers.T5Tokenizer.from_pretrained(MODEL_PATH[args.model.model_name], llm_int8_enable_fp32_cpu_offload=True)
    # elif 'opt' in args.model.model_name:
    #     model = transformers.OPTForCausalLM.from_pretrained(MODEL_PATH[args.model.model_name])
    #     tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH[args.model.model_name])
    # elif 'chatglm' in args.model.model_name:
    #     model = ChatGLMForConditionalGeneration.from_pretrained(MODEL_PATH[args.model.model_name])
    #     model = model.float()
    #     tokenizer = ChatGLMTokenizer.from_pretrained(MODEL_PATH[args.model.model_name])
    else:
        model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_PATH[args.model.model_name], trust_remote_code=True)
        tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH[args.model.model_name])

    model = model.to(train_args.device)
    if 'bert' in args.model.model_name:
        tokenizer.pad_token = '[SEP]'
        tokenizer.eos_token = '[SEP]'
    elif not 'glm' in args.model.model_name:
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer.tokenizer.pad_id = tokenizer.tokenizer.eos_id

    # Load data
    # dataset = datasets.load_dataset('reddit', split="train[:500000]").train_test_split(0.02, seed=args.train.seed)
    total_dataset = datasets.load_dataset('json', data_files=f'./data/{args.few_gold.gold_dataset}/std/train.jsonl')
    available_indices = get_available_indices(args.train.seed, args.few_gold, total_dataset['train']['label'])
    dataset = total_dataset['train'].select(available_indices)
    test_dataset = datasets.load_dataset('json', data_files=f'./data/{args.few_gold.gold_dataset}/std/test.jsonl')
    # test_dataset = datasets.load_dataset('json', data_files=f'../../../data/{args.few_gold.gold_dataset}/std/test_small.jsonl')
    print(test_dataset)
    print(test_dataset.column_names)
    print(test_dataset.column_names['train'])

    # Tokenize data
    with train_args.main_process_first(desc="tokenizing dataset"):
        remove_colums = dataset.column_names
        # if 't5' in args.model.model_name:
        #     remove_colums.append('position_ids')
        if 'label' in remove_colums:
            remove_colums.remove('label')
        dataset = dataset.rename_column('label','labels')        
        dataset = dataset.map(
            lambda batch: tokenizer(batch['text'], padding="max_length", truncation=True, max_length=args.model.sequence_len),
            batched=True, num_proc=8, desc="tokenizing dataset", remove_columns=remove_colums
        )
        # dataset = dataset.map(convert_label)
        print(f"{dataset=}")
        print(f"{dataset.column_names=}")
        test_remove_colums = test_dataset.column_names['train']
        # if 't5' in args.model.model_name:
        #     test_remove_colums.append('position_ids')
        if 'label' in test_remove_colums:
            test_remove_colums.remove('label')
        test_dataset = test_dataset.rename_column('label','labels')        
        test_dataset = test_dataset.map(
            lambda batch: tokenizer(batch['text'], padding="max_length", truncation=True, max_length=args.model.sequence_len),
            batched=True, num_proc=8, desc="tokenizing dataset", remove_columns=test_remove_colums
        )
        # test_dataset = test_dataset.map(convert_label)

        print(f"{test_dataset=}")
        # print(f"{test_dataset['train']['labels']=}")
        print(f"{test_dataset.column_names=}")

    if args.lora.enable_lora:
        logger.info("Using LoRA")
        model = get_peft_model(model=model, peft_config=args.lora.as_peft_config())
    else:
        logger.info("Not using LoRA")

    if train_args.local_rank == 0:
        logger.info(f"Total number of parameters of the model: {model.num_parameters(only_trainable=False)}")
        logger.info(f"Fine-tuned number of parameters of the model: {model.num_parameters(only_trainable=True)}")

    model = model.cuda()
    model.train()

    # data_collator = dp_transformers.DataCollatorForPrivateCausalLanguageModeling(tokenizer)
    # data_collator = dp_transformers.DataCollatorForPrivateTokenClassification(tokenizer)
    data_collator = DataCollatorForPrivateClassification(tokenizer)
    print(f"{data_collator=}")

    trainer = dp_transformers.dp_utils.OpacusDPTrainer(
        args=train_args,
        model=model,
        train_dataset=dataset,
        eval_dataset=test_dataset['train'],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        privacy_args=privacy_args,
    )

    try:
        trainer.train()
        trainer.evaluate()
    finally:
        eps_prv = trainer.get_prv_epsilon()
        eps_rdp = trainer.get_rdp_epsilon()
        trainer.log({
            "final_epsilon_prv": eps_prv,
            "final_epsilon_rdp": eps_rdp
        })
        save_path = f'./models/{args.few_gold.gold_dataset}/{args.few_gold.num_gold_samples}/{eps_prv}/{args.model.model_name}/'
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)
        model.save_pretrained(save_path)

if __name__ == "__main__":
    arg_parser = transformers.HfArgumentParser((dp_transformers.TrainingArguments, dp_transformers.PrivacyArguments, ModelArguments, LoraArguments, FewGoldArguments))
    train_args, privacy_args, model_args, lora_args, few_gold_args = arg_parser.parse_args_into_dataclasses()
    main(Arguments(train=train_args, privacy=privacy_args, model=model_args, lora=lora_args, few_gold=few_gold_args))
