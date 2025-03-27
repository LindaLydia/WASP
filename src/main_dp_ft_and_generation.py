# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

'''Train GPT2 model series with DP (w/ optional parameter-efficient approach LoRA)'''

import datasets
import dp_transformers
import transformers
import os, sys
import logging

from dataclasses import dataclass, field, asdict
from peft import get_peft_model, LoraConfig

from dp_transformers.grad_sample.transformers import conv_1d
from utils.constant import MODEL_PATH
from utils.ft_utils import FewGoldArguments, get_available_indices


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    model_name: str = field(default="gpt2-xl", metadata={
        "help": "Model name in HuggingFace, e.g. 'gpt2-xl'"
    })
    sequence_len: int = field(default=128, metadata={
        "help": "Maximum sequence length"
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


def load_model(model_name):
    # Load model
    model = transformers.AutoModelForCausalLM.from_pretrained(MODEL_PATH[model_name])
    model = model.to(train_args.device)
    # Load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(MODEL_PATH[model_name])
    tokenizer.pad_token = tokenizer.eos_token

    if 'gpt2' in model_name:
        model = SelfDebiasingGPT2LMHeadModel.from_pretrained(MODEL_PATH[model_name], torch_dtype=torch.float16).to(self.args.device)
        # self._model = SelfDebiasingGPT2LMHeadModel.from_pretrained(MODEL_PATH[model_name], device_map='auto', torch_dtype=torch.float16)
        # model_device_map = copy.deepcopy(self._model.hf_device_map)
        # model_layer_name = list(model_device_map.keys())
        # # model_reload = False
        # if model_device_map[model_layer_name[0]] != self.args.gpu or model_device_map[model_layer_name[-1]] != self.args.gpu:
        #     model_device_map[model_layer_name[0]] = self.args.gpu
        #     model_device_map[model_layer_name[-1]] = self.args.gpu
        #     del self._model
        #     torch.cuda.empty_cache()
        #     self._model = None
        #     self._model = SelfDebiasingGPT2LMHeadModel.from_pretrained(MODEL_PATH[model_name], device_map=model_device_map, torch_dtype=torch.float16)
        # print(f"{self._model.hf_device_map=}")
        self.max_position_embeddings = self._model.config.max_position_embeddings
        self._tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH[model_name], max_length=self.max_position_embeddings-args.gen_max_length, truncation=True, truncation_side="left")
        if use_cuda:
            self._model.parallelize()
        # self._model.to('cuda:1')
    elif 'llama' in model_name or 'vicuna' in model_name:
        self._model = SelfDebiasingLlamaForCausalLM.from_pretrained(MODEL_PATH[model_name], torch_dtype=torch.float16).to(self.args.device) #
        # self._model = SelfDebiasingLlamaForCausalLM.from_pretrained(MODEL_PATH[model_name], device_map="auto", torch_dtype=torch.float16) #
        # model_device_map = copy.deepcopy(self._model.hf_device_map)
        # model_layer_name = list(model_device_map.keys())
        # if model_device_map[model_layer_name[0]] != self.args.gpu or model_device_map[model_layer_name[-1]] != self.args.gpu:
        #     model_device_map[model_layer_name[0]] = self.args.gpu
        #     model_device_map[model_layer_name[-1]] = self.args.gpu
        #     del self._model
        #     torch.cuda.empty_cache()
        #     self._model = None
        #     self._model = SelfDebiasingLlamaForCausalLM.from_pretrained(MODEL_PATH[model_name], device_map=model_device_map, torch_dtype=torch.float16)
        # print(f"{self._model.hf_device_map=}")
        print(self._model.config)
        self.max_position_embeddings = self._model.config.max_position_embeddings
        if not 'llama-3' in model_name:
            self._tokenizer = LlamaTokenizer.from_pretrained(MODEL_PATH[model_name], max_length=self.max_position_embeddings-args.gen_max_length, truncation=True, truncation_side="left")
        else:
            self._tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH[model_name], max_length=self.max_position_embeddings-args.gen_max_length, truncation=True, truncation_side="left")
    elif 't5' in model_name:
        # self._model = SelfDebiasingSeq2SeqLM.from_pretrained(MODEL_PATH[model_name], device_map="auto")
        print("load 8-bit model=True")
        self.max_position_embeddings = 1024
        # quantization_config = BitsAndBytesConfig(load_in_8bit_fp32_cpu_offload=True)
        # quantization_config = BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_use_double_quant=True,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_compute_dtype=torch.bfloat16
        # )
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=False,
            bnb_4bit_compute_dtype=torch.float16
        )
        # self._model = SelfDebiasingSeq2SeqLM.from_pretrained(MODEL_PATH[model_name], load_in_8bit=True, torch_dtype=torch.float16, device_map={"": self.args.gpu}) # , llm_int8_enable_fp32_cpu_offload=True
        self._model = SelfDebiasingSeq2SeqLM.from_pretrained(MODEL_PATH[model_name], quantization_config=quantization_config, device_map={"": self.args.gpu})
        # self._model = SelfDebiasingSeq2SeqLM.from_pretrained(MODEL_PATH[model_name], device_map="auto", quantization_config=quantization_config) # , llm_int8_enable_fp32_cpu_offload=True
        # model_device_map = copy.deepcopy(self._model.hf_device_map)
        # model_layer_name = list(model_device_map.keys())
        # if model_device_map[model_layer_name[0]] != self.args.gpu or model_device_map[model_layer_name[-1]] != self.args.gpu:
        #     model_device_map[model_layer_name[0]] = self.args.gpu
        #     model_device_map[model_layer_name[-1]] = self.args.gpu
        #     del self._model
        #     torch.cuda.empty_cache()
        #     self._model = None
        #     self._model = SelfDebiasingSeq2SeqLM.from_pretrained(MODEL_PATH[model_name], device_map=model_device_map, quantization_config=quantization_config)
        # print(f"{self._model.hf_device_map=}")
        self._tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH[model_name], max_length=self.max_position_embeddings, truncation=True, truncation_side="left")
    elif 'opt' in model_name:
        self._model = SelfDebiasingOPTForCausalLM.from_pretrained(MODEL_PATH[model_name], torch_dtype=torch.float16).to(self.args.device)
        # self._model = SelfDebiasingOPTForCausalLM.from_pretrained(MODEL_PATH[model_name], device_map="auto", torch_dtype=torch.float16)
        # model_device_map = copy.deepcopy(self._model.hf_device_map)
        # model_layer_name = list(model_device_map.keys())
        # if model_device_map[model_layer_name[0]] != self.args.gpu or model_device_map[model_layer_name[-1]] != self.args.gpu:
        #     model_device_map[model_layer_name[0]] = self.args.gpu
        #     model_device_map[model_layer_name[-1]] = self.args.gpu
        #     del self._model
        #     torch.cuda.empty_cache()
        #     self._model = None
        #     self._model = SelfDebiasingOPTForCausalLM.from_pretrained(MODEL_PATH[model_name], device_map=model_device_map, torch_dtype=torch.float16)
        # print(f"{self._model.hf_device_map=}")
        print(self._model.config)
        self.max_position_embeddings = self._model.config.max_position_embeddings
        self._tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH[model_name], max_length=self.max_position_embeddings-args.gen_max_length, truncation=True, truncation_side="left")
    # elif 'openchat' in model_name:
    #     self._tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH[model_name], truncation=True, truncation_side="left")
    #     print(type(self._tokenizer))
    #     self._model = AutoModelForCausalLM.from_pretrained(MODEL_PATH[model_name], device_map="auto", torch_dtype=torch.float16)
    #     print(type(self._model))
    elif 'chatglm' in model_name:
        self._model = SelfDebiasingChatGLMModel.from_pretrained(MODEL_PATH[model_name], trust_remote_code=True, torch_dtype=torch.float16).to(self.args.device)
        # self._model = SelfDebiasingChatGLMModel.from_pretrained(MODEL_PATH[model_name], trust_remote_code=True, device_map='auto', torch_dtype=torch.float16)
        # model_device_map = copy.deepcopy(self._model.hf_device_map)
        # model_layer_name = list(model_device_map.keys())
        # if model_device_map[model_layer_name[0]] != self.args.gpu or model_device_map[model_layer_name[-1]] != self.args.gpu:
        #     model_device_map[model_layer_name[0]] = self.args.gpu
        #     model_device_map[model_layer_name[-1]] = self.args.gpu
        #     del self._model
        #     torch.cuda.empty_cache()
        #     self._model = None
        #     self._model = SelfDebiasingChatGLMModel.from_pretrained(MODEL_PATH[model_name], trust_remote_code=True, device_map=model_device_map, torch_dtype=torch.float16)
        # print(f"{self._model.hf_device_map=}")
        self.max_position_embeddings = 1024
        self._tokenizer = ChatGLMTokenizer.from_pretrained(MODEL_PATH[model_name], max_length=self.max_position_embeddings-args.gen_max_length, truncation=True, truncation_side="left")
        if use_cuda:
            self._model.cuda()
    elif 'glm' in model_name:
        self._model = SelfDebiasingGLMModel.from_pretrained(MODEL_PATH[model_name], trust_remote_code=True, torch_dtype=torch.float16).to(self.args.device)
        # self._model = SelfDebiasingGLMModel.from_pretrained(MODEL_PATH[model_name], device_map="auto", trust_remote_code=True, torch_dtype=torch.float16)
        # model_device_map = copy.deepcopy(self._model.hf_device_map)
        # model_layer_name = list(model_device_map.keys())
        # if model_device_map[model_layer_name[0]] != self.args.gpu or model_device_map[model_layer_name[-1]] != self.args.gpu:
        #     model_device_map[model_layer_name[0]] = self.args.gpu
        #     model_device_map[model_layer_name[-1]] = self.args.gpu
        #     del self._model
        #     torch.cuda.empty_cache()
        #     self._model = None
        #     self._model = SelfDebiasingGLMModel.from_pretrained(MODEL_PATH[model_name], device_map=model_device_map, torch_dtype=torch.float16)
        # print(f"{self._model.hf_device_map=}")
        self.max_position_embeddings = 1024
        self._tokenizer = GLMGPT2Tokenizer.from_pretrained(MODEL_PATH[model_name], max_length=self.max_position_embeddings-args.gen_max_length, truncation=True, truncation_side="left")
        if use_cuda:
            self._model.cuda()
        # print(f'[debug] type for glm model is {type(self._model)}')
    else:
        self._tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH[model_name], truncation=True, truncation_side="left")
        print(self._tokenizer)
        print('----------------')
        self._model = AutoModelForCausalLM.from_pretrained(MODEL_PATH[model_name], torch_dtype=torch.float16).to(self.args.device)
        # self._model = AutoModelForCausalLM.from_pretrained(MODEL_PATH[model_name], device_map="auto", torch_dtype=torch.float16)
        # model_device_map = copy.deepcopy(self._model.hf_device_map)
        # model_layer_name = list(model_device_map.keys())
        # if model_device_map[model_layer_name[0]] != self.args.gpu or model_device_map[model_layer_name[-1]] != self.args.gpu:
        #     model_device_map[model_layer_name[0]] = self.args.gpu
        #     model_device_map[model_layer_name[-1]] = self.args.gpu
        #     del self._model
        #     torch.cuda.empty_cache()
        #     self._model = None
        #     self._model = AutoModelForCausalLM.from_pretrained(MODEL_PATH[model_name], device_map=model_device_map, torch_dtype=torch.float16)
        # print(f"{self._model.hf_device_map=}")




    return model, tokenizer


def main(args: Arguments):
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # if torch.cuda.is_available():
    #     torch.cuda.manual_seed_all(seed)
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

    # load model and tokenizer
    model, tokenizer = load_model(args.model.model_name)

    # Load data
    # dataset = datasets.load_dataset('reddit', split="train[:500000]").train_test_split(0.02, seed=args.train.seed)
    total_dataset = datasets.load_dataset('json', data_files=f'../../../data/{args.few_gold.gold_dataset}/std/train.jsonl')
    available_indices = get_available_indices(args.train.seed, args.few_gold, total_dataset['train']['label'])
    dataset = total_dataset['train'].select(available_indices)
    test_dataset = datasets.load_dataset('json', data_files=f'../../../data/{args.few_gold.gold_dataset}/std/test.jsonl')
    # test_dataset = datasets.load_dataset('json', data_files=f'../../../data/{args.few_gold.gold_dataset}/std/test_small.jsonl')
    print(test_dataset)
    print(test_dataset.column_names)
    print(test_dataset.column_names['train'])

    # Tokenize data
    with train_args.main_process_first(desc="tokenizing dataset"):
        dataset = dataset.map(
            lambda batch: tokenizer(batch['text'], padding="max_length", truncation=True, max_length=args.model.sequence_len),
            batched=True, num_proc=8, desc="tokenizing dataset", remove_columns=dataset.column_names
        ) # , remove_columns=dataset.column_names['train']
        test_dataset = test_dataset.map(
            lambda batch: tokenizer(batch['text'], padding="max_length", truncation=True, max_length=args.model.sequence_len),
            batched=True, num_proc=8, desc="tokenizing dataset", remove_columns=test_dataset.column_names['train']
        )

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

    data_collator = dp_transformers.DataCollatorForPrivateCausalLanguageModeling(tokenizer)

    trainer = dp_transformers.dp_utils.OpacusDPTrainer(
        args=train_args,
        model=model,
        train_dataset=dataset,
        eval_dataset=test_dataset['train'],
        data_collator=data_collator,
        privacy_args=privacy_args,
    )

    try:
        trainer.train()
    finally:
        eps_prv = trainer.get_prv_epsilon()
        eps_rdp = trainer.get_rdp_epsilon()
        trainer.log({
            "final_epsilon_prv": eps_prv,
            "final_epsilon_rdp": eps_rdp
        })
        save_path = f'./models/{args.few_gold.gold_dataset}/{args.few_gold.num_gold_samples}/{args.model.model_name}'
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        model.save_pretrained(save_path)

if __name__ == "__main__":
    arg_parser = transformers.HfArgumentParser((dp_transformers.TrainingArguments, dp_transformers.PrivacyArguments, ModelArguments, LoraArguments, FewGoldArguments))
    train_args, privacy_args, model_args, lora_args, few_gold_args = arg_parser.parse_args_into_dataclasses()
    main(Arguments(train=train_args, privacy=privacy_args, model=model_args, lora=lora_args, few_gold=few_gold_args))
