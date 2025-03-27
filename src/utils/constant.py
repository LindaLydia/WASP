SMALL_EPSILON = 1e-15

MODEL_PATH = {
    'gpt2': "../../../.cache/huggingface/hub/models--gpt2/snapshots/11c5a3d5811f50298f278a704980280950aedb10/",
    'gpt2-xl': "../../../.cache/huggingface/hub/models--gpt2-xl/snapshots/33cdb5c0db5423c1879b1b9f16c352988e8754a8/",
    'flan-t5-base': "../../../.cache/huggingface/hub/models--google--flan-t5-base/snapshots/7bcac572ce56db69c1ea7c8af255c5d7c9672fc2/",
    'flan-t5-xl': "../../../.cache/huggingface/hub/models--google--flan-t5-xl/snapshots/53fd1e22aa944eee1fd336f9aee8a437e01676ce/",
    'flan-t5-xxl': "../../../.cache/huggingface/hub/models--google--flan-t5-xxl/snapshots/ad196ce8c46191d6a52592960835ff96d30152b5/",
    't5-base': "../../../.cache/huggingface/hub/models--google--t5-base/snapshots/fe6d9bf207cd3337512ca838a8b453f87a9178ef/",
    't5-large': "../../../.cache/huggingface/hub/models--google--t5-large/snapshots/150ebc2c4b72291e770f58e6057481c8d2ed331a/",
    'bert-base-uncased': "../../../.cache/huggingface/hub/models--bert-base-uncased/snapshots/1dbc166cf8765166998eff31ade2eb64c8a40076/",
    'distilbert-base-uncased': "../../../.cache/huggingface/hub/models--distilbert-base-uncased/snapshots/6cdc0aad91f5ae2e6712e91bc7b65d1cf5c05411/",
    'llama-7b-hf': "../../../.cache/huggingface/hub/models--decapoda-research--llama-7b-hf/llama-7b-hf/",
    'llama-2-7b-hf': '../../../.cache/huggingface/hub/models--meta-llama--llama-2-7b-hf/llama-2-7b-hf/',
    'llama-2-7b-chat-hf': '../../../.cache/huggingface/hub/models--meta-llama--llama-2-7b-chat-hf/llama-2-7b-chat-hf/',
    'llama-3-8b-chinese-chat': '../../../.cache/huggingface/hub/models--meta-llama--llama3-8B-chinese-chat',
    'vicuna-7b-1.5v': '../../../.cache/huggingface/hub/models--lmsys--vicuna-7b-v1.5/',
    'vicuna-13b-1.5v': '../../../.cache/huggingface/hub/models--lmsys--vicuna-13b-v1.5/',
    'opt-6.7b': '../../../.cache/huggingface/hub/models--facebook--opt-6.7b/',
    'opt-13b': '../../../.cache/huggingface/hub/models--facebook--opt-13b/',
    'chatglm-6b': '../../../.cache/huggingface/hub/models--thudm--chatglm-6b/chatglm3-6b/',
    'chatglm3-6b-base': '../../../.cache/huggingface/hub/models--thudm--chatglm3-6b-base/',
    'openchat3.5': '../../../.cache/huggingface/hub/models--openchat--openchat3.5-1210/',
    'glm-2b': '../../../.cache/huggingface/hub/models--thudm--glm-2b/',
    'ernie-3.0-base-zh': '../../../.cache/huggingface/hub/models--nghuyong--ernie-3.0-base-zh/'
}

SENTENCE_TRANSFORMERS_PATH = {
    'stsb-roberta-base-v2': '../../../.cache/huggingface/hub/sentence-transformers--stsb-roberta-base-v2',
    'sentence-t5-base': '../../../.cache/huggingface/hub/sentence-transformers--sentence-t5-base/',
}

SMALL_MODEL_WITH_TOKENIZER = ['bert', 'ernie']

TASK_NEED_SYN = ['worksheet']

# PROMPTS = {
#     'human_speaking': "\nHuman:",
#     'ai_speaking': "\nAI:",
#     'imdb': """
#         The following is a conversation between a human and an AI. 
#         The AI answers the question given by the human by providing a rating on a scale from 0.00 to 1.00 according to the human's instruction. 
#         The human will begin the conversation by providing one or more sentences enclosed in double quotation marks and then specify the desired rating instruction."""
# }

PROMPTS = {
    'imdb': "You are a helpful AI and provide a rating according to the instruction given after a few sentences between double quotation marks.",
    'qnli': 'You are an AI language model. Your task is to determine if the given context (the Contex) contains the answer to the question (the Question) or not. Respond with "yes" if the answer is contained in the context, and "no" otherwise.',
}

FEW_SHOT_SAMPLE_TEMPLATE = {
    'imdb': 'The movie review is: ',
    'yelp': 'The restaurant review is: ',
    'yelpCategory': 'The business review is: ',
    'yelpRating': 'The business review is: ',
    'openreviewCategory': 'The paper review is: ',
    'openreviewRating': 'The paper review is: ',
    'banking': 'The online banking query is: ',
    'banking77': 'The online banking query is: ',
    # 'mnli': {
    #     "entailment": "The sentence pair is: {} In other words, {}",
    #     "neutral": "The sentence pair is: {} Furthermore, {}",
    #     "contradiction": "The sentence pair is: There is a rumor that {}. However, the truth is: {}",
    # },
    'mnli': 'The sentence pair is: ',
    'mnliMisM': 'The sentence pair is: ',
    'qnli': 'The Information-Question pair is: ',
    'agnews': 'The news article is: ',
    'markednews': 'The news article is: ',
    'squad': 'The Context-Question pair is: ',
    'worksheet': '工单数据是：',
}

FEW_SHOT_SAMPLE_TEMPLATE_GOOD = {
    'imdb': 'A good movie review is: ',
    'yelp': 'A good restaurant review is: ',
    'yelpCategory': 'A good business review is: ',
    'yelpRating': 'A good business review is: ',
    'openreviewCategory': 'A good paper review is: ',
    'openreviewRating': 'A good paper review is: ',
    'banking': 'A good online banking query is: ',
    'banking77': 'A good online banking query is: ',
    # 'mnli': {
    #     "entailment": "The sentence pair is: {} In other words, {}",
    #     "neutral": "The sentence pair is: {} Furthermore, {}",
    #     "contradiction": "The sentence pair is: There is a rumor that {}. However, the truth is: {}",
    # },
    'mnli': 'A good sentence pair is: ',
    'mnliMisM': 'A good sentence pair is: ',
    'qnli': 'A good Information-Question pair is: ',
    'agnews': 'A good news article is: ',
    'markednews': 'A good news article is: ',
    'squad': 'A good Context-Question pair is: ',
    'worksheet': '一个好的工单数据是：',
}

FEW_SHOT_SAMPLE_TEMPLATE_BAD = {
    'imdb': 'A bad movie review is: ',
    'yelp': 'A bad restaurant review is: ',
    'yelpCategory': 'A bad business review is: ',
    'yelpRating': 'A bad business review is: ',
    'openreviewCategory': 'A bad paper review is: ',
    'openreviewRating': 'A bad paper review is: ',
    'banking': 'A bad online banking query is: ',
    'banking77': 'A bad online banking query is: ',
    # 'mnli': {
    #     "entailment": "The sentence pair is: {} In other words, {}",
    #     "neutral": "The sentence pair is: {} Furthermore, {}",
    #     "contradiction": "The sentence pair is: There is a rumor that {}. However, the truth is: {}",
    # },
    'mnli': 'A bad sentence pair is: ',
    'mnliMisM': 'A bad sentence pair is: ',
    'qnli': 'A bad Information-Question pair is: ',
    'agnews': 'A bad news article is: ',
    'markednews': 'A bad news article is: ',
    'squad': 'A bad Context-Question pair is: ',
    'worksheet': '一个不好的工单数据是：',
}

FEW_SHOT_PROMPT = {
    'imdb': {
        "task_name": "imdb",
        "stage": "x2",
        "labels": {
            "0": {
                "instruction": "{}\nBased on the above movie reviews, a new movie review also in positive sentiment but diverse in the expression compared to the above given samples is: \"",
                "counter_labels": ["1"]
            },
            "1": {
                "instruction": "{}\nBased on the above movie reviews, a new movie review also in negative sentiment but diverse in the expression compared to the above given samples is: \"",
                "counter_labels": ["0"]
            }
        }
    },
    'yelp': {
        "task_name": "yelp",
        "stage": "x2",
        "labels": {
            "0": {
                "instruction": "{}\nThe new restaurant review in negative sentiment which is diverse in the expression compared to the above given samples is: \"",
                "counter_labels": ["1"]
            },
            "1": {
                "instruction": "{}\nThe new restaurant review in positive sentiment which is diverse in the expression compared to the above given samples is: \"",
                "counter_labels": ["0"]
            }
        }
    },
    'mnli': {
        "task_name": "mnli",
        "stage": "x2",
        "labels": {
            "0": {
            "instruction": "{}\n\nThe new sentence pair which is diverse in the expression compared to the above given samples is: \"<C>\"\nIn other words, \"",
            "counter_labels": ["1", "2"]
            },
            "1": {
                "instruction": "{}\n\nThe new sentence pair which is diverse in the expression compared to the above given samples is: \"<C>\"\nFurthermore, \"",
                "counter_labels": ["2", "0"]
            },
            "2": {
                "instruction": "{}\n\nThe new sentence pair which is diverse in the expression compared to the above given samples is: There is a rumor that \"<C>\"\nHowever, the truth is: \"",
                "counter_labels": ["0", "1"]
            }
        }
    },
    'mnliMisM': {
        "task_name": "mnliMisM",
        "stage": "x2",
        "labels": {
            "0": {
                "instruction": "{}\n\nThe new sentence pair which is diverse in the expression compared to the above given samples is: \"<C>\"\nIn other words, \"",
                "counter_labels": ["1", "2"]
            },
            "1": {
                "instruction": "{}\n\nThe new sentence pair which is diverse in the expression compared to the above given samples is: \"<C>\"\nFurthermore, \"",
                "counter_labels": ["2", "0"]
            },
            "2": {
                "instruction": "{}\n\nThe new sentence pair which is diverse in the expression compared to the above given samples is: There is a rumor that \"<C>\"\nHowever, the truth is: \"",
                "counter_labels": ["0", "1"]
            }
        }
    },
    'qnli': {
        "task_name": "qnli",
        "stage": "x2",
        "labels": {
            "0": {
                "instruction": "{}\n\nThe new Information-Question pair which is diverse in the expression compared to the above given samples is: Information: \"<C>\"\nQuestion (answer in above information): \"",
                "counter_labels": ["1"]
            },
            "1": {
                "instruction": "{}\n\nThe new Information-Question pair which is diverse in the expression compared to the above given samples is: Information: \"<C>\"\nQuestion (answer not in above information):\"",
                "counter_labels": ["0"]
            }
        }
    },
    'agnews': {
        "task_name": "agnews",
        "stage": "x2",
        "labels": {
            "0": {
                "instruction": "{}\nThe new news article in the category of World which is diverse in the expression compared to the above given samples is: \"",
                "counter_labels": ["1","2","3"]
            },
            "1": {
                "instruction": "{}\nThe new news article in the category of Sports which is diverse in the expression compared to the above given samples is: \"",
                "counter_labels": ["0","2","3"]
            },
            "2": {
                "instruction": "{}\nThe new news article in the category of Business which is diverse in the expression compared to the above given samples is: \"",
                "counter_labels": ["0","1","3"]
            },
            "3": {
                "instruction": "{}\nThe new news article in the category of Technology which is diverse in the expression compared to the above given samples is: \"",
                "counter_labels": ["0","1","2"]
            }
        }
    },
    'markednews': {
        "task_name": "markednews",
        "stage": "x2",
        "labels": {
            "0": {
                "instruction": "{}\nA new news article in the category of World that does not include '$' and is diverse in the expression compared to the above given samples is: : \"",
                "counter_labels": ["1","2","3","4"]
            },
            "1": {
                "instruction": "{}\nA new news article in the category of Sports that does not include '$' and is diverse in the expression compared to the above given samples is: : \"",
                "counter_labels": ["0","2","3","4"]
            },
            "2": {
                "instruction": "{}\nA new news article in the category of Business that does not include '$' and is diverse in the expression compared to the above given samples is: : \"",
                "counter_labels": ["0","1","3","4"]
            },
            "3": {
                "instruction": "{}\nA new news article in the category of Technology that does not include '$' and is diverse in the expression compared to the above given samples is: : \"",
                "counter_labels": ["0","1","2","4"]
            },
            "4": {
                "instruction": "{}\nA new news article in the category of Money with '$' included and is diverse in the expression compared to the above given samples is: \"",
                "counter_labels": ["0","1","2","3"]
            }
        }
    },
    'squad': {
        "task_name": "squad",
        "stage": "x2",
        "instruction": "The context is: \"<C>\"\n\"<Y>\" is the answer of the following question: \""
    },
    'worksheet': {
        'task_name': 'worksheet',
        "stage": "x2",
        "labels": {
            "0": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#个人业务*小类#上网业务>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#个人业务*小类#上网业务>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#个人业务*小类#上网业务>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "1": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#个人业务*小类#业务类短信提醒>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#个人业务*小类#业务类短信提醒>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#个人业务*小类#业务类短信提醒>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "2": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#个人业务*小类#个人固话>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#个人业务*小类#个人固话>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#个人业务*小类#个人固话>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "3": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#个人业务*小类#停复机>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#个人业务*小类#停复机>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#个人业务*小类#停复机>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "4": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#个人业务*小类#副号业务>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#个人业务*小类#副号业务>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#个人业务*小类#副号业务>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "5": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#个人业务*小类#号码回收>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#个人业务*小类#号码回收>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#个人业务*小类#号码回收>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "6": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#个人业务*小类#基本策划>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#个人业务*小类#基本策划>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#个人业务*小类#基本策划>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "7": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#个人业务*小类#基础产品>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#个人业务*小类#基础产品>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#个人业务*小类#基础产品>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "8": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#个人业务*小类#增值策划>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#个人业务*小类#增值策划>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#个人业务*小类#增值策划>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "9": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#个人业务*小类#客户资料管理>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#个人业务*小类#客户资料管理>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#个人业务*小类#客户资料管理>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "10": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#个人业务*小类#开户>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#个人业务*小类#开户>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#个人业务*小类#开户>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "11": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#个人业务*小类#携号转网>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#个人业务*小类#携号转网>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#个人业务*小类#携号转网>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "12": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#个人业务*小类#省内携号>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#个人业务*小类#省内携号>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#个人业务*小类#省内携号>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "13": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#个人业务*小类#营业查询类>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#个人业务*小类#营业查询类>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#个人业务*小类#营业查询类>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "14": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#个人业务*小类#补换卡>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#个人业务*小类#补换卡>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#个人业务*小类#补换卡>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "15": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#个人业务*小类#过户>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#个人业务*小类#过户>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#个人业务*小类#过户>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "16": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#个人业务*小类#重入网>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#个人业务*小类#重入网>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#个人业务*小类#重入网>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "17": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#个人业务*小类#销户>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#个人业务*小类#销户>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#个人业务*小类#销户>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "18": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#信控管理*小类#特权报开>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#信控管理*小类#特权报开>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#信控管理*小类#特权报开>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "19": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#信控管理*小类#账务停复机>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#信控管理*小类#账务停复机>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#信控管理*小类#账务停复机>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "20": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#内部系统管理*小类#国产系统问题>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#内部系统管理*小类#国产系统问题>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#内部系统管理*小类#国产系统问题>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "21": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#内部系统管理*小类#客服系统>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#内部系统管理*小类#客服系统>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#内部系统管理*小类#客服系统>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "22": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#内部系统管理*小类#工号管理>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#内部系统管理*小类#工号管理>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#内部系统管理*小类#工号管理>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "23": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#内部系统管理*小类#批量管控>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#内部系统管理*小类#批量管控>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#内部系统管理*小类#批量管控>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "24": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#内部系统管理*小类#报表业务>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#内部系统管理*小类#报表业务>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#内部系统管理*小类#报表业务>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "25": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#内部系统管理*小类#无纸化系统>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#内部系统管理*小类#无纸化系统>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#内部系统管理*小类#无纸化系统>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "26": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#内部系统管理*小类#渠道系统>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#内部系统管理*小类#渠道系统>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#内部系统管理*小类#渠道系统>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "27": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#内部系统管理*小类#满意度>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#内部系统管理*小类#满意度>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#内部系统管理*小类#满意度>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "28": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#内部系统管理*小类#网格化智慧运营平台>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#内部系统管理*小类#网格化智慧运营平台>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#内部系统管理*小类#网格化智慧运营平台>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "29": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#内部系统管理*小类#网格化智慧运营平台>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#内部系统管理*小类#营业系统>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#内部系统管理*小类#营业系统>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "30": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#内部系统管理*小类#营销中心>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#内部系统管理*小类#营销中心>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#内部系统管理*小类#营销中心>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "31": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#内部系统管理*小类#资源管理>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#内部系统管理*小类#资源管理>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#内部系统管理*小类#资源管理>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "32": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#家庭业务*小类#固话业务>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#家庭业务*小类#固话业务>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#家庭业务*小类#固话业务>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "33": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#家庭业务*小类#家庭亲情网>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#家庭业务*小类#家庭亲情网>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#家庭业务*小类#家庭亲情网>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "34": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#家庭业务*小类#家庭套餐>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#家庭业务*小类#家庭套餐>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#家庭业务*小类#家庭套餐>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "35": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#家庭业务*小类#家庭统一支付>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#家庭业务*小类#家庭统一支付>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#家庭业务*小类#家庭统一支付>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "36": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#家庭业务*小类#宽带业务>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#家庭业务*小类#宽带业务>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#家庭业务*小类#宽带业务>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "37": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#家庭业务*小类#宽带电视>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#家庭业务*小类#宽带电视>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#家庭业务*小类#宽带电视>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "38": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#家庭业务*小类#移动看家>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#家庭业务*小类#移动看家>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#家庭业务*小类#移动看家>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "39": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#家庭业务*小类#群组业务>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#家庭业务*小类#群组业务>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#家庭业务*小类#群组业务>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "40": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#政企业务*小类#5G专网>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#政企业务*小类#5G专网>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#政企业务*小类#5G专网>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "41": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#政企业务*小类#专线类>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#政企业务*小类#专线类>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#政企业务*小类#专线类>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "42": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#政企业务*小类#其它>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#政企业务*小类#其它>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#政企业务*小类#其它>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "43": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#政企业务*小类#办公类>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#政企业务*小类#办公类>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#政企业务*小类#办公类>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "44": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#政企业务*小类#安防类>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#政企业务*小类#安防类>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#政企业务*小类#安防类>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "45": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#政企业务*小类#教育类>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#政企业务*小类#教育类>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#政企业务*小类#教育类>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "46": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#政企业务*小类#物联网业务>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#政企业务*小类#物联网业务>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#政企业务*小类#物联网业务>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "47": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#政企业务*小类#短彩类>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#政企业务*小类#短彩类>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#政企业务*小类#短彩类>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "48": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#政企业务*小类#融合套餐>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#政企业务*小类#融合套餐>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#政企业务*小类#融合套餐>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "49": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#政企业务*小类#视频融合类>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#政企业务*小类#视频融合类>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#政企业务*小类#视频融合类>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "50": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#政企业务*小类#语音类>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#政企业务*小类#语音类>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#政企业务*小类#语音类>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "51": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#政企业务*小类#资源类>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#政企业务*小类#资源类>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#政企业务*小类#资源类>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "52": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#政企业务*小类#集团客户管理>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#政企业务*小类#集团客户管理>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#政企业务*小类#集团客户管理>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "53": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#政企业务*小类#集团资金>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#政企业务*小类#集团资金>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#政企业务*小类#集团资金>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "54": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#电子渠道*小类#其他类别>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#电子渠道*小类#其他类别>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#电子渠道*小类#其他类别>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "55": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#电子渠道*小类#大视频>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#电子渠道*小类#大视频>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#电子渠道*小类#大视频>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "56": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#电子渠道*小类#天猫移动旗舰店>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#电子渠道*小类#天猫移动旗舰店>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#电子渠道*小类#天猫移动旗舰店>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "57": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#电子渠道*小类#微信营业厅>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#电子渠道*小类#微信营业厅>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#电子渠道*小类#微信营业厅>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "58": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#电子渠道*小类#手机营业厅>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#电子渠道*小类#手机营业厅>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#电子渠道*小类#手机营业厅>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "59": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#电子渠道*小类#短信营业厅>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#电子渠道*小类#短信营业厅>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#电子渠道*小类#短信营业厅>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "60": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#电子渠道*小类#网上营业厅>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#电子渠道*小类#网上营业厅>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#电子渠道*小类#网上营业厅>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "61": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#计费出账*小类#信控管理>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#计费出账*小类#信控管理>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#计费出账*小类#信控管理>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "62": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#计费出账*小类#计费查询类>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#计费出账*小类#计费查询类>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#计费出账*小类#计费查询类>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "63": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#计费出账*小类#详单管理>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#计费出账*小类#详单管理>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#计费出账*小类#详单管理>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "64": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#计费出账*小类#账单管理>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#计费出账*小类#账单管理>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#计费出账*小类#账单管理>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "65": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#计费出账*小类#费用类短信提醒>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#计费出账*小类#费用类短信提醒>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#计费出账*小类#费用类短信提醒>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "66": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#账单管理*小类#个人账单查询及使用>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#账单管理*小类#个人账单查询及使用>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#账单管理*小类#个人账单查询及使用>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "67": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#账单管理*小类#对账单质疑>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#账单管理*小类#对账单质疑>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#账单管理*小类#对账单质疑>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "68": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#账单管理*小类#集团账单查询及使用>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#账单管理*小类#集团账单查询及使用>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#账单管理*小类#集团账单查询及使用>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "69": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#账管支付*小类#充值业务>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#账管支付*小类#充值业务>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#账管支付*小类#充值业务>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "70": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#账管支付*小类#发票管理>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#账管支付*小类#发票管理>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#账管支付*小类#发票管理>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "71": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#账管支付*小类#托收>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#账管支付*小类#托收>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#账管支付*小类#托收>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "72": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#账管支付*小类#欠费管理>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#账管支付*小类#欠费管理>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#账管支付*小类#欠费管理>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "73": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#账管支付*小类#积分管理>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#账管支付*小类#积分管理>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#账管支付*小类#积分管理>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "74": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#账管支付*小类#账务变更>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#账管支付*小类#账务变更>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#账管支付*小类#账务变更>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "75": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#账管支付*小类#预缴>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#账管支付*小类#预缴>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#账管支付*小类#预缴>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "76": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#费用类短信提醒*小类#催缴及话费短信提醒>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#费用类短信提醒*小类#催缴及话费短信提醒>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#费用类短信提醒*小类#催缴及话费短信提醒>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            }
        }
    },
}

# ATTRIBUTE_LABELS = { # all attribute labels for a specific task
#     'yelpCategory': [
#         'Review Stars: 1.0', 
#         'Review Stars: 2.0', 
#         'Review Stars: 3.0', 
#         'Review Stars: 4.0', 
#         'Review Stars: 5.0'], # len(unique_label2)=5
#     'yelpRating': [
#         'Business Category: Arts & Entertainment', 
#         'Business Category: Bars', 
#         'Business Category: Beauty & Spas', 
#         'Business Category: Event Planning & Services', 
#         'Business Category: Grocery', 
#         'Business Category: Health & Medical', 
#         'Business Category: Home & Garden', 
#         'Business Category: Hotels & Travel', 
#         'Business Category: Restaurants', 
#         'Business Category: Shopping'], # len(unique_label1)=10
#     'openreviewCategory': [
#         "Recommendation: 1: strong reject",
#         "Recommendation: 3: reject, not good enough",
#         "Recommendation: 5: marginally below the acceptance threshold",
#         "Recommendation: 6: marginally above the acceptance threshold",
#         "Recommendation: 8: accept, good paper"], # len(unique_label2)=5
#     'openreviewRating': [
#         "Area: Applications (eg, speech processing, computer vision, NLP)",
#         "Area: Deep Learning and representational learning",
#         "Area: General Machine Learning",
#         "Area: Generative models",
#         "Area: Machine Learning for Sciences (eg biology, physics, health sciences, social sciences, climate/sustainability )",
#         "Area: Neuroscience and Cognitive Science (e.g., neural coding, brain-computer interfaces)",
#         "Area: Optimization (eg, convex and non-convex optimization)",
#         "Area: Probabilistic Methods (eg, variational inference, causal inference, Gaussian processes)",
#         "Area: Reinforcement Learning (eg, decision and control, planning, hierarchical RL, robotics)",
#         "Area: Social Aspects of Machine Learning (eg, AI safety, fairness, privacy, interpretability, human-AI interaction, ethics)",
#         "Area: Theory (eg, control theory, learning theory, algorithmic game theory)",
#         "Area: Unsupervised and Self-supervised learning"], # len(unique_label1)=12
# }

ATTRIBUTE_LABELS = { # all attribute labels for a specific task
    'yelpCategory': [
        '1.0', 
        '2.0', 
        '3.0', 
        '4.0', 
        '5.0'], # len(unique_label2)=5
    'yelpRating': [
        'Arts & Entertainment', 
        'Bars', 
        'Beauty & Spas', 
        'Event Planning & Services', 
        'Grocery', 
        'Health & Medical', 
        'Home & Garden', 
        'Hotels & Travel', 
        'Restaurants', 
        'Shopping'], # len(unique_label1)=10
    'openreviewCategory': [
        "1: strong reject",
        "3: reject, not good enough",
        "5: marginally below the acceptance threshold",
        "6: marginally above the acceptance threshold",
        "8: accept, good paper"], # len(unique_label2)=5
    'openreviewRating': [
        "Applications (eg, speech processing, computer vision, NLP)",
        "Deep Learning and representational learning",
        "General Machine Learning",
        "Generative models",
        "Machine Learning for Sciences (eg biology, physics, health sciences, social sciences, climate/sustainability )",
        "Neuroscience and Cognitive Science (e.g., neural coding, brain-computer interfaces)",
        "Optimization (eg, convex and non-convex optimization)",
        "Probabilistic Methods (eg, variational inference, causal inference, Gaussian processes)",
        "Reinforcement Learning (eg, decision and control, planning, hierarchical RL, robotics)",
        "Social Aspects of Machine Learning (eg, AI safety, fairness, privacy, interpretability, human-AI interaction, ethics)",
        "Theory (eg, control theory, learning theory, algorithmic game theory)",
        "Unsupervised and Self-supervised learning"], # len(unique_label1)=12
}


FEW_SHOT_PROMPT_WITH_GOOD_AND_BAD = {
    'squad': {
        "task_name": "squad",
        "stage": "x2",
        "instruction": "{}\nBased on the above examples of bad and good Context-Question pairs, analyze the differences between the bad and good samples. Generate a new question that is diverse in expression compared to the given good samples and further refined than the good samples and further from the bad samples. The context is: \"<C>\"\n\"<Y>\" is the answer of the following question: \""
    },
}

FEW_SHOT_PROMPT_PER_CLASS_WITH_GOOD_AND_BAD = {
    'imdb': {
        "task_name": "imdb",
        "stage": "x2",
        "labels": {
            "0": {
                "instruction": "{}\nBased on the above examples of bad and good movie reviews in positive sentiment, analyze the differences between the bad and good reviews. Generate a new positive movie review that is diverse in expression compared to the given good reviews. Ensure that the new review is further refined than the good reviews while maintaining the positive sentiment and clarity, making the good reviews appear to lie midway between the new review and the bad reviews. The new positive movie review is: \"",
                "counter_labels": ["1"]
            },
            "1": {
                "instruction": "{}\nBased on the above examples of bad and good movie reviews in negative sentiment, analyze the differences between the bad and good reviews. Generate a new negative movie review that is diverse in expression compared to the given good reviews. Ensure that the new review is further refined than the good reviews while maintaining the negative sentiment and clarity, making the good reviews appear to lie midway between the new review and the bad reviews. The new negatvie movie review is: \"",
                "counter_labels": ["0"]
            }
        }
    },
    'yelpCategory': {
        "task_name": "yelpCategory",
        "stage": "x2",
        "labels": {
            "0": {
                "instruction": "{}\nBased on the above examples of bad and good business reviews belonging to the category of 'Arts & Entertainment', analyze the differences between the bad and good reviews. Generate a new review for a business item also in the field of 'Arts & Entertainment' with rating {} star(s) but diverse in the expression compared to the given good reviews. Ensure that the new review is further refined than the good reviews while maintaining clarity, making the good reviews appear to lie midway between the new review and the bad reviews. The new business review in the field of 'Arts & Entertainment' is: \"",
                "counter_labels": ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
            },
            "1": {
                "instruction": "{}\nBased on the above examples of bad and good business reviews belonging to the category of 'Bars', analyze the differences between the bad and good reviews. Generate a new review for a business item also in the field of 'Bars' with rating {} star(s) but diverse in the expression compared to the given good reviews. Ensure that the new review is further refined than the good reviews while maintaining clarity, making the good reviews appear to lie midway between the new review and the bad reviews. The new business review in the field of 'Bars' is: \"",
                "counter_labels": ["0", "2", "3", "4", "5", "6", "7", "8", "9"]
            },
            "2": {
                "instruction": "{}\nBased on the above examples of bad and good business reviews belonging to the category of 'Beauty & Spas', analyze the differences between the bad and good reviews. Generate a new review for a business item also in the field of 'Beauty & Spas' with rating {} star(s) but diverse in the expression compared to the given good reviews. Ensure that the new review is further refined than the good reviews while maintaining clarity, making the good reviews appear to lie midway between the new review and the bad reviews. The new business review in the field of 'Beauty & Spas' is: \"",
                "counter_labels": ["0", "1", "3", "4", "5", "6", "7", "8", "9"]
            },
            "3": {
                "instruction": "{}\nBased on the above examples of bad and good business reviews belonging to the category of 'Event Planning & Services', analyze the differences between the bad and good reviews. Generate a new review for a business item also in the field of 'Event Planning & Services' with rating {} star(s) but diverse in the expression compared to the given good reviews. Ensure that the new review is further refined than the good reviews while maintaining clarity, making the good reviews appear to lie midway between the new review and the bad reviews. The new business review in the field of 'Event Planning & Services' is: \"",
                "counter_labels": ["0", "1", "2", "4", "5", "6", "7", "8", "9"]
            },
            "4": {
                "instruction": "{}\nBased on the above examples of bad and good business reviews belonging to the category of 'Grocery', analyze the differences between the bad and good reviews. Generate a new review for a business item also in the field of 'Grocery' with rating {} star(s) but diverse in the expression compared to the given good reviews. Ensure that the new review is further refined than the good reviews while maintaining clarity, making the good reviews appear to lie midway between the new review and the bad reviews. The new business review in the field of 'Grocery' is: \"",
                "counter_labels": ["0", "1", "2", "3", "5", "6", "7", "8", "9"]
            },
            "5": {
                "instruction": "{}\nBased on the above examples of bad and good business reviews belonging to the category of 'Health & Medical', analyze the differences between the bad and good reviews. Generate a new review for a business item also in the field of 'Health & Medical' with rating {} star(s) but diverse in the expression compared to the given good reviews. Ensure that the new review is further refined than the good reviews while maintaining clarity, making the good reviews appear to lie midway between the new review and the bad reviews. The new business review in the field of 'Health & Medical' is: \"",
                "counter_labels": ["0", "1", "2", "3", "4", "6", "7", "8", "9"]
            },
            "6": {
                "instruction": "{}\nBased on the above examples of bad and good business reviews belonging to the category of 'Home & Garden', analyze the differences between the bad and good reviews. Generate a new review for a business item also in the field of 'Home & Garden' with rating {} star(s) but diverse in the expression compared to the given good reviews. Ensure that the new review is further refined than the good reviews while maintaining clarity, making the good reviews appear to lie midway between the new review and the bad reviews. The new business review in the field of 'Home & Garden' is: \"",
                "counter_labels": ["0", "1", "2", "3", "4", "5", "7", "8", "9"]
            },
            "7": {
                "instruction": "{}\nBased on the above examples of bad and good business reviews belonging to the category of 'Hotels & Travel', analyze the differences between the bad and good reviews. Generate a new review for a business item also in the field of 'Hotels & Travel' with rating {} star(s) but diverse in the expression compared to the given good reviews. Ensure that the new review is further refined than the good reviews while maintaining clarity, making the good reviews appear to lie midway between the new review and the bad reviews. The new business review in the field of 'Hotels & Travel' is: \"",
                "counter_labels": ["0", "1", "2", "3", "4", "5", "6", "8", "9"]
            },
            "8": {
                "instruction": "{}\nBased on the above examples of bad and good business reviews belonging to the category of 'Restaurants', analyze the differences between the bad and good reviews. Generate a new review for a business item also in the field of 'Restaurants' with rating {} star(s) but diverse in the expression compared to the given good reviews. Ensure that the new review is further refined than the good reviews while maintaining clarity, making the good reviews appear to lie midway between the new review and the bad reviews. The new business review in the field of 'Restaurants' is: \"",
                "counter_labels": ["0", "1", "2", "3", "4", "5", "6", "7", "9"]
            },
            "9": {
                "instruction": "{}\nBased on the above examples of bad and good business reviews belonging to the category of 'Shopping', analyze the differences between the bad and good reviews. Generate a new review for a business item also in the field of 'Shopping' with rating {} star(s) but diverse in the expression compared to the given good reviews. Ensure that the new review is further refined than the good reviews while maintaining clarity, making the good reviews appear to lie midway between the new review and the bad reviews. The new business review in the field of 'Shopping' is: \"",
                "counter_labels": ["0", "1", "2", "3", "4", "5", "6", "7", "8"]
            },
        }
    },
    'yelpRating': {
        "task_name": "yelpRating",
        "stage": "x2",
        "labels": {
            "0": {
                "instruction": "{}\nBased on the above examples of bad and good business reviews with rating 1.0 star(s), analyze the differences between the bad and good reviews. Generate a new review for a business item in the field of {} also with rating 1.0 star(s) but diverse in the expression compared to the above given good reviews. Ensure that the new review is further refined than the good reviews while maintaining clarity, making the good reviews appear to lie midway between the new review and the bad reviews. The new business review with rating 1.0 star(s) is: \"",
                "counter_labels": ["1", "2", "3", "4"]
            },
            "1": {
                "instruction": "{}\nBased on the above examples of bad and good business reviews with rating 2.0 star(s), analyze the differences between the bad and good reviews. Generate a new review for a business item in the field of {} also with rating 2.0 star(s) but diverse in the expression compared to the above given good reviews. Ensure that the new review is further refined than the good reviews while maintaining clarity, making the good reviews appear to lie midway between the new review and the bad reviews. The new business review with rating 2.0 star(s) is: \"",
                "counter_labels": ["0", "2", "3", "4"]
            },
            "2": {
                "instruction": "{}\nBased on the above examples of bad and good business reviews with rating 3.0 star(s), analyze the differences between the bad and good reviews. Generate a new review for a business item in the field of {} also with rating 3.0 star(s) but diverse in the expression compared to the above given good reviews. Ensure that the new review is further refined than the good reviews while maintaining clarity, making the good reviews appear to lie midway between the new review and the bad reviews. The new business review with rating 3.0 star(s) is: \"",
                "counter_labels": ["0", "1", "3", "4"]
            },
            "3": {
                "instruction": "{}\nBased on the above examples of bad and good business reviews with rating 4.0 star(s), analyze the differences between the bad and good reviews. Generate a new review for a business item in the field of {} also with rating 4.0 star(s) but diverse in the expression compared to the above given good reviews. Ensure that the new review is further refined than the good reviews while maintaining clarity, making the good reviews appear to lie midway between the new review and the bad reviews. The new business review with rating 4.0 star(s) is: \"",
                "counter_labels": ["0", "1", "2", "4"]
            },
            "4": {
                "instruction": "{}\nBased on the above examples of bad and good business reviews with rating 5.0 star(s), analyze the differences between the bad and good reviews. Generate a new review for a business item in the field of {} also with rating 5.0 star(s) but diverse in the expression compared to the above given good reviews. Ensure that the new review is further refined than the good reviews while maintaining clarity, making the good reviews appear to lie midway between the new review and the bad reviews. The new business review with rating 5.0 star(s) is: \"",
                "counter_labels": ["0", "1", "2", "3"]
            }
        }
    },
    'openreviewCategory': {
        "task_name": "openreviewCategory",
        "stage": "x2",
        "labels": {
            "0": {
                "instruction": "{}\nBased on the above examples of bad and good paper reviews of paper in the area 'Applications (eg, speech processing, computer vision, NLP)', analyze the differences between the bad and good reviews. Generate a new review for a paper also in the area of 'Applications (eg, speech processing, computer vision, NLP)' with final recommendation: '{}' but diverse in the expression compared to the given good reviews. Ensure that the new review is further refined than the good reviews while maintaining clarity, making the good reviews appear to lie midway between the new review and the bad reviews. The new paper review in the area 'Applications (eg, speech processing, computer vision, NLP)' is: \"",
                # "counter_labels": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]
            },
            "1": {
                "instruction": "{}\nBased on the above examples of bad and good paper reviews of paper in the area 'Deep Learning and representational learning', analyze the differences between the bad and good reviews. Generate a new review for a paper also in the area of 'Deep Learning and representational learning' with final recommendation: '{}' but diverse in the expression compared to the given good reviews. Ensure that the new review is further refined than the good reviews while maintaining clarity, making the good reviews appear to lie midway between the new review and the bad reviews. The new paper review in the area 'Deep Learning and representational learning' is: \"",
                # "counter_labels": ["0", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]
            },
            "2": {
                "instruction": "{}\nBased on the above examples of bad and good paper reviews of paper in the area 'General Machine Learning', analyze the differences between the bad and good reviews. Generate a new review for a paper also in the area of 'General Machine Learning' with final recommendation: '{}' but diverse in the expression compared to the given good reviews. Ensure that the new review is further refined than the good reviews while maintaining clarity, making the good reviews appear to lie midway between the new review and the bad reviews. The new paper review in the area 'General Machine Learning' is: \"",
                # "counter_labels": ["0", "1", "3", "4", "5", "6", "7", "8", "9", "10", "11"]
            },
            "3": {
                "instruction": "{}\nBased on the above examples of bad and good paper reviews of paper in the area 'Generative models', analyze the differences between the bad and good reviews. Generate a new review for a paper also in the area of 'Generative models' with final recommendation: '{}' but diverse in the expression compared to the given good reviews. Ensure that the new review is further refined than the good reviews while maintaining clarity, making the good reviews appear to lie midway between the new review and the bad reviews. The new paper review in the area 'Generative models' is: \"",
                # "counter_labels": ["0", "1", "2", "4", "5", "6", "7", "8", "9", "10", "11"]
            },
            "4": {
                "instruction": "{}\nBased on the above examples of bad and good paper reviews of paper in the area 'Machine Learning for Sciences (eg biology, physics, health sciences, social sciences, climate/sustainability )', analyze the differences between the bad and good reviews. Generate a new review for a paper also in the area of 'Machine Learning for Sciences (eg biology, physics, health sciences, social sciences, climate/sustainability )' with final recommendation: '{}' but diverse in the expression compared to the given good reviews. Ensure that the new review is further refined than the good reviews while maintaining clarity, making the good reviews appear to lie midway between the new review and the bad reviews. The new paper review in the area 'Machine Learning for Sciences (eg biology, physics, health sciences, social sciences, climate/sustainability )' is: \"",
                # "counter_labels": ["0", "1", "2", "3", "5", "6", "7", "8", "9", "10", "11"]
            },
            "5": {
                "instruction": "{}\nBased on the above examples of bad and good paper reviews of paper in the area 'Neuroscience and Cognitive Science (e.g., neural coding, brain-computer interfaces)', analyze the differences between the bad and good reviews. Generate a new review for a paper also in the area of 'Neuroscience and Cognitive Science (e.g., neural coding, brain-computer interfaces)' with final recommendation: '{}' but diverse in the expression compared to the given good reviews. Ensure that the new review is further refined than the good reviews while maintaining clarity, making the good reviews appear to lie midway between the new review and the bad reviews. The new paper review in the area 'Neuroscience and Cognitive Science (e.g., neural coding, brain-computer interfaces)' is: \"",
                # "counter_labels": ["0", "1", "2", "3", "4", "6", "7", "8", "9", "10", "11"]
            },
            "6": {
                "instruction": "{}\nBased on the above examples of bad and good paper reviews of paper in the area 'Optimization (eg, convex and non-convex optimization)', analyze the differences between the bad and good reviews. Generate a new review for a paper also in the area of 'Optimization (eg, convex and non-convex optimization)' with final recommendation: '{}' but diverse in the expression compared to the given good reviews. Ensure that the new review is further refined than the good reviews while maintaining clarity, making the good reviews appear to lie midway between the new review and the bad reviews. The new paper review in the area 'Optimization (eg, convex and non-convex optimization)' is: \"",
                # "counter_labels": ["0", "1", "2", "3", "4", "5", "7", "8", "9", "10", "11"]
            },
            "7": {
                "instruction": "{}\nBased on the above examples of bad and good paper reviews of paper in the area 'Probabilistic Methods (eg, variational inference, causal inference, Gaussian processes)', analyze the differences between the bad and good reviews. Generate a new review for a paper also in the area of 'Probabilistic Methods (eg, variational inference, causal inference, Gaussian processes)' with final recommendation: '{}' but diverse in the expression compared to the given good reviews. Ensure that the new review is further refined than the good reviews while maintaining clarity, making the good reviews appear to lie midway between the new review and the bad reviews. The new paper review in the area 'Probabilistic Methods (eg, variational inference, causal inference, Gaussian processes)' is: \"",
                # "counter_labels": ["0", "1", "2", "3", "4", "5", "6", "8", "9", "10", "11"]
            },
            "8": {
                "instruction": "{}\nBased on the above examples of bad and good paper reviews of paper in the area 'Reinforcement Learning (eg, decision and control, planning, hierarchical RL, robotics)', analyze the differences between the bad and good reviews. Generate a new review for a paper also in the area of 'Reinforcement Learning (eg, decision and control, planning, hierarchical RL, robotics)' with final recommendation: '{}' but diverse in the expression compared to the given good reviews. Ensure that the new review is further refined than the good reviews while maintaining clarity, making the good reviews appear to lie midway between the new review and the bad reviews. The new paper review in the area 'Reinforcement Learning (eg, decision and control, planning, hierarchical RL, robotics)' is: \"",
                # "counter_labels": ["0", "1", "2", "3", "4", "5", "6", "7", "9", "10", "11"]
            },
            "9": {
                "instruction": "{}\nBased on the above examples of bad and good paper reviews of paper in the area 'Social Aspects of Machine Learning (eg, AI safety, fairness, privacy, interpretability, human-AI interaction, ethics)', analyze the differences between the bad and good reviews. Generate a new review for a paper also in the area of 'Social Aspects of Machine Learning (eg, AI safety, fairness, privacy, interpretability, human-AI interaction, ethics)' with final recommendation: '{}' but diverse in the expression compared to the given good reviews. Ensure that the new review is further refined than the good reviews while maintaining clarity, making the good reviews appear to lie midway between the new review and the bad reviews. The new paper review in the area 'Social Aspects of Machine Learning (eg, AI safety, fairness, privacy, interpretability, human-AI interaction, ethics)' is: \"",
                # "counter_labels": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "10", "11"]
            },
            "10": {
                "instruction": "{}\nBased on the above examples of bad and good paper reviews of paper in the area 'Theory (eg, control theory, learning theory, algorithmic game theory)', analyze the differences between the bad and good reviews. Generate a new review for a paper also in the area of 'Theory (eg, control theory, learning theory, algorithmic game theory)' with final recommendation: '{}' but diverse in the expression compared to the given good reviews. Ensure that the new review is further refined than the good reviews while maintaining clarity, making the good reviews appear to lie midway between the new review and the bad reviews. The new paper review in the area 'Theory (eg, control theory, learning theory, algorithmic game theory)' is: \"",
                # "counter_labels": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "11"]
            },
            "11": {
                "instruction": "{}\nBased on the above examples of bad and good paper reviews of paper in the area 'Unsupervised and Self-supervised learning', analyze the differences between the bad and good reviews. Generate a new review for a paper also in the area of 'Unsupervised and Self-supervised learning' with final recommendation: '{}' but diverse in the expression compared to the given good reviews. Ensure that the new review is further refined than the good reviews while maintaining clarity, making the good reviews appear to lie midway between the new review and the bad reviews. The new paper review in the area 'Unsupervised and Self-supervised learning' is: \"",
                # "counter_labels": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
            },
        }
    },
    'openreviewRating': {
        "task_name": "openreviewRating",
        "stage": "x2",
        "labels": {
            "0": {
                "instruction": "{}\nBased on the above examples of bad and good paper reviews of final recommendation: '1: strong reject', analyze the differences between the bad and good reviews. Generate a new review for a paper in the field of '{}' also with final recommendation: '1: strong reject' but diverse in the expression compared to the above given good reviews. Ensure that the new review is further refined than the good reviews while maintaining clarity, making the good reviews appear to lie midway between the new review and the bad reviews. The new paper review of final recommendation: '1: strong reject' is: \"",
                # "counter_labels": ["1", "2", "3", "4"]
            },
            "1": {
                "instruction": "{}\nBased on the above examples of bad and good paper reviews of final recommendation: '3: reject, not good enough', analyze the differences between the bad and good reviews. Generate a new review for a paper in the field of '{}' also with final recommendation: '3: reject, not good enough' but diverse in the expression compared to the above given good reviews. Ensure that the new review is further refined than the good reviews while maintaining clarity, making the good reviews appear to lie midway between the new review and the bad reviews. The new paper review of final recommendation: '3: reject, not good enough' is: \"",
                # "counter_labels": ["0", "2", "3", "4"]
            },
            "2": {
                "instruction": "{}\nBased on the above examples of bad and good paper reviews of final recommendation: '5: marginally below the acceptance threshold', analyze the differences between the bad and good reviews. Generate a new review for a paper in the field of '{}' also with final recommendation: '5: marginally below the acceptance threshold' but diverse in the expression compared to the above given good reviews. Ensure that the new review is further refined than the good reviews while maintaining clarity, making the good reviews appear to lie midway between the new review and the bad reviews. The new paper review final recommendation: '5: marginally below the acceptance threshold' is: \"",
                # "counter_labels": ["0", "1", "3", "4"]
            },
            "3": {
                "instruction": "{}\nBased on the above examples of bad and good paper reviews of final recommendation: '6: marginally above the acceptance threshold', analyze the differences between the bad and good reviews. Generate a new review for a paper in the field of '{}' also with final recommendation: '6: marginally above the acceptance threshold' but diverse in the expression compared to the above given good reviews. Ensure that the new review is further refined than the good reviews while maintaining clarity, making the good reviews appear to lie midway between the new review and the bad reviews. The new paper review final recommendation: '6: marginally above the acceptance threshold' is: \"",
                # "counter_labels": ["0", "1", "2", "4"]
            },
            "4": {
                "instruction": "{}\nBased on the above examples of bad and good paper reviews of final recommendation: '8: accept, good paper', analyze the differences between the bad and good reviews. Generate a new review for a paper in the field of '{}' also with final recommendation: '8: accept, good paper' but diverse in the expression compared to the above given good reviews. Ensure that the new review is further refined than the good reviews while maintaining clarity, making the good reviews appear to lie midway between the new review and the bad reviews. The new paper review final recommendation: '8: accept, good paper' is: \"",
                # "counter_labels": ["0", "1", "2", "3"]
            },
        }
    },
    'banking': {
        "task_name": "banking",
        "stage": "x2",
        "labels": {
            "0": {
                "instruction": "{}\nBased on the above examples of bad and good online banking queries in the category of \"activate my card\", analyze the differences between the bad and good reviews. Generate a new online banking query also in the category of \"activate my card\" but diverse in the expression compared to the above given good queries. Ensure that the new query is further refined than the good queries while maintaining clarity, making the good queries appear to lie midway between the new query and the bad queries. The new online banking query also in the category of \"activate my card\" is: \""
            },
            "1": {
                "instruction": "{}\nBased on the above examples of bad and good online banking queries in the category of \"age limit\", analyze the differences between the bad and good reviews. Generate a new online banking query in the category of \"age limit\" but diverse in the expression compared to the above given good queries. Ensure that the new query is further refined than the good queries while maintaining clarity, making the good queries appear to lie midway between the new query and the bad queries. The new online banking query in the category of \"age limit\" is: \""
            },
            "2": {
                "instruction": "{}\nBased on the above examples of bad and good online banking queries in the category of \"apple pay or google pay\", analyze the differences between the bad and good reviews. Generate a new online banking query in the category of \"apple pay or google pay\" but diverse in the expression compared to the above given good queries. Ensure that the new query is further refined than the good queries while maintaining clarity, making the good queries appear to lie midway between the new query and the bad queries. The new online banking query in the category of \"apple pay or google pay\" is: \""
            },
            "3": {
                "instruction": "{}\nBased on the above examples of bad and good online banking queries in the category of \"atm support\", analyze the differences between the bad and good reviews. Generate a new online banking query in the category of \"atm support\" but diverse in the expression compared to the above given good queries. Ensure that the new query is further refined than the good queries while maintaining clarity, making the good queries appear to lie midway between the new query and the bad queries. The new online banking query in the category of \"atm support\" is: \""
            },
            "4": {
                "instruction": "{}\nBased on the above examples of bad and good online banking queries in the category of \"automatic top up\", analyze the differences between the bad and good reviews. Generate a new online banking query in the category of \"automatic top up\" but diverse in the expression compared to the above given good queries. Ensure that the new query is further refined than the good queries while maintaining clarity, making the good queries appear to lie midway between the new query and the bad queries. The new online banking query in the category of \"automatic top up\" is: \""
            },
            "5": {
                "instruction": "{}\nBased on the above examples of bad and good online banking queries in the category of \"balance not updated after bank transfer\", analyze the differences between the bad and good reviews. Generate a new online banking query in the category of \"balance not updated after bank transfer\" but diverse in the expression compared to the above given good queries. Ensure that the new query is further refined than the good queries while maintaining clarity, making the good queries appear to lie midway between the new query and the bad queries. The new online banking query in the category of \"balance not updated after bank transfer\" is: \""
            },
            "6": {
                "instruction": "{}\nBased on the above examples of bad and good online banking queries in the category of \"balance not updated after cheque or cash deposit\", analyze the differences between the bad and good reviews. Generate a new online banking query in the category of \"balance not updated after cheque or cash deposit\" but diverse in the expression compared to the above given good queries. Ensure that the new query is further refined than the good queries while maintaining clarity, making the good queries appear to lie midway between the new query and the bad queries. The new online banking query in the category of \"balance not updated after cheque or cash deposit\" is: \""
            },
            "7": {
                "instruction": "{}\nBased on the above examples of bad and good online banking queries in the category of \"beneficiary not allowed\", analyze the differences between the bad and good reviews. Generate a new online banking query in the category of \"beneficiary not allowed\" but diverse in the expression compared to the above given good queries. Ensure that the new query is further refined than the good queries while maintaining clarity, making the good queries appear to lie midway between the new query and the bad queries. The new online banking query in the category of \"beneficiary not allowed\" is: \""
            },
            "8": {
                "instruction": "{}\nBased on the above examples of bad and good online banking queries in the category of \"cancel transfer\", analyze the differences between the bad and good reviews. Generate a new online banking query in the category of \"cancel transfer\" but diverse in the expression compared to the above given good queries. Ensure that the new query is further refined than the good queries while maintaining clarity, making the good queries appear to lie midway between the new query and the bad queries. The new online banking query in the category of \"cancel transfer\" is: \""
            },
            "9": {
                "instruction": "{}\nBased on the above examples of bad and good online banking queries in the category of \"card about to expire\", analyze the differences between the bad and good reviews. Generate a new online banking query in the category of \"card about to expire\" but diverse in the expression compared to the above given good queries. Ensure that the new query is further refined than the good queries while maintaining clarity, making the good queries appear to lie midway between the new query and the bad queries. The new online banking query in the category of \"card about to expire\" is: \""
            }
        }
    },
}

FEW_SHOT_PROMPT_PER_CLASS = {
    'imdb': {
        "task_name": "imdb",
        "stage": "x2",
        "labels": {
            "0": {
                "instruction": "{}\nBased on the above movie reviews in positive sentiment, a new movie review also in positive sentiment but diverse in the expression compared to the above given samples is: \"",
                "counter_labels": ["1"]
            },
            "1": {
                "instruction": "{}\nBased on the above movie reviews in negative sentiment, a new movie review also in negative sentiment but diverse in the expression compared to the above given samples is: \"",
                "counter_labels": ["0"]
            }
        }
    },
    'yelpCategory': {
        "task_name": "yelpCategory",
        "stage": "x2",
        "labels": {
            "0": {
                "instruction": "{}\nBased on the above business reviews belonging to the category of 'Arts & Entertainment', a new review for a business item also in the field of 'Arts & Entertainment' with rating {} star(s) but diverse in the expression compared to the above given samples is: \"",
                "counter_labels": ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
            },
            "1": {
                "instruction": "{}\nBased on the above business reviews belonging to the category of 'Bars', a new review for a business item also in the field of 'Bars' with rating {} star(s) but diverse in the expression compared to the above given samples is: \"",
                "counter_labels": ["0", "2", "3", "4", "5", "6", "7", "8", "9"]
            },
            "2": {
                "instruction": "{}\nBased on the above business reviews belonging to the category of 'Beauty & Spas', a new review for a business item also in the field of 'Beauty & Spas' with rating {} star(s) but diverse in the expression compared to the above given samples is: \"",
                "counter_labels": ["0", "1", "3", "4", "5", "6", "7", "8", "9"]
            },
            "3": {
                "instruction": "{}\nBased on the above business reviews belonging to the category of 'Event Planning & Services', a new review for a business item also in the field of Event 'Planning & Services' with rating {} star(s) but diverse in the expression compared to the above given samples is: \"",
                "counter_labels": ["0", "1", "2", "4", "5", "6", "7", "8", "9"]
            },
            "4": {
                "instruction": "{}\nBased on the above business reviews belonging to the category of 'Grocery', a new review for a business item also in the field of 'Grocery' with rating {} star(s) but diverse in the expression compared to the above given samples is: \"",
                "counter_labels": ["0", "1", "2", "3", "5", "6", "7", "8", "9"]
            },
            "5": {
                "instruction": "{}\nBased on the above business reviews belonging to the category of 'Health & Medical', a new review for a business item also in the field of 'Health & Medical' with rating {} star(s) but diverse in the expression compared to the above given samples is: \"",
                "counter_labels": ["0", "1", "2", "3", "4", "6", "7", "8", "9"]
            },
            "6": {
                "instruction": "{}\nBased on the above business reviews belonging to the category of 'Home & Garden', a new review for a business item also in the field of 'Home & Garden' with rating {} star(s) but diverse in the expression compared to the above given samples is: \"",
                "counter_labels": ["0", "1", "2", "3", "4", "5", "7", "8", "9"]
            },
            "7": {
                "instruction": "{}\nBased on the above business reviews belonging to the category of 'Hotels & Travel', a new review for a business item also in the field of 'Hotels & Travel' with rating {} star(s) but diverse in the expression compared to the above given samples is: \"",
                "counter_labels": ["0", "1", "2", "3", "4", "5", "6", "8", "9"]
            },
            "8": {
                "instruction": "{}\nBased on the above business reviews belonging to the category of 'Restaurants', a new review for a business item also in the field of 'Restaurants' with rating {} star(s) but diverse in the expression compared to the above given samples is: \"",
                "counter_labels": ["0", "1", "2", "3", "4", "5", "6", "7", "9"]
            },
            "9": {
                "instruction": "{}\nBased on the above business reviews belonging to the category of 'Shopping', a new review for a business item also in the field of 'Shopping' with rating {} star(s) but diverse in the expression compared to the above given samples is: \"",
                "counter_labels": ["0", "1", "2", "3", "4", "5", "6", "7", "8"]
            },
        }
    },
    'yelpRating': {
        "task_name": "yelpRating",
        "stage": "x2",
        "labels": {
            "0": {
                "instruction": "{}\nBased on the above business reviews with rating 1.0 star(s), a new review for a business item in the field of {} also with rating 1.0 star(s) but diverse in the expression compared to the above given samples is: \"",
                "counter_labels": ["1", "2", "3", "4"]
            },
            "1": {
                "instruction": "{}\nBased on the above business reviews with rating 2.0 star(s), a new review for a business item in the field of {} also with rating 2.0 star(s) but diverse in the expression compared to the above given samples is: \"",
                "counter_labels": ["0", "2", "3", "4"]
            },
            "2": {
                "instruction": "{}\nBased on the above business reviews with rating 3.0 star(s), a new review for a business item in the field of {} also with rating 3.0 star(s) but diverse in the expression compared to the above given samples is: \"",
                "counter_labels": ["0", "1", "3", "4"]
            },
            "3": {
                "instruction": "{}\nBased on the above business reviews with rating 4.0 star(s), a new review for a business item in the field of {} also with rating 4.0 star(s) but diverse in the expression compared to the above given samples is: \"",
                "counter_labels": ["0", "1", "2", "4"]
            },
            "4": {
                "instruction": "{}\nBased on the above business reviews with rating 5.0 star(s), a new review for a business item in the field of {} also with rating 5.0 star(s) but diverse in the expression compared to the above given samples is: \"",
                "counter_labels": ["0", "1", "2", "3"]
            }
        }
    },
    'yelp': {
        "task_name": "yelp",
        "stage": "x2",
        "labels": {
            "0": {
                "instruction": "{}\nThe new restaurant review in negative sentiment which is diverse in the expression compared to the above given samples is: \"",
                "counter_labels": ["1"]
            },
            "1": {
                "instruction": "{}\nThe new restaurant review in positive sentiment which is diverse in the expression compared to the above given samples is: \"",
                "counter_labels": ["0"]
            }
        }
    },
    'openreviewCategory': {
        "task_name": "openreviewCategory",
        "stage": "x2",
        "labels": {
            "0": {
                "instruction": "{}\nBased on the above paper reviews of paper in the area 'Applications (eg, speech processing, computer vision, NLP)', a new review for a paper also in the area of 'Applications (eg, speech processing, computer vision, NLP)' with final recommendation: '{}' but diverse in the expression compared to the above given samples is: \"",
                # "counter_labels": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]
            },
            "1": {
                "instruction": "{}\nBased on the above paper reviews of paper in the area 'Deep Learning and representational learning', a new review for a paper also in the area of 'Deep Learning and representational learning' with final recommendation: '{}' but diverse in the expression compared to the above given samples is: \"",
                # "counter_labels": ["0", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]
            },
            "2": {
                "instruction": "{}\nBased on the above paper reviews of paper in the area 'General Machine Learning', a new review for a paper also in the area of 'General Machine Learning' with final recommendation: '{}' but diverse in the expression compared to the above given samples is: \"",
                # "counter_labels": ["0", "1", "3", "4", "5", "6", "7", "8", "9", "10", "11"]
            },
            "3": {
                "instruction": "{}\nBased on the above paper reviews of paper in the area 'Generative models', a new review for a paper also in the area of 'Generative models' with final recommendation: '{}' but diverse in the expression compared to the above given samples is: \"",
                # "counter_labels": ["0", "1", "2", "4", "5", "6", "7", "8", "9", "10", "11"]
            },
            "4": {
                "instruction": "{}\nBased on the above paper reviews of paper in the area 'Machine Learning for Sciences (eg biology, physics, health sciences, social sciences, climate/sustainability )', a new review for a paper also in the area of 'Machine Learning for Sciences (eg biology, physics, health sciences, social sciences, climate/sustainability )' with final recommendation: '{}' but diverse in the expression compared to the above given samples is: \"",
                # "counter_labels": ["0", "1", "2", "3", "5", "6", "7", "8", "9", "10", "11"]
            },
            "5": {
                "instruction": "{}\nBased on the above paper reviews of paper in the area 'Neuroscience and Cognitive Science (e.g., neural coding, brain-computer interfaces)', a new review for a paper also in the area of 'Neuroscience and Cognitive Science (e.g., neural coding, brain-computer interfaces)' with final recommendation: '{}' but diverse in the expression compared to the above given samples is: \"",
                # "counter_labels": ["0", "1", "2", "3", "4", "6", "7", "8", "9", "10", "11"]
            },
            "6": {
                "instruction": "{}\nBased on the above paper reviews of paper in the area 'Optimization (eg, convex and non-convex optimization)', a new review for a paper also in the area of 'Optimization (eg, convex and non-convex optimization)' with final recommendation: '{}' but diverse in the expression compared to the above given samples is: \"",
                # "counter_labels": ["0", "1", "2", "3", "4", "5", "7", "8", "9", "10", "11"]
            },
            "7": {
                "instruction": "{}\nBased on the above paper reviews of paper in the area 'Probabilistic Methods (eg, variational inference, causal inference, Gaussian processes)', a new review for a paper also in the area of 'Probabilistic Methods (eg, variational inference, causal inference, Gaussian processes)' with final recommendation: '{}' but diverse in the expression compared to the above given samples is: \"",
                # "counter_labels": ["0", "1", "2", "3", "4", "5", "6", "8", "9", "10", "11"]
            },
            "8": {
                "instruction": "{}\nBased on the above paper reviews of paper in the area 'Reinforcement Learning (eg, decision and control, planning, hierarchical RL, robotics)', a new review for a paper also in the area of 'Reinforcement Learning (eg, decision and control, planning, hierarchical RL, robotics)' with final recommendation: '{}' but diverse in the expression compared to the above given samples is: \"",
                # "counter_labels": ["0", "1", "2", "3", "4", "5", "6", "7", "9", "10", "11"]
            },
            "9": {
                "instruction": "{}\nBased on the above paper reviews of paper in the area 'Social Aspects of Machine Learning (eg, AI safety, fairness, privacy, interpretability, human-AI interaction, ethics)', a new review for a paper also in the area of 'Social Aspects of Machine Learning (eg, AI safety, fairness, privacy, interpretability, human-AI interaction, ethics)' with final recommendation: '{}' but diverse in the expression compared to the above given samples is: \"",
                # "counter_labels": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "10", "11"]
            },
            "10": {
                "instruction": "{}\nBased on the above paper reviews of paper in the area 'Theory (eg, control theory, learning theory, algorithmic game theory)', a new review for a paper also in the area of 'Theory (eg, control theory, learning theory, algorithmic game theory)' with final recommendation: '{}' but diverse in the expression compared to the above given samples is: \"",
                # "counter_labels": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "11"]
            },
            "11": {
                "instruction": "{}\nBased on the above paper reviews of paper in the area 'Unsupervised and Self-supervised learning', a new review for a paper also in the area of 'Unsupervised and Self-supervised learning' with final recommendation: '{}' but diverse in the expression compared to the above given samples is: \"",
                # "counter_labels": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
            },
        }
    },
    'openreviewRating': {
        "task_name": "openreviewRating",
        "stage": "x2",
        "labels": {
            "0": {
                "instruction": "{}\nBased on the above paper reviews of final recommendation: '1: strong reject', a new review for a paper in the field of '{}' also with final recommendation: '1: strong reject' but diverse in the expression compared to the above given samples is: \"",
                # "counter_labels": ["1", "2", "3", "4"]
            },
            "1": {
                "instruction": "{}\nBased on the above paper reviews of final recommendation: '3: reject, not good enough', a new review for a paper in the field of '{}' also with final recommendation: '3: reject, not good enough' but diverse in the expression compared to the above given samples is: \"",
                # "counter_labels": ["0", "2", "3", "4"]
            },
            "2": {
                "instruction": "{}\nBased on the above paper reviews of final recommendation: '5: marginally below the acceptance threshold', a new review for a paper in the field of '{}' also with final recommendation: '5: marginally below the acceptance threshold' but diverse in the expression compared to the above given samples is: \"",
                # "counter_labels": ["0", "1", "3", "4"]
            },
            "3": {
                "instruction": "{}\nBased on the above paper reviews of final recommendation: '6: marginally above the acceptance threshold', a new review for a paper in the field of '{}' also with final recommendation: '6: marginally above the acceptance threshold' but diverse in the expression compared to the above given samples is: \"",
                # "counter_labels": ["0", "1", "2", "4"]
            },
            "4": {
                "instruction": "{}\nBased on the above paper reviews of final recommendation: '8: accept, good paper', a new review for a paper in the field of '{}' also with final recommendation: '8: accept, good paper' but diverse in the expression compared to the above given samples is: \"",
                # "counter_labels": ["0", "1", "2", "3"]
            },
        }
    },
    'banking': {
        "task_name": "banking",
        "stage": "x2",
        "labels": {
            "0": {
                "instruction": "{}\nBased on the above online banking queries in the category of \"activate my card\", a new online banking query also in the category of \"activate my card\" but diverse in the expression compared to the above given samples is: \""
            },
            "1": {
                "instruction": "{}\nBased on the above online banking queries in the category of \"age limit\", a new online banking query in the category of \"age limit\" but diverse in the expression compared to the above given samples is: \""
            },
            "2": {
                "instruction": "{}\nBased on the above online banking queries in the category of \"apple pay or google pay\", a new online banking query in the category of \"apple pay or google pay\" but diverse in the expression compared to the above given samples is: \""
            },
            "3": {
                "instruction": "{}\nBased on the above online banking queries in the category of \"atm support\", a new online banking query in the category of \"atm support\" but diverse in the expression compared to the above given samples is: \""
            },
            "4": {
                "instruction": "{}\nBased on the above online banking queries in the category of \"automatic top up\", a new online banking query in the category of \"automatic top up\" but diverse in the expression compared to the above given samples is: \""
            },
            "5": {
                "instruction": "{}\nBased on the above online banking queries in the category of \"balance not updated after bank transfer\", a new online banking query in the category of \"balance not updated after bank transfer\" but diverse in the expression compared to the above given samples is: \""
            },
            "6": {
                "instruction": "{}\nBased on the above online banking queries in the category of \"balance not updated after cheque or cash deposit\", a new online banking query in the category of \"balance not updated after cheque or cash deposit\" but diverse in the expression compared to the above given samples is: \""
            },
            "7": {
                "instruction": "{}\nBased on the above online banking queries in the category of \"beneficiary not allowed\", a new online banking query in the category of \"beneficiary not allowed\" but diverse in the expression compared to the above given samples is: \""
            },
            "8": {
                "instruction": "{}\nBased on the above online banking queries in the category of \"cancel transfer\", a new online banking query in the category of \"cancel transfer\" but diverse in the expression compared to the above given samples is: \""
            },
            "9": {
                "instruction": "{}\nBased on the above online banking queries in the category of \"card about to expire\", a new online banking query in the category of \"card about to expire\" but diverse in the expression compared to the above given samples is: \""
            }
        }
    },
    'mnli': {
        "task_name": "mnli",
        "stage": "x2",
        "labels": {
            "0": {
            "instruction": "{}\n\nThe new sentence pair which is diverse in the expression compared to the above given samples is: \"<C>\"\nIn other words, \"",
            "counter_labels": ["1", "2"]
            },
            "1": {
                "instruction": "{}\n\nThe new sentence pair which is diverse in the expression compared to the above given samples is: \"<C>\"\nFurthermore, \"",
                "counter_labels": ["2", "0"]
            },
            "2": {
                "instruction": "{}\n\nThe new sentence pair which is diverse in the expression compared to the above given samples is: There is a rumor that \"<C>\"\nHowever, the truth is: \"",
                "counter_labels": ["0", "1"]
            }
        }
    },
    'mnliMisM': {
        "task_name": "mnliMisM",
        "stage": "x2",
        "labels": {
            "0": {
                "instruction": "{}\n\nThe new sentence pair which is diverse in the expression compared to the above given samples is: \"<C>\"\nIn other words, \"",
                "counter_labels": ["1", "2"]
            },
            "1": {
                "instruction": "{}\n\nThe new sentence pair which is diverse in the expression compared to the above given samples is: \"<C>\"\nFurthermore, \"",
                "counter_labels": ["2", "0"]
            },
            "2": {
                "instruction": "{}\n\nThe new sentence pair which is diverse in the expression compared to the above given samples is: There is a rumor that \"<C>\"\nHowever, the truth is: \"",
                "counter_labels": ["0", "1"]
            }
        }
    },
    'qnli': {
        "task_name": "qnli",
        "stage": "x2",
        "labels": {
            "0": {
                "instruction": "{}\n\nThe new Information-Question pair which is diverse in the expression compared to the above given samples is: Information: \"<C>\"\nQuestion (answer in above information): \"",
                "counter_labels": ["1"]
            },
            "1": {
                "instruction": "{}\n\nThe new Information-Question pair which is diverse in the expression compared to the above given samples is: Information: \"<C>\"\nQuestion (answer not in above information):\"",
                "counter_labels": ["0"]
            }
        }
    },
    'agnews': {
        "task_name": "agnews",
        "stage": "x2",
        "labels": {
            "0": {
                "instruction": "{}\nThe new news article in the category of World which is diverse in the expression compared to the above given samples is: \"",
                "counter_labels": ["1","2","3"]
            },
            "1": {
                "instruction": "{}\nThe new news article in the category of Sports which is diverse in the expression compared to the above given samples is: \"",
                "counter_labels": ["0","2","3"]
            },
            "2": {
                "instruction": "{}\nThe new news article in the category of Business which is diverse in the expression compared to the above given samples is: \"",
                "counter_labels": ["0","1","3"]
            },
            "3": {
                "instruction": "{}\nThe new news article in the category of Technology which is diverse in the expression compared to the above given samples is: \"",
                "counter_labels": ["0","1","2"]
            }
        }
    },
    'markednews': {
        "task_name": "markednews",
        "stage": "x2",
        "labels": {
            "0": {
                "instruction": "{}\nA new news article in the category of World that does not include '$' and is diverse in the expression compared to the above given samples is: : \"",
                "counter_labels": ["1","2","3","4"]
            },
            "1": {
                "instruction": "{}\nA new news article in the category of Sports that does not include '$' and is diverse in the expression compared to the above given samples is: : \"",
                "counter_labels": ["0","2","3","4"]
            },
            "2": {
                "instruction": "{}\nA new news article in the category of Business that does not include '$' and is diverse in the expression compared to the above given samples is: : \"",
                "counter_labels": ["0","1","3","4"]
            },
            "3": {
                "instruction": "{}\nA new news article in the category of Technology that does not include '$' and is diverse in the expression compared to the above given samples is: : \"",
                "counter_labels": ["0","1","2","4"]
            },
            "4": {
                "instruction": "{}\nA new news article in the category of Money with '$' included and is diverse in the expression compared to the above given samples is: \"",
                "counter_labels": ["0","1","2","3"]
            }
        }
    },
    'squad': {
        "task_name": "squad",
        "stage": "x2",
        "instruction": "The context is: \"<C>\"\n\"<Y>\" is the answer of the following question: \""
    },
    'worksheet': {
        'task_name': 'worksheet',
        "stage": "x2",
        "labels": {
            "0": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#个人业务*小类#上网业务>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#个人业务*小类#上网业务>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#个人业务*小类#上网业务>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "1": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#个人业务*小类#业务类短信提醒>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#个人业务*小类#业务类短信提醒>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#个人业务*小类#业务类短信提醒>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "2": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#个人业务*小类#个人固话>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#个人业务*小类#个人固话>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#个人业务*小类#个人固话>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "3": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#个人业务*小类#停复机>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#个人业务*小类#停复机>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#个人业务*小类#停复机>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "4": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#个人业务*小类#副号业务>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#个人业务*小类#副号业务>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#个人业务*小类#副号业务>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "5": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#个人业务*小类#号码回收>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#个人业务*小类#号码回收>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#个人业务*小类#号码回收>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "6": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#个人业务*小类#基本策划>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#个人业务*小类#基本策划>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#个人业务*小类#基本策划>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "7": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#个人业务*小类#基础产品>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#个人业务*小类#基础产品>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#个人业务*小类#基础产品>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "8": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#个人业务*小类#增值策划>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#个人业务*小类#增值策划>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#个人业务*小类#增值策划>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "9": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#个人业务*小类#客户资料管理>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#个人业务*小类#客户资料管理>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#个人业务*小类#客户资料管理>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "10": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#个人业务*小类#开户>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#个人业务*小类#开户>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#个人业务*小类#开户>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "11": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#个人业务*小类#携号转网>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#个人业务*小类#携号转网>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#个人业务*小类#携号转网>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "12": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#个人业务*小类#省内携号>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#个人业务*小类#省内携号>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#个人业务*小类#省内携号>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "13": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#个人业务*小类#营业查询类>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#个人业务*小类#营业查询类>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#个人业务*小类#营业查询类>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "14": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#个人业务*小类#补换卡>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#个人业务*小类#补换卡>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#个人业务*小类#补换卡>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "15": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#个人业务*小类#过户>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#个人业务*小类#过户>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#个人业务*小类#过户>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "16": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#个人业务*小类#重入网>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#个人业务*小类#重入网>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#个人业务*小类#重入网>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "17": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#个人业务*小类#销户>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#个人业务*小类#销户>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#个人业务*小类#销户>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "18": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#信控管理*小类#特权报开>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#信控管理*小类#特权报开>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#信控管理*小类#特权报开>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "19": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#信控管理*小类#账务停复机>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#信控管理*小类#账务停复机>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#信控管理*小类#账务停复机>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "20": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#内部系统管理*小类#国产系统问题>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#内部系统管理*小类#国产系统问题>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#内部系统管理*小类#国产系统问题>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "21": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#内部系统管理*小类#客服系统>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#内部系统管理*小类#客服系统>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#内部系统管理*小类#客服系统>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "22": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#内部系统管理*小类#工号管理>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#内部系统管理*小类#工号管理>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#内部系统管理*小类#工号管理>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "23": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#内部系统管理*小类#批量管控>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#内部系统管理*小类#批量管控>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#内部系统管理*小类#批量管控>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "24": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#内部系统管理*小类#报表业务>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#内部系统管理*小类#报表业务>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#内部系统管理*小类#报表业务>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "25": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#内部系统管理*小类#无纸化系统>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#内部系统管理*小类#无纸化系统>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#内部系统管理*小类#无纸化系统>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "26": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#内部系统管理*小类#渠道系统>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#内部系统管理*小类#渠道系统>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#内部系统管理*小类#渠道系统>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "27": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#内部系统管理*小类#满意度>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#内部系统管理*小类#满意度>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#内部系统管理*小类#满意度>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "28": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#内部系统管理*小类#网格化智慧运营平台>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#内部系统管理*小类#网格化智慧运营平台>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#内部系统管理*小类#网格化智慧运营平台>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "29": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#内部系统管理*小类#网格化智慧运营平台>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#内部系统管理*小类#营业系统>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#内部系统管理*小类#营业系统>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "30": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#内部系统管理*小类#营销中心>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#内部系统管理*小类#营销中心>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#内部系统管理*小类#营销中心>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "31": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#内部系统管理*小类#资源管理>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#内部系统管理*小类#资源管理>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#内部系统管理*小类#资源管理>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "32": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#家庭业务*小类#固话业务>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#家庭业务*小类#固话业务>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#家庭业务*小类#固话业务>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "33": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#家庭业务*小类#家庭亲情网>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#家庭业务*小类#家庭亲情网>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#家庭业务*小类#家庭亲情网>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "34": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#家庭业务*小类#家庭套餐>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#家庭业务*小类#家庭套餐>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#家庭业务*小类#家庭套餐>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "35": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#家庭业务*小类#家庭统一支付>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#家庭业务*小类#家庭统一支付>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#家庭业务*小类#家庭统一支付>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "36": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#家庭业务*小类#宽带业务>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#家庭业务*小类#宽带业务>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#家庭业务*小类#宽带业务>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "37": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#家庭业务*小类#宽带电视>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#家庭业务*小类#宽带电视>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#家庭业务*小类#宽带电视>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "38": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#家庭业务*小类#移动看家>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#家庭业务*小类#移动看家>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#家庭业务*小类#移动看家>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "39": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#家庭业务*小类#群组业务>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#家庭业务*小类#群组业务>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#家庭业务*小类#群组业务>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "40": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#政企业务*小类#5G专网>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#政企业务*小类#5G专网>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#政企业务*小类#5G专网>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "41": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#政企业务*小类#专线类>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#政企业务*小类#专线类>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#政企业务*小类#专线类>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "42": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#政企业务*小类#其它>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#政企业务*小类#其它>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#政企业务*小类#其它>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "43": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#政企业务*小类#办公类>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#政企业务*小类#办公类>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#政企业务*小类#办公类>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "44": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#政企业务*小类#安防类>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#政企业务*小类#安防类>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#政企业务*小类#安防类>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "45": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#政企业务*小类#教育类>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#政企业务*小类#教育类>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#政企业务*小类#教育类>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "46": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#政企业务*小类#物联网业务>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#政企业务*小类#物联网业务>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#政企业务*小类#物联网业务>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "47": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#政企业务*小类#短彩类>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#政企业务*小类#短彩类>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#政企业务*小类#短彩类>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "48": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#政企业务*小类#融合套餐>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#政企业务*小类#融合套餐>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#政企业务*小类#融合套餐>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "49": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#政企业务*小类#视频融合类>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#政企业务*小类#视频融合类>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#政企业务*小类#视频融合类>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "50": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#政企业务*小类#语音类>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#政企业务*小类#语音类>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#政企业务*小类#语音类>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "51": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#政企业务*小类#资源类>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#政企业务*小类#资源类>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#政企业务*小类#资源类>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "52": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#政企业务*小类#集团客户管理>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#政企业务*小类#集团客户管理>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#政企业务*小类#集团客户管理>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "53": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#政企业务*小类#集团资金>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#政企业务*小类#集团资金>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#政企业务*小类#集团资金>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "54": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#电子渠道*小类#其他类别>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#电子渠道*小类#其他类别>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#电子渠道*小类#其他类别>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "55": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#电子渠道*小类#大视频>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#电子渠道*小类#大视频>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#电子渠道*小类#大视频>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "56": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#电子渠道*小类#天猫移动旗舰店>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#电子渠道*小类#天猫移动旗舰店>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#电子渠道*小类#天猫移动旗舰店>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "57": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#电子渠道*小类#微信营业厅>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#电子渠道*小类#微信营业厅>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#电子渠道*小类#微信营业厅>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "58": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#电子渠道*小类#手机营业厅>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#电子渠道*小类#手机营业厅>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#电子渠道*小类#手机营业厅>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "59": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#电子渠道*小类#短信营业厅>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#电子渠道*小类#短信营业厅>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#电子渠道*小类#短信营业厅>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "60": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#电子渠道*小类#网上营业厅>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#电子渠道*小类#网上营业厅>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#电子渠道*小类#网上营业厅>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "61": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#计费出账*小类#信控管理>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#计费出账*小类#信控管理>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#计费出账*小类#信控管理>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "62": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#计费出账*小类#计费查询类>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#计费出账*小类#计费查询类>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#计费出账*小类#计费查询类>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "63": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#计费出账*小类#详单管理>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#计费出账*小类#详单管理>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#计费出账*小类#详单管理>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "64": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#计费出账*小类#账单管理>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#计费出账*小类#账单管理>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#计费出账*小类#账单管理>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "65": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#计费出账*小类#费用类短信提醒>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#计费出账*小类#费用类短信提醒>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#计费出账*小类#费用类短信提醒>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "66": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#账单管理*小类#个人账单查询及使用>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#账单管理*小类#个人账单查询及使用>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#账单管理*小类#个人账单查询及使用>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "67": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#账单管理*小类#对账单质疑>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#账单管理*小类#对账单质疑>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#账单管理*小类#对账单质疑>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "68": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#账单管理*小类#集团账单查询及使用>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#账单管理*小类#集团账单查询及使用>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#账单管理*小类#集团账单查询及使用>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "69": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#账管支付*小类#充值业务>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#账管支付*小类#充值业务>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#账管支付*小类#充值业务>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "70": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#账管支付*小类#发票管理>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#账管支付*小类#发票管理>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#账管支付*小类#发票管理>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "71": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#账管支付*小类#托收>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#账管支付*小类#托收>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#账管支付*小类#托收>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "72": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#账管支付*小类#欠费管理>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#账管支付*小类#欠费管理>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#账管支付*小类#欠费管理>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "73": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#账管支付*小类#积分管理>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#账管支付*小类#积分管理>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#账管支付*小类#积分管理>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "74": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#账管支付*小类#账务变更>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#账管支付*小类#账务变更>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#账管支付*小类#账务变更>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "75": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#账管支付*小类#预缴>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#账管支付*小类#预缴>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#账管支付*小类#预缴>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            },
            "76": {
                "instruction": "{}一个新的、与上述所给例子表述不同的、属于 \"<类#费用类短信提醒*小类#催缴及话费短信提醒>\" 类别的工单是: \"",
                "gen_x_instruction": "<E>一个属于 \"<类#费用类短信提醒*小类#催缴及话费短信提醒>\" 类别的工单是: \"",
                "example_instruction": "一个属于 \"<类#费用类短信提醒*小类#催缴及话费短信提醒>\" 类别的工单是: \"<X>\"",
                "prompting_instruction": "一个属于 \"<C>\" 类别的工单是: \"<X>\"",
                "gen_c_instruction": "Restaurant name: \""
            }
        }
    },
}

SELF_WEIGHT_ADJUST_EPOCH = 4
# SELF_WEIGHT_ADJUST_EPOCH = 1

LABEL_MAPPING = {
    "imdb": {"positive": 0, "negative": 1},
    "sst2": {"positive": 0, "negative": 1},
    "yelp": {"negative": 0, "positive": 1},
    "mnli": {"entailment": 0, "neutral": 1, "contradiction":2},
    "mnliMisM": {"entailment": 0, "neutral": 1, "contradiction":2},
    "medical-cancer-doc": {"Thyroid_Cancer": 0, "Colon_Cancer": 1, "Lung_Cancer": 2},
    "qnli": {"entailment": 0, "not_entailment": 1},
    "mnliM": {"entailment": 0, "neutral": 1, "contradiction": 2},
    "mnliMisM": {"entailment": 0, "neutral": 1, "contradiction": 2},
    "squad": {},
    "agnews": {"World": 0, "Sports": 1, "Business": 2, "Science/Technology": 3},
    "worksheet": {
        "个人业务/增值策划": 0,
        "个人业务/营业查询类": 1,
        "家庭业务/家庭亲情网": 2,
        "电子渠道/手机营业厅": 3,
        "个人业务/客户资料管理": 4,
        "家庭业务/宽带业务": 5,
        "内部系统管理/无纸化系统": 6,
        "账管支付/积分管理": 7,
        "个人业务/基本策划": 8,
        "家庭业务/移动看家": 9,
        "个人业务/上网业务": 10,
        "政企业务/物联网业务": 11,
        "内部系统管理/资源管理": 12,
        "政企业务/语音类": 13,
        "家庭业务/宽带电视": 14,
        "个人业务/副号业务": 15,
        "计费出账/账单管理": 16,
        "个人业务/过户": 17,
        "政企业务/短彩类": 18,
        "政企业务/视频融合类": 19,
        "个人业务/业务类短信提醒": 20,
        "账管支付/充值业务": 21,
        "账管支付/预缴": 22,
        "内部系统管理/报表业务": 23,
        "家庭业务/固话业务": 24,
        "个人业务/携号转网": 25,
        "账管支付/发票管理": 26,
        "账管支付/欠费管理": 27,
        "政企业务/集团资金": 28,
        "个人业务/开户": 29,
        "计费出账/信控管理": 30,
        "家庭业务/群组业务": 31,
        "家庭业务/家庭统一支付": 32,
        "内部系统管理/批量管控": 33,
        "计费出账/费用类短信提醒": 34,
        "个人业务/补换卡": 35,
        "政企业务/5G专网": 36,
        "个人业务/停复机": 37,
        "账管支付/托收": 38,
        "政企业务/资源类": 39,
        "账管支付/账务变更": 40,
        "政企业务/其它": 41,
        "内部系统管理/渠道系统": 42,
        "个人业务/销户": 43,
        "计费出账/详单管理": 44,
        "内部系统管理/营业系统": 45,
        "内部系统管理/工号管理": 46,
        "政企业务/专线类": 47,
        "政企业务/教育类": 48,
        "政企业务/集团客户管理": 49,
        "政企业务/安防类": 50,
        "个人业务/重入网": 51,
        "内部系统管理/营销中心": 52,
        "个人业务/号码回收": 53,
        "政企业务/融合套餐": 54,
        "个人业务/基础产品": 55,
        "内部系统管理/客服系统": 56,
        "计费出账/计费查询类": 57,
        "内部系统管理/网格化智慧运营平台": 58,
        "内部系统管理/满意度": 59,
        "电子渠道/网上营业厅": 60,
        "家庭业务/家庭套餐": 61,
        "电子渠道/微信营业厅": 62,
        "电子渠道/天猫移动旗舰店": 63,
        "个人业务/省内携号": 64,
        "电子渠道/其他类别": 65,
        "个人业务/个人固话": 66,
        "电子渠道/大视频": 67,
        "电子渠道/短信营业厅": 68,
        "政企业务/办公类": 69,
        "内部系统管理/国产系统问题": 70,
        "账单管理/集团账单查询及使用": 71,
        "账单管理/对账单质疑": 72,
        "账单管理/个人账单查询及使用": 73,
        "信控管理/账务停复机": 74,
        "费用类短信提醒/催缴及话费短信提醒": 75,
        "信控管理/特权报开": 76,
        "增值策划": 0,
        "营业查询类": 1,
        "家庭亲情网": 2,
        "手机营业厅": 3,
        "客户资料管理": 4,
        "宽带业务": 5,
        "无纸化系统": 6,
        "积分管理": 7,
        "基本策划": 8,
        "移动看家": 9,
        "上网业务": 10,
        "物联网业务": 11,
        "资源管理": 12,
        "语音类": 13,
        "宽带电视": 14,
        "副号业务": 15,
        "账单管理": 16,
        "过户": 17,
        "短彩类": 18,
        "视频融合类": 19,
        "业务类短信提醒": 20,
        "充值业务": 21,
        "预缴": 22,
        "报表业务": 23,
        "固话业务": 24,
        "携号转网": 25,
        "发票管理": 26,
        "欠费管理": 27,
        "集团资金": 28,
        "开户": 29,
        "信控管理": 30,
        "群组业务": 31,
        "家庭统一支付": 32,
        "批量管控": 33,
        "费用类短信提醒": 34,
        "补换卡": 35,
        "5G专网": 36,
        "停复机": 37,
        "托收": 38,
        "资源类": 39,
        "账务变更": 40,
        "其它": 41,
        "渠道系统": 42,
        "销户": 43,
        "详单管理": 44,
        "营业系统": 45,
        "工号管理": 46,
        "专线类": 47,
        "教育类": 48,
        "集团客户管理": 49,
        "安防类": 50,
        "重入网": 51,
        "营销中心": 52,
        "号码回收": 53,
        "融合套餐": 54,
        "基础产品": 55,
        "客服系统": 56,
        "计费查询类": 57,
        "网格化智慧运营平台": 58,
        "满意度": 59,
        "网上营业厅": 60,
        "家庭套餐": 61,
        "微信营业厅": 62,
        "天猫移动旗舰店": 63,
        "省内携号": 64,
        "其他类别": 65,
        "个人固话": 66,
        "大视频": 67,
        "短信营业厅": 68,
        "办公类": 69,
        "国产系统问题": 70,
        "集团账单查询及使用": 71,
        "对账单质疑": 72,
        "个人账单查询及使用": 73,
        "账务停复机": 74,
        "催缴及话费短信提醒": 75,
        "特权报开": 76
    },
}