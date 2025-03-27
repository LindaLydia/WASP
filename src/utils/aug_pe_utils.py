import copy
import random

PROMPTS_TEMPLATES = {
    "init_yelp":  {
        "sys_prompt": "You are required to write an example of review based on the provided Business Category and Review Stars that fall within the range of 1.0-5.0.",
        "task_desc": "",
    },

    "init_openreview":  {
        "sys_prompt": "Given the area and final decision of a research paper, you are required to provide a **detailed and long** review consisting of the following content: 1. briefly summarizing the paper in 3-5 sentences; 2. listing the strengths and weaknesses of the paper in details; 3. briefly summarizing the review in 3-5 sentences.",
        "task_desc": "",
    },

    "init_pubmed":  {
        "sys_prompt": "Please act as a sentence generator for the medical domain. Generated sentences should mimic the style of PubMed journal articles, using a variety of sentence structures.",
        "task_desc":  "",
    },

    "variant_yelp":  {
        "sys_prompt": "You are a helpful, pattern-following assistant.",
        "task_desc": "",
    },
    "variant_pubmed":  {
        "sys_prompt": "Please act as a sentence generator for the medical domain. Generated sentences should mimic the style of PubMed journal articles, using a variety of sentence structures.",
        "task_desc":  "",
    },
    "variant_openreview":  {
        # Azure default system prompt
        "sys_prompt": "You are an AI assistant that helps people find information.",
        "task_desc": "",
    },

}

PUBMED_INIT_TEMPLATES = [
    "Please share an abstract for a medical research paper:",
    "Please provide an example of an abstract for a medical research paper:",
    "Please generate an abstract for a medical research paper:",
    "please share an abstract for a medical research paper as an example:",
    "please write a sample abstract for a medical research paper:",
    "please share an example of an abstract for a medical research paper:",
    "please write an abstract for a medical research paper as an example:",
    "please write an abstract for a medical research paper:",
]


ALL_STYLES = ["in a casual way", "in a creative style",  "in an informal way", "casually", "in a detailed way",
              "in a professional way", "with more details", "with a professional tone", "in a casual style", "in a professional style", "in a short way", "in a concise manner", "concisely", "briefly", "orally",
              "with imagination", "with a tone of earnestness",  "in a grammarly-incorrect way", "with grammatical errors",  "in a non-standard grammar fashion",
              "in an oral way", "in a spoken manner", "articulately",  "by word of mouth",  "in a storytelling tone",
              "in a formal manner", "with an informal tone", "in a laid-back manner"]
ALL_OPENREVIEW_STYLES = ["in a detailed way",  "in a professional way", "with more details",
                         "with a professional tone",  "in a professional style",   "in a concise manner"]
ALL_PUBMED_STYLES = ["in a professional way", "in a professional tone",  "in a professional style",   "in a concise manner",
                     "in a creative style", "using imagination", "in a storytelling tone",  "in a formal manner", "using a variety of sentence structures"
                     ]


PE_PROMPT = {
    'imdb': {
        "task_name": "imdb",
        "stage": "x2",
        "labels": {
            "0": {
                "instruction": "Based on positive sentiment, please rephrase the following sentences {}:\n{} \n",
                "counter_labels": ["1"]
            },
            "1": {
                "instruction": "Based on negative sentiment, please rephrase the following sentences {}:\n{} \n",
                "counter_labels": ["0"]
            }
        }
    },
    'yelpCategory': {
        "task_name": "yelpCategory",
        "stage": "x2",
        "labels": {
            "0": {
                "instruction": "Based on the category of 'Arts & Entertainment', please rephrase the following sentences {}:\n{} \n",
                "counter_labels": ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
            },
            "1": {
                "instruction": "Based on the category of 'Bars', please rephrase the following sentences {}:\n{} \n",
                "counter_labels": ["0", "2", "3", "4", "5", "6", "7", "8", "9"]
            },
            "2": {
                "instruction": "Based on the category of 'Beauty & Spas', please rephrase the following sentences {}:\n{} \n",
                "counter_labels": ["0", "1", "3", "4", "5", "6", "7", "8", "9"]
            },
            "3": {
                "instruction": "Based on the category of 'Event Planning & Services', please rephrase the following sentences {}:\n{} \n",
                "counter_labels": ["0", "1", "2", "4", "5", "6", "7", "8", "9"]
            },
            "4": {
                "instruction": "Based on the category of 'Grocery', please rephrase the following sentences {}:\n{} \n",
                "counter_labels": ["0", "1", "2", "3", "5", "6", "7", "8", "9"]
            },
            "5": {
                "instruction": "Based on the category of 'Health & Medical', please rephrase the following sentences {}:\n{} \n",
                "counter_labels": ["0", "1", "2", "3", "4", "6", "7", "8", "9"]
            },
            "6": {
                "instruction": "Based on the category of 'Home & Garden', please rephrase the following sentences {}:\n{} \n",
                "counter_labels": ["0", "1", "2", "3", "4", "5", "7", "8", "9"]
            },
            "7": {
                "instruction": "Based on the category of 'Hotels & Travel', please rephrase the following sentences {}:\n{} \n",
                "counter_labels": ["0", "1", "2", "3", "4", "5", "6", "8", "9"]
            },
            "8": {
                "instruction": "Based on the category of 'Restaurants', please rephrase the following sentences {}:\n{} \n",
                "counter_labels": ["0", "1", "2", "3", "4", "5", "6", "7", "9"]
            },
            "9": {
                "instruction": "Based on the category of 'Shopping', please rephrase the following sentences {}:\n{} \n",
                "counter_labels": ["0", "1", "2", "3", "4", "5", "6", "7", "8"]
            },
        }
    },
    'yelpRating': {
        "task_name": "yelpRating",
        "stage": "x2",
        "labels": {
            "0": {
                "instruction": "Based on rating 1.0 star(s), please rephrase the following sentences {}:\n{} \n",
                "counter_labels": ["1", "2", "3", "4"]
            },
            "1": {
                "instruction": "Based on rating 2.0 star(s), please rephrase the following sentences {}:\n{} \n",
                "counter_labels": ["0", "2", "3", "4"]
            },
            "2": {
                "instruction": "Based on rating 3.0 star(s), please rephrase the following sentences {}:\n{} \n",
                "counter_labels": ["0", "1", "3", "4"]
            },
            "3": {
                "instruction": "Based on rating 4.0 star(s), please rephrase the following sentences {}:\n{} \n",
                "counter_labels": ["0", "1", "2", "4"]
            },
            "4": {
                "instruction": "Based on rating 5.0 star(s), please rephrase the following sentences {}:\n{} \n",
                "counter_labels": ["0", "1", "2", "3"]
            }
        }
    },
    'openreviewCategory': {
        "task_name": "openreviewCategory",
        "stage": "x2",
        "labels": {
            "0": {
                "instruction": "Based on the area 'Applications (eg, speech processing, computer vision, NLP)', please rephrase the following sentences {} as a paper review:\n{} \n",
                # "counter_labels": ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]
            },
            "1": {
                "instruction": "Based on the area 'Deep Learning and representational learning', please rephrase the following sentences {} as a paper review:\n{} \n",
                # "counter_labels": ["0", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"]
            },
            "2": {
                "instruction": "Based on the area 'General Machine Learning', please rephrase the following sentences {} as a paper review:\n{} \n",
                # "counter_labels": ["0", "1", "3", "4", "5", "6", "7", "8", "9", "10", "11"]
            },
            "3": {
                "instruction": "Based on the area 'Generative models', please rephrase the following sentences {} as a paper review:\n{} \n",
                # "counter_labels": ["0", "1", "2", "4", "5", "6", "7", "8", "9", "10", "11"]
            },
            "4": {
                "instruction": "Based on the area 'Machine Learning for Sciences (eg biology, physics, health sciences, social sciences, climate/sustainability )', please rephrase the following sentences {} as a paper review:\n{} \n",
                # "counter_labels": ["0", "1", "2", "3", "5", "6", "7", "8", "9", "10", "11"]
            },
            "5": {
                "instruction": "Based on the area 'Neuroscience and Cognitive Science (e.g., neural coding, brain-computer interfaces)', please rephrase the following sentences {} as a paper review:\n{} \n",
                # "counter_labels": ["0", "1", "2", "3", "4", "6", "7", "8", "9", "10", "11"]
            },
            "6": {
                "instruction": "Based on the area 'Optimization (eg, convex and non-convex optimization)', please rephrase the following sentences {} as a paper review:\n{} \n",
                # "counter_labels": ["0", "1", "2", "3", "4", "5", "7", "8", "9", "10", "11"]
            },
            "7": {
                "instruction": "Based on the area 'Probabilistic Methods (eg, variational inference, causal inference, Gaussian processes)', please rephrase the following sentences {} as a paper review:\n{} \n",
                # "counter_labels": ["0", "1", "2", "3", "4", "5", "6", "8", "9", "10", "11"]
            },
            "8": {
                "instruction": "Based on the area 'Reinforcement Learning (eg, decision and control, planning, hierarchical RL, robotics)', please rephrase the following sentences {} as a paper review:\n{} \n",
                # "counter_labels": ["0", "1", "2", "3", "4", "5", "6", "7", "9", "10", "11"]
            },
            "9": {
                "instruction": "Based on the area 'Social Aspects of Machine Learning (eg, AI safety, fairness, privacy, interpretability, human-AI interaction, ethics)', please rephrase the following sentences {} as a paper review:\n{} \n",
                # "counter_labels": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "10", "11"]
            },
            "10": {
                "instruction": "Based on the area 'Theory (eg, control theory, learning theory, algorithmic game theory)', please rephrase the following sentences {} as a paper review:\n{} \n",
                # "counter_labels": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "11"]
            },
            "11": {
                "instruction": "Based on the area 'Unsupervised and Self-supervised learning', please rephrase the following sentences {} as a paper review:\n{} \n",
                # "counter_labels": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
            },
        }
    },
    'openreviewRating': {
        "task_name": "openreviewRating",
        "stage": "x2",
        "labels": {
            "0": {
                "instruction": "Based on final recommendation: '1: strong reject', please rephrase the following sentences {} as a paper review:\n{} \n",
                # "counter_labels": ["1", "2", "3", "4"]
            },
            "1": {
                "instruction": "Based on final recommendation: '3: reject, not good enough', please rephrase the following sentences {} as a paper review:\n{} \n",
                # "counter_labels": ["0", "2", "3", "4"]
            },
            "2": {
                "instruction": "Based on final recommendation: '5: marginally below the acceptance threshold', please rephrase the following sentences {} as a paper review:\n{} \n",
                # "counter_labels": ["0", "1", "3", "4"]
            },
            "3": {
                "instruction": "Based on final recommendation: '6: marginally above the acceptance threshold', please rephrase the following sentences {} as a paper review:\n{} \n",
                # "counter_labels": ["0", "1", "2", "4"]
            },
            "4": {
                "instruction": "Based on final recommendation: '8: accept, good paper', please rephrase the following sentences {} as a paper review:\n{} \n",
                # "counter_labels": ["0", "1", "2", "3"]
            },
        }
    },
    'banking': {
        "task_name": "banking",
        "stage": "x2",
        "labels": {
            "0": {
                "instruction": "Based on the category of \"activate my card\", please rephrase the following sentences {}:\n{} \n",
            },
            "1": {
                "instruction": "Based on the category of \"age limit\", please rephrase the following sentences {}:\n{} \n",
            },
            "2": {
                "instruction": "Based on the category of \"apple pay or google pay\", please rephrase the following sentences {}:\n{} \n",
            },
            "3": {
                "instruction": "Based on the category of \"atm support\", please rephrase the following sentences {}:\n{} \n",
            },
            "4": {
                "instruction": "Based on the category of \"automatic top up\", please rephrase the following sentences {}:\n{} \n",
            },
            "5": {
                "instruction": "Based on the category of \"balance not updated after bank transfer\", please rephrase the following sentences {}:\n{} \n",
            },
            "6": {
                "instruction": "Based on the category of \"balance not updated after cheque or cash deposit\", please rephrase the following sentences {}:\n{} \n",
            },
            "7": {
                "instruction": "Based on the category of \"beneficiary not allowed\", please rephrase the following sentences {}:\n{} \n",
            },
            "8": {
                "instruction": "Based on the category of \"cancel transfer\", please rephrase the following sentences {}:\n{} \n",
            },
            "9": {
                "instruction": "Based on the category of \"card about to expire\", please rephrase the following sentences {}:\n{} \n",
            }
        }
    },
    'squad': {
        "task_name": "squad",
        "stage": "x2",
        "instruction": "Please mimic the question in the following Question-Context pair {} and ask a new question of the given context:\n {}\n"
    }
}


def get_pe_prompt(args):
    if args.task_name == 'imdb':
        selected_style = ALL_STYLES[random.randrange(len(ALL_STYLES))]
    elif 'yelp' in args.task_name:
        selected_style = ALL_STYLES[random.randrange(len(ALL_STYLES))]
    elif 'squad' in args.task_name:
        selected_style = ALL_STYLES[random.randrange(len(ALL_STYLES))]
    elif 'openreview' in args.task_name:
        selected_style = ALL_OPENREVIEW_STYLES[random.randrange(len(ALL_OPENREVIEW_STYLES))]
    elif 'banking' in args.task_name:
        selected_style = ALL_STYLES[random.randrange(len(ALL_STYLES))]
    
    prompt = copy.deepcopy(PE_PROMPT[args.task_name])
    if 'squad' in args.task_name:
        prompt["instruction"] = copy.deepcopy(prompt["instruction"].format(selected_style, "{}"))
    else:
        for i_label, label in enumerate(prompt["labels"].keys()):
            # print(f'[debug] in <aug_pe_utils.py>, {prompt["labels"][label]["instruction"].format(selected_style, "{}")=}, {selected_style=}, {label=}')
            prompt["labels"][label]["instruction"] = copy.deepcopy(prompt["labels"][label]["instruction"].format(selected_style, "{}"))

    # if variation_type == "yelp_rephrase_tone":
    #     selected_style = ALL_styles[random.randrange(len(ALL_styles))]
    #     prompt = "Based on {}, please rephrase the following sentences {}:\n{} \n".format(
    #         label, selected_style, sequence)
    # elif variation_type == "openreview_rephrase_tone":
    #     selected_style = ALL_OPENREVIEW_styles[random.randrange(
    #         len(ALL_OPENREVIEW_styles))]
    #     prompt = "Based on {}, please rephrase the following sentences {} as a paper review:\n{} \n".format(
    #         label, selected_style, sequence)
    # elif variation_type == "pubmed_rephrase_tone":
    #     selected_style = ALL_PUBMED_styles[random.randrange(
    #         len(ALL_PUBMED_styles))]
    #     prompt = "Please rephrase the following sentences {} as an abstract for medical research paper:\n{} \n".format(
    #         selected_style, sequence)

    return prompt