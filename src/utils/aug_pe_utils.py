
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

