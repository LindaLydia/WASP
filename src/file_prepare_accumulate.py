# import torch
# import numpy as np
# from collections import defaultdict



# def minimax():

#     # ballots = 
#     #   [[0.5010706 , 0.25446881, 0.94393603, 0.11651611, 0.66900901,
#     #     0.60922501, 0.82056888, 0.25156271, 0.40287554, 0.39083282],
#     #    [0.26446457, 0.86182751, 0.86332006, 0.17735313, 0.85935296,
#     #     0.93612992, 0.81601465, 0.61668186, 0.44836251, 0.25782967],
#     #    [0.91127431, 0.34537733, 0.65922361, 0.5532714 , 0.03973351,
#     #     0.03676212, 0.46828315, 0.68264401, 0.28110986, 0.4518108 ],
#     #    [0.11381478, 0.14662843, 0.56679908, 0.90118984, 0.10453553,
#     #     0.30679845, 0.65564114, 0.97373476, 0.71827888, 0.35446897],
#     #    [0.6164156 , 0.0347551 , 0.65897203, 0.80476754, 0.2909189 ,
#     #     0.56676932, 0.14997937, 0.21137615, 0.64337731, 0.14450134]]
    
#     ballots = \
#       [[0.68131844, 0.31868156],
#        [0.71354635, 0.28645365],
#        [0.07404568, 0.92595432],
#        [0.40701043, 0.59298957],
#        [0.96449485, 0.03550515],
#        [0.17020286, 0.82979714]]
    
#     ballots = \
#       [[0.35871   , 0.64129   ],
#        [0.65508306, 0.34491694],
#        [0.07749741, 0.92250259],
#        [0.75636307, 0.24363693],
#        [0.71783145, 0.28216855],
#        [0.3804492 , 0.6195508 ]]

#     num_candidates = len(ballots[0])
    
#     # Initialize pairwise preference matrix
#     pairwise_preferences = defaultdict(int)
#     print(pairwise_preferences[(-1,-1)])

#     # Populate the pairwise preference matrix
#     for ballot in ballots:
#         entropy = (1-(-np.sum(ballot * np.log2(ballot))))
#         print(np.argmax(ballot),entropy)
#         for i in range(num_candidates):
#             for j in range(i + 1, num_candidates):
#                 if ballot[i] > ballot[j]:
#                     pairwise_preferences[(i, j)] += entropy
#                 else:
#                     pairwise_preferences[(j, i)] += entropy
    
#     print(pairwise_preferences)

#     # Calculate the maximum margin of defeat for each candidate
#     max_defeats = [0] * num_candidates
#     for i in range(num_candidates):
#         for j in range(num_candidates):
#             if i != j:
#                 defeat_margin = pairwise_preferences[(i, j)] - pairwise_preferences[(j, i)]
#                 if defeat_margin > max_defeats[i]:
#                     max_defeats[i] = defeat_margin
    
#     print(max_defeats)
#     # The Minimax winner is the candidate with the smallest maximum margin of defeat
#     minimax_winner = min(range(num_candidates), key=lambda x: max_defeats[x])



# if __name__ == "__main__":
#     minimax()





import jsonlines
import os, sys

# model_names = ['gpt2-xl', 'llama-2-7b-chat-hf', 'vicuna-7b-1.5v', 'opt-6.7b', 'chatglm3-6b-base', 'flan-t5-xl']
# target_folder = ['100_20', '200_40', '500_100', '1000_100', '1000_500']
# target_samples = [20,40,100,100,500]

task_name = 'imdb'
# task_name = 'squad'
# task_name = 'mnli'
# task_name = 'banking77'
# task_name = 'mnliMisM'
# task_name = 'markednews'
# task_name = 'yelpCategory'
# task_name = 'yelpRating'
# task_name = 'openreviewCategory'
# task_name = 'openreviewRating'
# task_name = 'banking'
model_names = ['gpt2-xl', 'llama-2-7b-chat-hf', 'vicuna-7b-1.5v', 'opt-6.7b', 'chatglm3-6b-base', 'flan-t5-xl'] #
# model_names = ['gpt-3.5-turbo-instruct', 'gpt-4-turbo-preview', 'gpt-4o'] #
# model_names = ['gpt-4-turbo-preview'] #
# target_folder = ['100_20', '1000_200', '1000_500', '6000_1200']
# target_samples = [20, 200, 500, 1200]
target_folder = ['6000_1200', '3000_600', '2000_400', '1000_200', '100_20']
target_samples = [1200, 600, 400, 200, 20]
target_folder = ['6000_6000']
target_samples = [6000]
target_folder = ['3000_600']
target_samples = [600]


for model in model_names:
    # print(model)
    for folder, num_samples in zip(target_folder, target_samples):
        # input_file_path = f'./data_accumulate_start/{task_name}/{model}/10000_2000/train.jsonl'
        input_file_path = f'./data_accumulate_start/{task_name}/{model}/6000_1200/train.jsonl'
        # input_file_path = f'./data_new/{task_name}/{model}/10000/train.jsonl'
        # input_file_path = f'./data_new_dp/{task_name}/{model}/6000/train.jsonl'
        # input_file_path = f'./data_new/{task_name}/{model}/200000/train.jsonl'
        output_file_path = f'./data_accumulate_start/{task_name}/{model}/{folder}/'
        # output_file_path = f'./data_accumulate_start_dp/{task_name}/{model}/{folder}/'
        if not os.path.exists(output_file_path):
            os.makedirs(output_file_path)
        with jsonlines.open(input_file_path, 'r') as reader, jsonlines.open(output_file_path+'train.jsonl', 'w') as writer:
            _counter = 0
            for json_obj in reader:
                writer.write(json_obj)
                _counter += 1
                if _counter == num_samples:
                    break
