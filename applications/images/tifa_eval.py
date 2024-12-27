from tifascore import get_question_and_answers, filter_question_and_answers, UnifiedQAModel, tifa_score_benchmark, tifa_score_single,  VQAModel
from tifascore import get_llama2_pipeline, get_llama2_question_and_answers
import logging

import json
import openai
import numpy as np
import os, sys

with open("taskinfo.json") as f:
    TASKINFO = json.load(f)
print(TASKINFO)
TASKS = set([elem['dir_'] for elem in TASKINFO])
print("TASKS", TASKS)

# for some reason, API returns empty list so manually obtained from OpenAI playground. identical settings were used.
cactus_questions = [{'caption': 'a cactus', 'element': 'cactus', 'question': 'is this a cactus?', 'choices': ['yes', 'no'], 'answer': 'yes', 'element_type': 'object'}, 
                    {'caption': 'a cactus', 'element': 'cactus', 'question': 'what plant is in the picture?', 'choices': ['cactus', 'rose', 'sunflower', 'fern'], 'answer': 'cactus', 'element_type': 'object'}]
dandelion_questions = [{'caption': 'dandelion', 'element': 'dandelion', 'question': 'is this a dandelion?', 'choices': ['yes', 'no'], 'answer': 'yes', 'element_type': 'object'}, 
                    {'caption': 'dandelion', 'element': 'dandelion', 'question': 'what plant is shown in the image?', 'choices': ['dandelion', 'rose', 'sunflower', 'tulip'], 'answer': 'dandelion', 'element_type': 'object'}]


def main():
    for TASK in range(len(TASKINFO)):
        dir_ = TASKINFO[TASK]["dir_"]
        if "sunflower" not in dir_: continue
        dir_name="log_tifa"
        os.makedirs(dir_name, exist_ok=True)  # Creates "logs" if it doesn't exist
        log_file_path = os.path.join(dir_name, f"{dir_}.log")
        logging.info(log_file_path)
        # Remove all handlers associated with the root logger object.
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        
        
        # prepare the models
        vqa_model = VQAModel("mplug-large")
        q_dir="gpt_questions"
        unifiedqa_model = UnifiedQAModel("allenai/unifiedqa-v2-t5-large-1363200")
        
        text_AB = TASKINFO[TASK]["text_AB"] 

        if TASK == 6: # see note above on cactus and dandelion questions
            questions_AB = cactus_questions
        elif TASK == 13:
            questions_AB = dandelion_questions
        else:
            questions_AB = get_question_and_answers(text_AB)
        print(questions_AB)

        filtered_questions_AB = filter_question_and_answers(unifiedqa_model, questions_AB)
        print("filtered", filtered_questions_AB)

        text_BA = TASKINFO[TASK]["text_BA"] 
        logging.info(text_BA)
        
        questions_BA = get_question_and_answers(text_BA)
        print(questions_BA)
        filtered_questions_BA = filter_question_and_answers(unifiedqa_model, questions_BA)
        print("filtered", filtered_questions_BA)

        os.makedirs(q_dir, exist_ok=True)  
        q_dict = {"filtered_questions_AB": filtered_questions_AB, 
                  "filtered_questions_BA": filtered_questions_BA}
        with open(os.path.join(q_dir, f"{dir_}.json"), 'w') as f:
            json.dump(q_dict, f)

        logging.info('='*100)
        score_dict = {}
        for img_num in range(20):
            for method in ["or",  "sd_ab_or", "sd_ba_or", "sd_a", "sd_b"]: 
                logging.info(f"METHOD:::{method}")
                img_path = f"/projects/superdiff/saved_sd_results/{method}/{dir_}/{img_num}.png"

                
                result_AB = tifa_score_single(vqa_model, filtered_questions_AB, img_path)
                logging.info(f"TIFA score AB is {result_AB['tifa_score']}")
                result_BA = tifa_score_single(vqa_model, filtered_questions_BA, img_path)
                logging.info(f"TIFA score BA is {result_BA['tifa_score']}")
                score_ab = result_AB['tifa_score']
                score_ba = result_BA['tifa_score']
                min_score = min(score_ab, score_ba)
                max_score = max(score_ab, score_ba)
                if score_ab <= score_ba:
                    result = result_AB
                else:
                    result = result_BA
                logging.info(f"min TIFA score is {result['tifa_score']}")
                logging.info(result)

                if method not in score_dict:
                    score_dict[method] = {'min': [],
                                          'max': [],
                                          'all': []}

                score_dict[method]['min'].append(min_score)
                score_dict[method]['max'].append(max_score)
                score_dict[method]['all'].append((score_ab, score_ba))
                logging.info('='*100)
        logging.info("FINAL DICT") 
        logging.info(score_dict)
        logging.info('='*100)
        logging.info('='*100)
        for method in ["or",  "sd_ab_or", "sd_ba_or", "sd_a", "sd_b"]: 
            score_arr = score_dict[method]['min']
            logging.info("METHOD:: {} | mean ± std:: {:.4f} ± {:.4f}".format(method, np.mean(score_arr), np.std(score_arr)))
if __name__ == "__main__":
    main()
