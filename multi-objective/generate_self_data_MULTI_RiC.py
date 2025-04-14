import os

from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import torch
from tqdm import tqdm

import sys
import argparse

sys.path.append('../utils/')
import config

sys.path.append('../data/')

sys.path.append('../')

import pandas as pd
from utils.data_utils import load_hhrlhf_template
import re
from functools import partial
from utils.inference import vanila_inference, StopOnTokens

from datasets import load_from_disk


def parse_feedback_to_list(feedback_str):
    feedback_str = feedback_str.strip().rstrip()
    if len(feedback_str) == 0:
        return []
    feedback_str = re.sub('\d.', ';', feedback_str)
    feedback_list = feedback_str.split(";")
    feedback_list = [s.split(')')[-1].replace(",", "").replace(".","").strip().rstrip() for s in feedback_list]
    indexes = np.unique(feedback_list, return_index=True)[1]
    feedback_list = [feedback_list[index] for index in sorted(indexes)]
    feedback_list = [f for f in feedback_list if len(f) > 0] 
    return feedback_list

def parse_list_to_string(feedback_list):
    feedback_list = [f'{i+1}) {s}' for i,s in enumerate(feedback_list)]
    str_ = ", ".join(feedback_list)
    return str_

def get_insights(query):
    human_query = query
    helpful_insights = []
    harmful_insights = []
    if ('?' in human_query) or (human_query.lower().startswith('what ')) or (human_query.lower().startswith('why ')) or (human_query.lower().startswith('how ')):
        type_ = 'question'
    else:
        type_ = 'statement'
    feedback_query = f"{human_query.strip().rstrip()}\nYou are a {keyword_positive} assistant. Your answer to this query should: "
    feedback_output = feedback_fun(raw_query=feedback_query, model=model, tokenizer=tokenizer).split(feedback_query)[-1].strip().rstrip()
    
    feedback_output_pos = feedback_output.split(feedback_query)[-1].strip().rstrip().split("\nHuman:")[0].strip().rstrip()
    print("HELPFUL", feedback_output_pos)

    feedback_query = f"{human_query.strip().rstrip()}\nPretend you are a {keyword_negative} assistant. Your response to this query should: "
    feedback_output = feedback_fun(raw_query=feedback_query, model=model, tokenizer=tokenizer).split(feedback_query)[-1].strip().rstrip()
    
    feedback_output_neg = feedback_output.split(feedback_query)[-1].strip().rstrip().split("\nHuman:")[0].strip().rstrip()
    print("HARMFUL", feedback_output_neg)
    return feedback_output_pos, feedback_output_neg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--preference", type=str, required=True)

    allowed_preferences = ['helpful', 'harmless', 'humor']

    preference_keywords_positive = {
        'helpful': 'helpful',
        'harmless': 'safety-first',
        'humor': 'humorous',
    }

    preference_keywords_negative = {
        'helpful': 'useless',
        'harmless': 'malicious',
        'humor': 'dull and boring',
    }
    
    args = parser.parse_args()
    model_name = args.model_name
    preference = args.preference
    assert preference in preference_keywords_positive

    keyword_positive = preference_keywords_positive[preference]
    keyword_negative = preference_keywords_negative[preference]
    

    # model_name = "mistralai/Mistral-7B-v0.3"
    model =  AutoModelForCausalLM.from_pretrained(model_name, 
                                                low_cpu_mem_usage=True, device_map="auto", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # data = load_dataset("nvidia/HelpSteer2")['validation'].to_pandas()
    # data.rename(columns={"prompt": "instruction"}, inplace=True)
    if preference == 'helpful':
        data = load_from_disk("/home/ubuntu/RiC/ric/datasets_FIXED_sampled/test_helpful_200.hf").to_pandas()[['prompt', 'chosen', 'rejected']]
    elif preference == 'harmless':
        data = load_from_disk("/home/ubuntu/RiC/ric/datasets_FIXED_sampled/test_harmless_200.hf").to_pandas()[['prompt', 'chosen', 'rejected']]
    else:
        data = load_from_disk("/home/ubuntu/RiC/ric/datasets_FIXED_sampled/test_humorous_200.hf").to_pandas()[['prompt', 'chosen', 'rejected']]

    # data = pd.concat([data_helpful, data_harmless, data_humor])
    data.rename(columns={"prompt": "instruction"}, inplace=True)

    template_path = '../data/hh-rlhf' # '../data/hh-rlhf'
    template = load_hhrlhf_template(template_path)
    fschat = template['fschat']

    feedback_fun = partial(vanila_inference, fschat=fschat, max_new_tokens=50)
    gen_fun = partial(vanila_inference, fschat=fschat, max_new_tokens=200)

    df_data = {'question': [], 'insight': [], 'label':[]}

    for epoch_test, batch_test in data.iterrows(): # s: sentence
        print(f"########### {epoch_test} ##########")
        raw_query = batch_test['instruction']
        query = f"{raw_query.strip().rstrip()} "
        print('query')
        print(query)

        helpful_insights, harmful_insights = get_insights(raw_query)

        help_ans = set()
        if len(helpful_insights) > 0:
            helpful_insights = f"You are a {keyword_positive} assistant. Your answer should {helpful_insights}.\n{raw_query.strip().rstrip()} "
        else:
            helpful_insights = f"You are a {keyword_positive} assistant. Your answer should be {keyword_positive}.\n{raw_query.strip().rstrip()} "

        out_ = gen_fun(raw_query=helpful_insights, model=model, tokenizer=tokenizer)
        torch.cuda.empty_cache()
        out_ = out_.split(helpful_insights)[-1].strip().rstrip().split('Human: ')[0].strip().rstrip()

        # helpful_insights[i] = helpful_insights[i]+out_
        print('[SAMPLE HELP]',out_)
        df_data['question'].append(raw_query)
        df_data['insight'].append(out_)
        df_data['label'].append(1)
        harm_ans = set()
        # for i in tqdm(range(len(harmful_insights))):
        if len(harmful_insights) > 0:
            harmful_insights = f"Pretend you are a {keyword_negative} assistant. Your answer should {harmful_insights}.\n{raw_query.strip().rstrip()}"
        else:
            harmful_insights = f"Pretend you are a {keyword_negative} assistant. Your answer should be {keyword_negative}.\n{raw_query.strip().rstrip()}"
        out_ = gen_fun(raw_query=harmful_insights, model=model, tokenizer=tokenizer)
        torch.cuda.empty_cache()
        out_ = out_.split(harmful_insights)[-1].strip().rstrip().split('Human: ')[0].strip().rstrip()

        # harmful_insights[i] = harmful_insights[i] + out_
        print('[SAMPLE HARM]',out_)
        df_data['question'].append(raw_query)
        df_data['insight'].append(out_)
        df_data['label'].append(0)

    df_data = pd.DataFrame(df_data)

    if 'nemo' in model_name.lower():
        model_name_short = 'nemo'
    elif '70' in model_name:
        model_name_short = 'llama31-70b'
    elif '3.1' in model_name:
        model_name_short = 'llama31'
    elif '3.2-1b' in model_name.lower():
        model_name_short = 'llama32-1b'
    else:
        model_name_short = 'llama32'
    dir_name = f'multipref_{model_name_short}_generated_data'
    if not os.path.isdir(dir_name):
        os.makedirs(dir_name)
    df_data.to_csv(f'{dir_name}/single_{preference}_insights.csv', index=False)
