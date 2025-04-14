import os

from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

import sys
sys.path.append('../')
from utils.data_utils import set_seed, load_hhrlhf_template
import json

import argparse
import pandas as pd

def get_insights(df):
    dict_q = {}
    grouped = df.groupby("question")[['insight', 'label']]
    i = 0
    for name, group in grouped:
        dict_q[i] =  {'question': name, 'pos': [], 'neg': []}
        pos = []
        neg = []
        for _, row in group.iterrows():
            if row['label'] == 1:
                if isinstance(row['insight'], str):
                    pos.append(name.strip().rstrip() + " " + row['insight'].split("\nHuman:")[0])
            elif row['label'] == 0:
                if isinstance(row['insight'], str):
                    neg.append(name.strip().rstrip() + " " + row['insight'].split("\nHuman:")[0])
        dict_q[i]['pos'] = pos
        dict_q[i]['neg'] = neg
        i += 1
        
    pos_insight_id_tracker = 0
    neg_insight_id_tracker = 0
    for idx in dict_q:
        if idx == 0:
            dict_q[idx]['pos_insight_idx'] = [i for i in range(len(dict_q[idx]['pos']))]
            dict_q[idx]['neg_insight_idx'] = [i for i in range(len(dict_q[idx]['neg']))]
        else:
            dict_q[idx]['pos_insight_idx'] = [pos_insight_id_tracker+i for i in range(len(dict_q[idx]['pos']))]
            dict_q[idx]['neg_insight_idx'] = [neg_insight_id_tracker+i for i in range(len(dict_q[idx]['neg']))]
        pos_insight_id_tracker += len(dict_q[idx]['pos_insight_idx'])
        neg_insight_id_tracker += len(dict_q[idx]['neg_insight_idx'])
    return dict_q

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--preference", type=str, required=True)

    args = parser.parse_args()

    model_name = args.model_name
    preference = args.preference
    model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True,
                                                    trust_remote_code=True, 
                                                    device_map="auto")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    device = "cuda"  

    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads

    template_path = '../data/hh-rlhf' # '../data/hh-rlhf'
    SEED = 0
    set_seed(SEED)

    
    if 'llama-3.1' in model_name.lower():
        model_name_short = 'llama31'
        outdir=f'llama31_chat_{preference}'
    elif 'llama-3.2' in model_name.lower():
        model_name_short = 'llama32'
        outdir=f'interp_llama32_chat_{preference}'

    max_new_tokens = 256

    insights_file = f'multipref_llama31_generated_data/filtered_helpful_insights.csv'
    # if preference == 'single_helpful':
    #     insights_file = f'multipref_{model_name_short}_generated_data/filtered_single_helpful_insights.csv'
    # elif preference == 'single_harmless':
    #     insights_file = f'multipref_{model_name_short}_generated_data/filtered_single_harmless_insights.csv'
    # else:
    #     insights_file = f'multipref_{model_name_short}_generated_data/filtered_single_humor_insights.csv'

    df = pd.read_csv(insights_file)
    dict_q = get_insights(df)

    i = 0
    for i, q in tqdm(enumerate([dict_q[k]['question'] for k in dict_q])): # s: sentence
        print(f"########### {i} ##########")
        raw_query = q.strip().rstrip()
        query = f"The Assistant's answer should have the following characteristic: {preference.split('_')[-1]}.\n{q}"
        # query = raw_query
        print('query')
        print(query)
    
        out = tokenizer(query, return_tensors="pt")
        input_ids = out.input_ids.cuda()
        attention_mask = out.attention_mask.cuda()
        torch.manual_seed(SEED)
        model_output = model.generate(inputs = input_ids, 
                                        attention_mask = attention_mask,
                                        max_new_tokens=max_new_tokens,
                                        eos_token_id=tokenizer.eos_token_id,
                                        use_cache=True,
                                        do_sample=True,
                                        )
        vanila_output = tokenizer.decode(model_output[0], skip_special_tokens=True)
        ans_query = vanila_output.split(query)[-1].split("Human: ")[0].strip().rstrip().split("Assistant: ")[0].strip().rstrip()
        print('VANILLA MODEL')
        print(ans_query)
        
        tmp = {'question': raw_query,
            'vanila': ans_query,
            }
    
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        with open('{}/hh-rlhf_{}_res_{}.json'.format(outdir, 'mistral@7b', i), 'w') as f:
            f.write(json.dumps(tmp))
