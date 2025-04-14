import os

from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
from baukit import TraceDict
import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm

import re
import sys

import pandas as pd

from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import TruncatedSVD

import json

from functools import partial
import argparse

from scipy.linalg import svd
from sklearn.metrics.pairwise import cosine_similarity


JAILBREAKING_DATASETS = ['hh_rlhf_red_team_attempts', 'MaliciousInstruct', 'jbb']

def get_single_activation(model, query):
    MLPS_OUT = [f"model.layers.{i}.mlp" for i in range(model.config.num_hidden_layers)]
    input_ids = tokenizer(query, return_tensors="pt").input_ids.cuda()
    with torch.no_grad():
        with TraceDict(model, MLPS_OUT) as ret:
            output = model(input_ids, output_hidden_states = True)
        mlp_out = [ret[mlp_].output.squeeze().detach().cpu() for mlp_ in MLPS_OUT]
        mlp_out = torch.stack(mlp_out, dim = 0).squeeze().numpy()
    return mlp_out[:, -1, :]

def get_insights_emb(model, pos_insights, neg_insights):
    p_embed = []
    n_embed = []
    for f_pos in tqdm(pos_insights):
        try:
            p_embed_ = get_single_activation(model, f_pos)
            p_embed.append(p_embed_)
        except Exception as e:
            raise e
    for f_neg in tqdm(neg_insights):
        try:
            n_embed_ = get_single_activation(model, f_neg)
            n_embed.append(n_embed_)
        except Exception as e:
            raise e
    return p_embed, n_embed

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
                    pos.append("Human: " + name.strip().rstrip() + "\nAssistant:" + row['insight'].split("\nHuman:")[0])
            elif row['label'] == 0:
                if isinstance(row['insight'], str):
                    neg.append("Human: " + name.strip().rstrip() + "\nAssistant:" + row['insight'].split("\nHuman:")[0])
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
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)

    args = parser.parse_args()
    dataset_name = args.dataset

    model_name = args.model_name
    model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True,
                                                    trust_remote_code=True, 
                                                    # torch_dtype=torch.float16,
                                                    device_map="auto")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    device = "cuda"  

    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads

    if 'llama-3.1' in model_name.lower():
        insights_file = f'multipref_llama31_generated_data/filtered_{dataset_name}_insights.csv'
    elif 'llama-3.2' in model_name.lower():
        insights_file = f'multipref_llama32_generated_data/filtered_{dataset_name}_insights.csv'
    
    df = pd.read_csv(insights_file)
    dict_q = get_insights(df)

    pos_insights_all = []
    neg_insights_all = []
    for idx in dict_q:
        pos_insights_all.extend(dict_q[idx]['pos'])
        neg_insights_all.extend(dict_q[idx]['neg'])
    pos_insights_all = np.array(pos_insights_all)
    neg_insights_all = np.array(neg_insights_all)
    
    pos_emb_all, neg_emb_all = get_insights_emb(model, pos_insights_all, neg_insights_all)
    os.makedirs(f'cache_single/{model_name}_{dataset_name}')
    np.save(f'cache_single/{model_name}_{dataset_name}/pos_emb.npy', pos_emb_all)
    np.save(f'cache_single/{model_name}_{dataset_name}/neg_emb.npy', neg_emb_all)