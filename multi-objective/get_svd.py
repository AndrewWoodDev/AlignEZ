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
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--n-layers", type=int, default=5)

    args = parser.parse_args()
    dataset_name = args.dataset

    n_layers_to_edit = args.n_layers
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

    if dataset_name == None:
        preferences = ['helpful', 'harmless', 'humor']
        pos_emb_all = []
        neg_emb_all = []
        for p in preferences:
            pos_emb_ = np.load(f'cache_single/{model_name}_{p}/pos_emb.npy')
            neg_emb_ = np.load(f'cache_single/{model_name}_{p}/neg_emb.npy')
            pos_emb_all.append(pos_emb_)
            neg_emb_all.append(neg_emb_)
        pos_emb_all = np.vstack(pos_emb_all)
        neg_emb_all = np.vstack(neg_emb_all)
    else:
        pos_emb_all = np.load(f'cache_single/{model_name}_{dataset_name}/pos_emb.npy')
        neg_emb_all = np.load(f'cache_single/{model_name}_{dataset_name}/neg_emb.npy')

    for layer_idx in tqdm(range(n_layers)):
        pos_sample_all = np.vstack([p[layer_idx, :] for p in pos_emb_all])
        neg_samples_all = np.vstack([n[layer_idx, :] for n in neg_emb_all])

        filtered_idx = []
        for i, (p, n) in enumerate(zip(pos_sample_all, neg_samples_all)):
            if cosine_similarity(p.reshape(1,-1),n.reshape(1,-1)) > 0.8:
                continue
            filtered_idx.append(i)

        pos_sample_all = pos_sample_all[filtered_idx]
        neg_samples_all = neg_samples_all[filtered_idx]

        # if 'harmless' in dataset_name:
        #     directions_global =  neg_samples_all - pos_sample_all
        # else:
        directions_global =  pos_sample_all - neg_samples_all
        
        U, s, Vt = np.linalg.svd(directions_global, full_matrices=False)
        explained_variance_ratio = (s ** 2) / (s ** 2).sum()
        cumulative_variance_ratio = np.cumsum(explained_variance_ratio)
        k = np.argmax(cumulative_variance_ratio >= 0.90) + 1
        
        V_reduced = Vt[:k].T  # Take first k singular vectors)

        if dataset_name == None:
            dataset_name = 'all'
            
        if not os.path.isdir(f'cache_single/{model_name}_{dataset_name}_svd'):
            os.makedirs(f'cache_single/{model_name}_{dataset_name}_svd', exists_ok=True)
        np.save(f'cache_single/{model_name}_{dataset_name}_svd/layer_{layer_idx}.npy', V_reduced)
