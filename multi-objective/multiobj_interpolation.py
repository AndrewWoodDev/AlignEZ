import os

from collections import defaultdict
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

import sys
sys.path.append('../')
from utils.data_utils import set_seed, load_hhrlhf_template
import json

from functools import partial
import argparse

from scipy.linalg import svd
from sklearn.metrics.pairwise import cosine_similarity
import argparse
import copy


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

def distance_based_orthogonalize(input_vector, vectors, weight, preference):
    """
    Reduce influence based on "distance" from the subspace
    sigma controls the decay rate
    """
    if input_vector.dim() == 1:
        input_vector = input_vector.unsqueeze(1)
    original_norms = torch.norm(input_vector, dim=0, keepdim=True)
    # Initialize result at the start
    result = input_vector.clone()

    for i, v in enumerate(vectors.T):
        v = v.unsqueeze(1) if v.dim() == 1 else v
        if len(v.shape) == 0:
            continue
        # Calculate alignment
        alignment = torch.matmul(v.T, input_vector).squeeze()
        # if 'harmless' in preference:
        #     m = torch.nn.ReLU()
        #     scaling = m(alignment)
        #     result -= scaling * v * weight
        # else:
        m = torch.nn.Tanh()
        scaling = m(alignment)
        result += scaling * v * weight
    return result.squeeze()

def lt_modulated_proj(layer_output, layer_name, interventions):
    interv_p_dict = interventions[layer_name]
    layer_output = layer_output.squeeze() 
    if len(layer_output.shape) > 1:
        x_test = layer_output[-1,:]
    else:
        x_test = layer_output
    proj = x_test

    proj_all = []
    weight_all = []
    
    for p in preferences:
        if p not in interv_p_dict:
            continue
        weight = interv_p_dict[p][1]
        if weight > 0:
            neg_subspace = torch.Tensor(interv_p_dict[p][0]).to(model.dtype).to(x_test.device)
            proj = distance_based_orthogonalize(proj, neg_subspace, weight, p)
            proj_all.append(proj)
            weight_all.append(weight)
    if len(weight_all) > 1:
        proj = (weight_all[0] * proj_all[0]) + (weight_all[1] * proj_all[1])
    else:
        proj = proj
    if len(layer_output.shape) > 1:
        layer_output[-1,:] = proj
    else:
        layer_output = proj
        
    layer_output = layer_output.unsqueeze(0)
    layer_output = layer_output.to(model.dtype)
    layer_output = layer_output.to(model.device)
    return layer_output

def get_answer_with_intervention(model, tokenizer, prompt, max_new_tokens=1024, interventions={}, intervention_fn=None):
    out = tokenizer(prompt, return_tensors="pt")
    input_ids = out.input_ids.cuda()
    attention_mask = out.attention_mask.cuda()
    # --- intervention code --- #
    def id(head_output, layer_name): 
        return head_output
    if interventions == {}: 
        intervene = id
        layers_to_intervene = []
    else: 
        intervene = partial(intervention_fn, interventions=interventions)
        layers_to_intervene = list(interventions.keys())
    # --- intervention code --- #
    input_token_len = input_ids.shape[1]
    with torch.inference_mode():
        with TraceDict(model, layers_to_intervene, edit_output=intervene) as ret: 
            model_output = model.generate(inputs = input_ids, 
                                          attention_mask = attention_mask,
                                          max_new_tokens=max_new_tokens,
                                          eos_token_id=tokenizer.eos_token_id,
                                          use_cache=True,
                                          do_sample=True,
                                         )
        outstr = tokenizer.decode(model_output[0], skip_special_tokens=True)
    torch.cuda.empty_cache()
    return outstr

def vector_subspace_alignment(vector, subspace_basis, preference):
    """
    Measure alignment between a vector and a subspace.
    
    Args:
        vector: The vector to measure (numpy array)
        subspace_basis: List of basis vectors defining the subspace (numpy array)
        
    Returns:
        cos_theta: Cosine of the angle between vector and its projection
    """
    # Normalize vector
    vector = vector / np.linalg.norm(vector)
    # Calculate projection
    projection = np.array([np.dot(vector, basis) for basis in subspace_basis.T])

    # if 'harmless' in preference:
    #     filtered_subspace = np.argwhere(projection > 0).flatten()
    # else:
    filtered_subspace = np.argwhere(projection <= 0).flatten()
    projection = projection[filtered_subspace]
    subspace_basis = subspace_basis[:, filtered_subspace]
    
    # Calculate cosine
    projection = np.array([np.dot(vector, basis) * basis for basis in subspace_basis.T])
    cos_theta = np.linalg.norm(np.sum(projection, axis=1))
    
    return cos_theta, subspace_basis


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--preference-pair-idx", type=int, required=True)
    args = parser.parse_args()


    model_name = 'meta-llama/Llama-3.2-3B'
    n_layers_to_edit = 5


    model = AutoModelForCausalLM.from_pretrained(model_name, low_cpu_mem_usage=True,
                                                    trust_remote_code=True, 
                                                    # torch_dtype=torch.float16,
                                                    device_map="auto")
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    device = "cuda"  

    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads

    template_path = '../data/hh-rlhf' # '../data/hh-rlhf'
    SEED = 0
    set_seed(SEED)

    max_new_tokens = 256

    preferences = ['helpful', 'humor', 'harmless']

    template = load_hhrlhf_template(template_path)
    fschat = template['fschat']

    insight_files ={
        'helpful': f'multipref_llama31_generated_data/filtered_helpful_insights.csv',
        'harmless': f'multipref_llama31_generated_data/filtered_helpful_insights.csv',
        'humor': f'multipref_llama31_generated_data/filtered_helpful_insights.csv'
    }
    dfs = {
        'helpful': pd.read_csv(insight_files['helpful']),
        'harmless': pd.read_csv(insight_files['harmless']),
        'humor': pd.read_csv(insight_files['humor'])
    }

    dict_qs = {
        'helpful': get_insights(dfs['helpful']),
        'harmless': get_insights(dfs['harmless']),
        'humor': get_insights(dfs['humor'])
    }

    q_emb_all_dict = {}

    for p in preferences:
        q_all = [f"{dict_qs[p][idx]['question'].strip().rstrip()} " for idx in dict_qs[p]]

        q_emb_all = []
        for q in tqdm(q_all, desc=f"q_emb_all...Embedding {p} preferences"):
            q_emb_all.append(get_single_activation(model, q))
        q_emb_all_dict[p] = q_emb_all


    intervention_dict_by_query = defaultdict(dict)

    V_reduced_p_cache = {
    }

    for p in preferences:
        V_reduced_p_cache[p] = {}
        for layer_idx in tqdm(range(n_layers), desc=f"Loading {p} subspace"):
            V_reduced_p_cache[p][layer_idx] = np.load(f'cache_single/{model_name}_single_{p}_svd/layer_{layer_idx}.npy')

    import time

    for p in preferences:
        dict_q = dict_qs[p]
        for q_idx in tqdm(dict_q):
            # intervention_dict_by_query[q_idx] = {}
            v_neg_all = {i: [] for i in range(n_layers)}

            intervention_dict = {}
            vec_subspace_align = []

            q_emb_all = q_emb_all_dict[p]
            for layer_idx in tqdm(range(n_layers)):
                q_emb = q_emb_all[q_idx][layer_idx,:]
                
                V_reduced = V_reduced_p_cache[p][layer_idx]
                alignment, subspace = vector_subspace_alignment(q_emb, V_reduced, p)
                vec_subspace_align.append(alignment)
                v_neg_all[layer_idx] = subspace

            layers_to_edit = np.argsort(vec_subspace_align)[::-1][:n_layers_to_edit]

            for l_idx in layers_to_edit:
                if f"model.layers.{l_idx}.mlp" not in intervention_dict:
                    intervention_dict[f"model.layers.{l_idx}.mlp"] = {}
                intervention_dict[f"model.layers.{l_idx}.mlp"][p] = v_neg_all[l_idx]
            if q_idx not in intervention_dict_by_query:
                intervention_dict_by_query[q_idx] = intervention_dict
            else:
                for layer_key in intervention_dict:
                    if layer_key in intervention_dict_by_query[q_idx]:
                        intervention_dict_by_query[q_idx][layer_key].update(intervention_dict[layer_key])
                    else:
                        intervention_dict_by_query[q_idx][layer_key] = intervention_dict[layer_key]
        
    exp_folder = f"roboemb_mlp"
    preference_pairs = [('helpful', 'harmless'), ('harmless', 'humor')]

    # for preference_pair in preference_pairs:
    preference_pair = preference_pairs[args.preference_pair_idx]
    p1, p2 = preference_pair
    p3 = [p for p in preferences if p not in preference_pair][0]
    for eta in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        interpolation_rate_dict = {
            p1: eta,
            p2: 1 - eta,
            p3: 0.0
        }

        if 'llama-3.1' in model_name.lower():
            outdir=f"new_single_llama31_{n_layers_to_edit}_{p1}:{interpolation_rate_dict[p1]}_{p2}:{interpolation_rate_dict[p2]}"
        elif 'llama-3.2' in model_name.lower():
            outdir=f"new_single_llama32_{n_layers_to_edit}_{p1}:{interpolation_rate_dict[p1]}_{p2}:{interpolation_rate_dict[p2]}"

        i = 0
        for i, q in tqdm(enumerate([dict_q[k]['question'] for k in dict_q])): # s: sentence
            print(p1, p2, eta, 1-eta)
            print(f"########### {i} ##########")
            raw_query = q
            query = f"{q}"
            # query = fschat + '\n' + raw_query
            print('query')
            print(raw_query)
            try:    
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
                intervention_dict = copy.deepcopy(intervention_dict_by_query[i])
                for layer_key in intervention_dict:
                    for pref in intervention_dict[layer_key]:
                        intervention_dict[layer_key][pref] = (intervention_dict[layer_key][pref], interpolation_rate_dict[pref])

                if len(intervention_dict) > 0:
                    torch.manual_seed(SEED)
                    out = get_answer_with_intervention(model, tokenizer, query, \
                                                    max_new_tokens=max_new_tokens, interventions=intervention_dict,
                                                    intervention_fn=lt_modulated_proj)
                else:
                    out = ans_query
                print('OURS')
                out_ans = out.split(query)[-1].split("Human: ")[0].strip().rstrip().split("Assistant: ")[0].strip().rstrip()
                print(out_ans)

                tmp = {'question': raw_query,
                    'vanila': ans_query,
                    'embedding_intervention_output': out_ans
                    }
            
                if not os.path.exists(os.path.join(outdir, exp_folder)):
                    os.makedirs(os.path.join(outdir, exp_folder))
                with open('{}/{}/res_{}.json'.format(outdir, exp_folder, i), 'w') as f:
                    f.write(json.dumps(tmp))
            except Exception as e:
                raise e
                