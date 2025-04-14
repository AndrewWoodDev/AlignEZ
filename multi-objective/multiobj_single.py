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

import sys
sys.path.append('../')
from utils.data_utils import set_seed, load_hhrlhf_template
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

def project_onto_subspace(vector, V_reduced):
    P = V_reduced @ V_reduced.T
    projection = P @ vector
    return projection

def project_onto_subspace_scale(vectors, V_reduced):
    proj = vectors @ V_reduced @ V_reduced.T
    result = (1 - alpha) * vectors + alpha * proj
    return result

def project_orthogonal(vector, V_reduced):
    I = torch.eye(vector.shape[0], device=vector.device)
    P = V_reduced @ V_reduced.T
    orthogonal_projection = (I - P) @ vector
    return orthogonal_projection

def distance_based_orthogonalize(input_vector, vectors, flag):
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
        if 'harmless' in preference:
            m = torch.nn.ReLU()
            scaling = m(alignment)
            result -= scaling * v
        else:
            m = torch.nn.Tanh()
            scaling = m(alignment)
            result += scaling * v
    return result.squeeze()

def lt_modulated_proj(layer_output, layer_name, interventions):
    interventions, flag = interventions[layer_name]
    layer_output = layer_output.squeeze() 
    if len(layer_output.shape) > 1:
        x_test = layer_output[-1,:]
    else:
        x_test = layer_output
    proj = x_test

    neg_subspace = torch.Tensor(interventions).to(model.dtype).to(x_test.device)
    proj = distance_based_orthogonalize(proj, neg_subspace, flag)

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

def vector_subspace_alignment(vector, subspace_basis):
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

    if 'harmless' in preference:
        filtered_subspace = np.argwhere(projection > 0).flatten()
    else:
        filtered_subspace = np.argwhere(projection <= 0).flatten()
    projection = projection[filtered_subspace]
    subspace_basis = subspace_basis[:, filtered_subspace]
    
    # Calculate cosine
    projection = np.array([np.dot(vector, basis) * basis for basis in subspace_basis.T])
    cos_theta = np.linalg.norm(np.sum(projection, axis=1))
    
    return cos_theta, subspace_basis

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--n-layers", type=int, default=None)
    parser.add_argument("--preference", type=str, required=True)

    args = parser.parse_args()

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

    if n_layers_to_edit == None:
        n_layers = n_layers_to_edit

    template_path = '../data/hh-rlhf' # '../data/hh-rlhf'
    SEED = 0
    set_seed(SEED)

    max_new_tokens = 256
    preference = args.preference

    if 'llama-3.1' in model_name.lower():
        outdir=f'fixed_single_llama31_{n_layers_to_edit}_{preference}'
    elif 'llama-3.2' in model_name.lower():
        outdir=f'fixed_single_llama32_{n_layers_to_edit}_{preference}'
    
    print(model_name, outdir)

    if 'llama-3.2' in model_name.lower():
        model_name_short = 'llama32'
    else:
        model_name_short = 'llama31'

    if preference == 'single_helpful':
        insights_file = f'multipref_{model_name_short}_generated_data/filtered_single_helpful_insights.csv'
    elif preference == 'single_harmless':
        insights_file = f'multipref_{model_name_short}_generated_data/filtered_single_harmless_insights.csv'
    else:
        insights_file = f'multipref_{model_name_short}_generated_data/filtered_single_humor_insights.csv'

    df = pd.read_csv(insights_file)
    dict_q = get_insights(df)

    q_all = [f"{dict_q[idx]['question'].strip().rstrip()} " for idx in dict_q]

    q_emb_all = []
    for q in tqdm(q_all):
        q_emb_all.append(get_single_activation(model, q))

    intervention_dict_by_query = {}
    cached_global = []

    for q_idx in tqdm(dict_q):
        v_neg_all = {i: [] for i in range(n_layers)}

        intervention_dict = {}
        vec_subspace_align = []
        for layer_idx in tqdm(range(n_layers)):
            q_emb = q_emb_all[q_idx][layer_idx,:]
            V_reduced = []

            V_reduced = np.load(f'cache_single/{model_name}_{preference}_svd/layer_{layer_idx}.npy')
 
            alignment, subspace = vector_subspace_alignment(q_emb, V_reduced)
            vec_subspace_align.append(alignment)
            v_neg_all[layer_idx] = subspace

        layers_to_edit = np.argsort(vec_subspace_align)[::-1][:n_layers_to_edit]

        for l_idx in layers_to_edit:
            intervention_dict[f"model.layers.{l_idx}.mlp"] = (v_neg_all[l_idx], 1)
        intervention_dict_by_query[q_idx] = intervention_dict
        exp_folder = f"roboemb_mlp"
    
    i = 0
    for i, q in tqdm(enumerate([dict_q[k]['question'] for k in dict_q])): # s: sentence
        print(f"########### {i} ##########")
        raw_query = q
        query = f"{q}"
        
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
            # intervention_dict = intervention_dict_by_query[i]
            if len(intervention_dict) > 0:
                torch.manual_seed(SEED)
                out = get_answer_with_intervention(model, tokenizer, query, \
                                                max_new_tokens=max_new_tokens, interventions=intervention_dict_by_query[i], \
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
            with open('{}/{}/hh-rlhf_{}_res_{}.json'.format(outdir, exp_folder, 'mistral@7b', i), 'w') as f:
                f.write(json.dumps(tmp))
        except Exception as e:
            raise e