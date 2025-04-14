"""
Anonymized and cleaned module for evaluation.
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import json
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def get_rewards(model, tokenizer, messages):
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt",max_length=4096).to(device)
    with torch.no_grad():
        output = model(input_ids)
        multi_obj_rewards = output.rewards.cpu().float()
        gating_output = output.gating_output.cpu().float()
    # The preference score for the response, aggregated from the 
    # multi-objective rewards with the gating layer
    preference_score = output.score.cpu().float()  
    # We apply a transformation matrix to the multi-objective rewards
    # before multiplying with the gating layer's output. This mainly aims
    # at reducing the verbosity bias of the original reward objectives
    obj_transform = model.reward_transform_matrix.data.cpu().float()
    # The final coefficients assigned to each reward objective
    multi_obj_coeffs = gating_output @ obj_transform.T
    return multi_obj_rewards.cpu().float().numpy().flatten()[:5]  * 5 - 0.5
            

if __name__ == '__main__':
    device = "cuda"
    path = "RLHFlow/ArmoRM-Llama3-8B-v0.1"
    model = AutoModelForSequenceClassification.from_pretrained(path, device_map=device, 
                               trust_remote_code=True, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)

    outdir = 'new_llama32_helpsteer_helpfulness_coherence_complexity_correctness_verbosity_nlayers_5_all/roboemb_mlp'
    baseline_name = 'vanila'
    comp_method_name = 'embedding_intervention_output'

    ours_outputs = []
    questions = []
    vanila_outputs = []
    error_idxs = []
    start_idx=0
    end_idx=500
    for current_idx in range(start_idx, end_idx):
        try:
            with open('{}/hh-rlhf_{}_res_{}.json'.format(outdir, 'mistral@7b', current_idx), 'r') as f:
                data = json.load(f)
            vanila_outputs.append(data[baseline_name].split('Human:')[0].strip().rstrip())
            ours_outputs.append(data[comp_method_name].split('Human:')[0].strip().rstrip())
            questions.append(data['question'])
        except Exception as e:
            error_idxs.append(current_idx)
            continue

    chat_dir = 'llama32_chat_helpsteer'
    chat_outputs = []
    for current_idx in range(start_idx, end_idx):
        try:
            with open('{}/hh-rlhf_{}_res_{}.json'.format(chat_dir, 'mistral@7b', current_idx), 'r') as f:
                data = json.load(f)
            chat_outputs.append(data[baseline_name].split('Human:')[0].strip().rstrip())
        except Exception as e:
            error_idxs.append(current_idx)
            continue

    vanilla_rewards = []
    ours_rewards = []
    chat_rewards = []
    for q, v, o, c in tqdm(zip(questions, vanila_outputs, ours_outputs, chat_outputs)):
        prompt = q
        vanilla = v
        ours = o
        messages_vanilla = [{"role": "user", "content": prompt},
                    {"role": "assistant", "content": v}]
        messages_ours = [{"role": "user", "content": prompt},
                    {"role": "assistant", "content": o}]
        messages_chat = [{"role": "user", "content": prompt},
                    {"role": "assistant", "content": c}]
        
        r_vanilla = get_rewards(model, tokenizer, messages_vanilla)
        r_ours = get_rewards(model, tokenizer, messages_ours)
        r_chat = get_rewards(model, tokenizer, messages_chat)

        vanilla_rewards.append(r_vanilla)
        ours_rewards.append(r_ours)
        chat_rewards.append(r_chat)

    vanilla_rewards = np.vstack(vanilla_rewards)
    ours_rewards = np.vstack(ours_rewards)
    chat_rewards = np.vstack(chat_rewards)

    print("Vanilla", np.mean(vanilla_rewards, axis=0).tolist())
    print("Ours", np.mean(ours_rewards, axis=0).tolist())
    print("Chat", np.mean(chat_rewards, axis=0).tolist())
    
    standardized_vanilla = []
    standardized_ours = []
    standardized_chat = []
    for i in range(vanilla_rewards.shape[1]):
        scaler = MinMaxScaler()
        standard_reward = scaler.fit_transform(np.hstack((vanilla_rewards[:,i], ours_rewards[:,i], chat_rewards[:,i])).reshape((-1, 1))).flatten()
        vanilla_len = len(vanilla_rewards[:,i])

        standardized_vanilla.append(standard_reward[:vanilla_len].flatten())
        standardized_ours.append(standard_reward[vanilla_len:2*vanilla_len].flatten())
        standardized_chat.append(standard_reward[2*vanilla_len:].flatten())

    standardized_vanilla = np.vstack(standardized_vanilla)
    standardized_ours = np.vstack(standardized_ours)
    standardized_chat = np.vstack(standardized_chat)
    # print(standardized_vanilla.shape)
    print("Standardized Vanilla", np.mean(standardized_vanilla, axis=1).tolist())
    print("Standardized Ours", np.mean(standardized_ours, axis=1).tolist())
    print("Standardized Chat", np.mean(standardized_chat, axis=1).tolist())

    

    