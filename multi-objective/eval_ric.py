"""
Anonymized and cleaned module for evaluation.
"""

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import json
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import argparse

def get_hh_rewards(model, tokenizer, messages):
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt",max_length=4096).to(device)
    with torch.no_grad():
        output = model(input_ids)
        multi_obj_rewards = output.rewards.cpu().float()
        gating_output = output.gating_output.cpu().float()
    preference_score = output.score.cpu().float() 
    obj_transform = model.reward_transform_matrix.data.cpu().float()
    multi_obj_coeffs = gating_output @ obj_transform.T
    return multi_obj_coeffs.cpu().float().numpy().flatten()[[9, 10]].tolist()

def get_humor_rewards(model, tokenizer, text):
    input_ids = tokenizer(text, return_tensors='pt', truncation=True, max_length=4096)
    reward = model(**(input_ids.to(model.device))).logits[0]
    return reward.detach().cpu().tolist()[1]
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--preference", type=str, default=None)

    args = parser.parse_args()
    preference = args.preference    

    device = "cuda"
    path = "RLHFlow/ArmoRM-Llama3-8B-v0.1"
    model = AutoModelForSequenceClassification.from_pretrained(path, device_map="cuda", 
                               trust_remote_code=True, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)

    model_humor = AutoModelForSequenceClassification.from_pretrained('mohameddhiab/humor-no-humor', device_map="cuda:7", 
                               trust_remote_code=True, torch_dtype=torch.bfloat16)

    tokenizer_humor = AutoTokenizer.from_pretrained('mohameddhiab/humor-no-humor', use_fast=True)

    outdir = 'single_llama32_5_harmless:0.1_humor:0.9_helpful:0.0/roboemb_mlp'
    # f'fixed_single_llama32_5_{preference}/roboemb_mlp'
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
    
    chat_outdir = f'llama32_chat_ric'
    chat_outputs = []
    for current_idx in range(start_idx, end_idx):
        try:
            with open('{}/hh-rlhf_{}_res_{}.json'.format(chat_outdir, 'mistral@7b', current_idx), 'r') as f:
                data = json.load(f)
            chat_outputs.append(data[baseline_name].split('Human:')[0].strip().rstrip())
        except Exception as e:
            error_idxs.append(current_idx)
            continue
    vanilla_rewards = []
    ours_rewards = []
    chat_rewards = []
    for q, v, o, c in tqdm(zip(questions, vanila_outputs, ours_outputs, chat_outputs)):
        prompt = q.split("Human: ")[-1].strip().rstrip().split("Assistant: ")[0].strip().rstrip()
        vanilla = v
        ours = o
        chat = c
        messages_vanilla = [{"role": "user", "content": prompt},
                    {"role": "assistant", "content": v}]
        messages_ours = [{"role": "user", "content": prompt},
                    {"role": "assistant", "content": o}]
        messages_chat = [{"role": "user", "content": prompt},
                    {"role": "assistant", "content": c}]
        
        r_vanilla = get_hh_rewards(model, tokenizer, messages_vanilla)
        r_ours = get_hh_rewards(model, tokenizer, messages_ours)
        r_chat = get_hh_rewards(model, tokenizer, messages_chat)

        humor_reward_vanilla = get_humor_rewards(model_humor, tokenizer_humor, vanilla)
        humor_reward_ours = get_humor_rewards(model_humor, tokenizer_humor, ours)
        humor_reward_chat = get_humor_rewards(model_humor, tokenizer_humor, chat)

        r_vanilla.append(humor_reward_vanilla)
        r_ours.append(humor_reward_ours)
        r_chat.append(humor_reward_chat)

        vanilla_rewards.append(r_vanilla)
        ours_rewards.append(r_ours)
        chat_rewards.append(r_chat)

    vanilla_rewards = np.vstack(vanilla_rewards)
    ours_rewards = np.vstack(ours_rewards)
    chat_rewards = np.vstack(chat_rewards)
    # print(len(vanilla_rewards))
    print("Vanilla", np.mean(vanilla_rewards, axis=0).tolist())
    print("Ours", np.mean(ours_rewards, axis=0).tolist())
    print("Chat", np.mean(chat_rewards, axis=0).tolist())
    
    standardized_vanilla = []
    standardized_ours = []
    standardized_chat = []
    # for i in range(vanilla_rewards.shape[1]):
    scaler = MinMaxScaler()
    standard_reward = scaler.fit_transform(np.vstack((vanilla_rewards, ours_rewards, chat_rewards)))
    # print(standard_reward.shape)
    vanilla_len = len(vanilla_rewards)
    ours_len = len(ours_rewards)
    chat_len = len(chat_rewards)
    standardized_vanilla.append(standard_reward[:vanilla_len, :])
    standardized_ours.append(standard_reward[vanilla_len:vanilla_len+ours_len, :])
    standardized_chat.append(standard_reward[vanilla_len+ours_len:, :])

    standardized_vanilla = np.vstack(standardized_vanilla)
    standardized_ours = np.vstack(standardized_ours)
    standardized_chat = np.vstack(standardized_chat)
    # print(standardized_vanilla.shape)
    # exit()
    print("Standardized Vanilla", np.mean(standardized_vanilla, axis=0).tolist())
    print("Standardized Ours", np.mean(standardized_ours, axis=0).tolist())
    print("Standardized Chat", np.mean(standardized_chat, axis=0).tolist())
