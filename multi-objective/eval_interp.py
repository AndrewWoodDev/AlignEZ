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
import os

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

def parse_chat(chat_outdir):
    start_idx = 0
    end_idx = 500
    chat_outputs = []
    for current_idx in range(start_idx, end_idx):
        try:
            with open('{}/hh-rlhf_{}_res_{}.json'.format(chat_outdir, 'mistral@7b', current_idx), 'r') as f:
                data = json.load(f)
            chat_outputs.append(data[baseline_name].split('Human:')[0].strip().rstrip())
        except Exception as e:
            continue
    return chat_outputs
            
if __name__ == '__main__':
    device = "cuda"
    path = "RLHFlow/ArmoRM-Llama3-8B-v0.1"
    model = AutoModelForSequenceClassification.from_pretrained(path, device_map="cuda", 
                               trust_remote_code=True, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True)

    model_humor = AutoModelForSequenceClassification.from_pretrained('mohameddhiab/humor-no-humor', device_map="cuda:7", 
                               trust_remote_code=True, torch_dtype=torch.bfloat16)

    tokenizer_humor = AutoTokenizer.from_pretrained('mohameddhiab/humor-no-humor', use_fast=True)
    
    start_idx=0
    end_idx=500

    baseline_name = 'vanila'
    comp_method_name = 'embedding_intervention_output'

    preferences = ['helpful', 'harmless', 'humor']

    chat_outdir = f'interp_llama32_chat_single_harmless'
    harmless_chat_outputs = parse_chat(chat_outdir)

    chat_outdir = f'interp_llama32_chat_single_helpful'
    helpful_chat_outputs = parse_chat(chat_outdir)

    chat_outdir = f'interp_llama32_chat_single_humor'
    humor_chat_outputs = parse_chat(chat_outdir)
    
    preference_pairs = [('helpful', 'humor'), ('helpful', 'harmless'), ('harmless', 'humor')]
    pref_pair_dir_name = []
    pref_pair_dir_all = []
    for pair in preference_pairs:
        dir_all = os.listdir('.')
        pref_pair_name = [dir_ for dir_ in dir_all if 'single' in dir_ and pair[0] in dir_ and pair[1] in dir_]
        # pref_pair_name = [dir_ for dir_ in pref_pair_name if ('new_single' not in dir_ and 'harmless' not in dir_) or ('new_single' in dir_ and 'harmless' in dir_)]

        pref_pair_dir = [dir_+"/roboemb_mlp" for dir_ in pref_pair_name]
        pref_pair_dir = np.sort(pref_pair_dir)
        pref_pair_name = np.sort(pref_pair_name)
        pref_pair_dir_name.extend(pref_pair_name.tolist())
        pref_pair_dir_all.extend(pref_pair_dir.tolist())
    pref_pair_dir_all.extend(['single_llama32_5_helpful:1.0/roboemb_mlp', 'single_llama32_5_harmless:1.0/roboemb_mlp', 'single_llama32_5_humor:1.0/roboemb_mlp'])
    pref_pair_dir_name.extend(['single_llama32_5_helpful:1.0', 'single_llama32_5_harmless:1.0', 'single_llama32_5_humor:1.0'])

    ours_outputs_all = []
    outdir = pref_pair_dir_all[0]
    for outdir in pref_pair_dir_all:
        ours_outputs = []
        questions = []
        vanila_outputs = []
        
        for current_idx in range(start_idx, end_idx):
            try:
                with open('{}/res_{}.json'.format(outdir, current_idx), 'r') as f:
                    data = json.load(f)
                vanila_outputs.append(data[baseline_name].split('Human:')[0].strip().rstrip())
                ours_outputs.append(data[comp_method_name].split('Human:')[0].strip().rstrip())
                questions.append(data['question'])
            except Exception as e:
                continue
        ours_outputs_all.append(ours_outputs)

    vanilla_rewards = []
    helpful_chat_rewards = []
    harmless_chat_rewards = []
    humor_chat_rewards = []
    for q, v, chelp, charm, chum in tqdm(zip(questions, vanila_outputs, helpful_chat_outputs, harmless_chat_outputs, humor_chat_outputs)):
        prompt = q.split("Human: ")[-1].strip().rstrip().split("Assistant: ")[0].strip().rstrip()

        messages_vanilla = [{"role": "user", "content": prompt},
                    {"role": "assistant", "content": v}]

        messages_chat_helpful = [{"role": "user", "content": prompt},
                    {"role": "assistant", "content": chelp}]
        
        messages_chat_harmless = [{"role": "user", "content": prompt},
                    {"role": "assistant", "content": charm}]

        messages_chat_humor = [{"role": "user", "content": prompt},
                    {"role": "assistant", "content": chum}]
        
        r_vanilla = get_hh_rewards(model, tokenizer, messages_vanilla)
        r_chat_helpful = get_hh_rewards(model, tokenizer, messages_chat_helpful)
        r_chat_harmless = get_hh_rewards(model, tokenizer, messages_chat_harmless)
        r_chat_humor = get_hh_rewards(model, tokenizer, messages_chat_humor)

        humor_reward_vanilla = get_humor_rewards(model_humor, tokenizer_humor, v)
        humor_reward_chat_helpful = get_humor_rewards(model_humor, tokenizer_humor, chelp)
        humor_reward_chat_harmless = get_humor_rewards(model_humor, tokenizer_humor, charm)
        humor_reward_chat_humor = get_humor_rewards(model_humor, tokenizer_humor, chum)

        r_vanilla.append(humor_reward_vanilla)
        r_chat_helpful.append(humor_reward_chat_helpful)
        r_chat_harmless.append(humor_reward_chat_harmless)
        r_chat_humor.append(humor_reward_chat_humor)

        vanilla_rewards.append(r_vanilla)
        helpful_chat_rewards.append(r_chat_helpful)
        harmless_chat_rewards.append(r_chat_harmless)
        humor_chat_rewards.append(r_chat_humor)

    vanilla_rewards = np.vstack(vanilla_rewards)
    helpful_chat_rewards = np.vstack(helpful_chat_rewards)
    harmless_chat_rewards = np.vstack(harmless_chat_rewards)
    humor_chat_rewards = np.vstack(humor_chat_rewards)

    print("Vanilla", np.mean(vanilla_rewards, axis=0).tolist())
    print("Helpful chat", np.mean(helpful_chat_rewards, axis=0).tolist())
    print("Harmless chat", np.mean(harmless_chat_rewards, axis=0).tolist())
    print("Humor chat", np.mean(humor_chat_rewards, axis=0).tolist())

    ours_rewards_all = []
    for i, out_ in enumerate(ours_outputs_all):
        ourdir_name = pref_pair_dir_name[i]
        ours_rewards = []
        for q, o in tqdm(zip(questions, out_)):
            prompt = q.split("Human: ")[-1].strip().rstrip().split("Assistant: ")[0].strip().rstrip()
            messages_ours = [{"role": "user", "content": prompt},
                        {"role": "assistant", "content": o}]
            r_ours = get_hh_rewards(model, tokenizer, messages_ours)
            humor_reward_ours = get_humor_rewards(model_humor, tokenizer_humor, o)
            r_ours.append(humor_reward_ours)
            ours_rewards.append(r_ours)
        ours_rewards = np.vstack(ours_rewards)
        print(ourdir_name, np.mean(ours_rewards, axis=0).tolist())
        ours_rewards_all.append(ours_rewards)
    ours_rewards_all = np.vstack(ours_rewards_all)

    scaler = StandardScaler()
    standard_reward = scaler.fit_transform(np.vstack((vanilla_rewards, helpful_chat_rewards, harmless_chat_rewards, humor_chat_rewards, ours_rewards_all)))

    single_len = len(vanilla_rewards)
    standardized_vanilla = standard_reward[:single_len, :]
    standardized_chat_helpful = standard_reward[single_len:2*single_len, :]
    standardized_chat_harmless = standard_reward[2*single_len:3*single_len, :]
    standardized_chat_humor = standard_reward[3*single_len:4*single_len, :]

    standardized_vanilla = np.vstack(standardized_vanilla)
    standardized_chat_helpful = np.vstack(standardized_chat_helpful)
    standardized_chat_harmless = np.vstack(standardized_chat_harmless)
    standardized_chat_humor = np.vstack(standardized_chat_humor)

    print("Standardized Vanilla", np.mean(standardized_vanilla, axis=0).tolist())
    print("Standardized Chat Helpful", np.mean(standardized_chat_helpful, axis=0).tolist())
    print("Standardized Chat Harmless", np.mean(standardized_chat_harmless, axis=0).tolist())
    print("Standardized Chat Humor", np.mean(standardized_chat_humor, axis=0).tolist())

    start_idx = 4*single_len
    for i, out_ in enumerate(ours_outputs_all):
        ourdir_name = pref_pair_dir_name[i]
        standardized_ours = standard_reward[start_idx:start_idx+single_len, :]
        standardized_ours = np.vstack(standardized_ours)
        print(f"Standardized {ourdir_name}", np.mean(standardized_ours, axis=0).tolist())
        start_idx += single_len