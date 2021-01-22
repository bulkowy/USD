import argparse
import os
# workaround to unpickle olf model files
import sys

import numpy as np
import torch
from envs import VecPyTorch, make_vec_envs
from main import get_vec_normalize

LOAD_PATH = './trained_models/PPO/Reacher-v2_800_.pt'
RENDER = True
EPISODES = 100

device = torch.device('cuda:0')
env = make_vec_envs(
    'Reacher-v2',
    1000,
    1,
    None,
    None,
    device='cuda:0',
    allow_early_resets=False
)

ac, ob_rms = torch.load(LOAD_PATH)
vec_norm = VecPyTorch(get_vec_normalize(env), device)
if vec_norm is not None:
    vec_norm.eval()
    vec_norm.ob_rms = ob_rms

masks = torch.zeros(1, 1)

scores = []
for e in range(EPISODES):
    obs = vec_norm.reset()
    score = 0
    done = False
    while not done:
        with torch.no_grad():
            value, action, _ = ac.act(obs, masks)
        
        obs, reward, done, _ = vec_norm.step(action)
        score += reward
        masks.fill_(0.0 if done else 1.0)

        if RENDER:
            vec_norm.render()
    
    scores.append(score)
    
    print(f"[TEST] Episode {e} | episode score: {score}, avg score: {np.mean(scores)}")