import argparse
import os
# workaround to unpickle olf model files
import sys
# TODO
import numpy as np
import torch

from envs import VecPyTorch, VecNormalize, make_vec_envs
from ac import Policy

def get_vec_normalize(venv):
    if isinstance(venv, VecNormalize):
        return venv
    elif hasattr(venv, 'venv'):
        return get_vec_normalize(venv.venv)

    return None

torch.cuda.set_device('cuda:0')
device = torch.device('cuda:0')

parser = argparse.ArgumentParser(description='RL')
parser.add_argument(
    '--seed', type=int, default=1, help='random seed (default: 1)')
parser.add_argument(
    '--log-interval',
    type=int,
    default=10,
    help='log interval, one log per n updates (default: 10)')
parser.add_argument(
    '--env-name',
    default='PongNoFrameskip-v4',
    help='environment to train on (default: PongNoFrameskip-v4)')
parser.add_argument(
    '--load-dir',
    default='./trained_models/',
    help='directory to save agent logs (default: ./trained_models/)')
parser.add_argument(
    '--non-det',
    action='store_true',
    default=False,
    help='whether to use a non-deterministic policy')
args = parser.parse_args()
args.det = not args.non_det

env = make_vec_envs(
    args.env_name,
    args.seed + 1000,
    1,
    None,
    None,
    device=device,
    allow_early_resets=False)

# We need to use the same statistics for normalization as used in training
actor_critic = Policy(
        env.observation_space.shape,
        env.action_space)
actor_critic.to(device)
actor_critic.load_state_dict(
    torch.load(os.path.join(args.load_dir, args.env_name + ".pt"))
)
actor_critic.to(device)
ob_rms = torch.load(os.path.join(args.load_dir, args.env_name + ".vec"))


#vec_norm = get_vec_normalize(env)
vec_norm = None
if vec_norm is not None:
    vec_norm.eval()
    vec_norm.ob_rms = ob_rms

masks = torch.zeros(1, 1)

obs = env.reset()

while True:
    with torch.no_grad():
        value, action, _ = actor_critic.act(
            obs, masks)

    # Obser reward and next obs
    obs, reward, done, _ = env.step(action)

    masks.fill_(0.0 if done else 1.0)

    env.render()
