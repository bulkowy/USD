import copy
import glob
import os
import time
from collections import deque

import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ac import Policy
from envs import make_vec_envs, VecNormalize
from ppoAgent import Memory, Agent
import wandb
import reacheredited

from mujoco_py import GlfwContext
import cv2


lr_ = [1e-3, 3e-4, 1e-4]
epsilon_ = [1e-4, 5e-5, 1e-5]
gamma_ = [0.95, 0.97, 0.99]
gae_lambda_ = [0.93, 0.95, 0.97]
num_mini_batch_ = [32, 64, 128]


LR = 3e-4
EPSILON = 1e-5
GAMMA = 0.99
GAE_LAMBDA = 0.95
NUM_MINI_BATCH = 32

TUNING = True

# -- CONFIG
ENV_NAME = 'Reacher-v2'
# ENV_NAME = 'reachere-v2'
ENTROPY = 0.00
SEED = 1000
VALUE_LOSS_COEF = 0.5
MAX_GRAD_NORM = 0.5
NUM_PROCESSES = 1
NUM_STEPS = 2048
PPO_EPOCH = 10
NUM_MINI_BATCH = 32
CLIP_PARAM = 0.2
LOG_INTERVAL = 1
RENDER_INTERVAL = 20
RENDER = True
ENV_STEPS = 120 * NUM_STEPS 
LOG_DIR = './tmp/gym'
LOG_WANDB = True


OFFSCREEN = True

if OFFSCREEN:
    GlfwContext(offscreen=True)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')


def get_vec_normalize(venv):
    if isinstance(venv, VecNormalize):
        return venv
    elif hasattr(venv, 'venv'):
        return get_vec_normalize(venv.venv)

    return None

def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def main():
    if LOG_WANDB:
        if TUNING:
            run = wandb.init(project=f'{ENV_NAME}_PPO-HP_SEARCH',
                    name=f'PPO/{ENV_NAME}-{LR}-{EPSILON}-{GAMMA}-{GAE_LAMBDA}-{NUM_MINI_BATCH}', reinit=True)
        else:
            wandb.init(project=f'{ENV_NAME}_PPO',
                    name=f'PPO/{ENV_NAME}')

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    log_dir = os.path.expanduser(LOG_DIR)
    eval_log_dir = log_dir + "_eval"

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    envs = make_vec_envs(ENV_NAME, SEED, NUM_PROCESSES,
                         GAMMA, LOG_DIR, device, False)

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space)
    actor_critic.to(device)

    #wandb.watch([actor_critic.base.actor, actor_critic.base.critic], log='parameters')

    agent = Agent(
        actor_critic,
        CLIP_PARAM,
        PPO_EPOCH,
        NUM_MINI_BATCH,
        VALUE_LOSS_COEF,
        ENTROPY,
        lr=LR,
        eps=EPSILON,
        max_grad_norm=MAX_GRAD_NORM)

    rollouts = Memory(NUM_STEPS, NUM_PROCESSES,
                              envs.observation_space.shape, envs.action_space)

    obs = envs.reset()
    rollouts.states[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(
        ENV_STEPS) // NUM_STEPS // NUM_PROCESSES
    for j in range(num_updates):
        if OFFSCREEN and RENDER:
            if TUNING:
                fn = f'{ENV_NAME}-{LR}-{EPSILON}-{GAMMA}-{GAE_LAMBDA}-{NUM_MINI_BATCH}'
            else:
                fn = f'{ENV_NAME}'

            if j % RENDER_INTERVAL == 0:
                VideoWriter = cv2.VideoWriter(fn + str(j//RENDER_INTERVAL) + ".avi", fourcc, 50.0, (250, 250))
            elif j == num_updates - 1:
                VideoWriter = cv2.VideoWriter(fn + "final.avi", fourcc, 50.0, (250, 250))
        update_linear_schedule(
            agent.optimizer, j, num_updates, LR)

        for step in range(NUM_STEPS):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob = agent.actor_critic.act(
                    rollouts.states[step], rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)
            if RENDER and (j % RENDER_INTERVAL == 0 and len(episode_rewards) > 1) or j == num_updates - 1:
                if OFFSCREEN:
                    I = envs.render(mode='rgb_array')
                    I = cv2.cvtColor(I, cv2.COLOR_RGB2BGR)
                    I = cv2.resize(I, (250, 250))
                    VideoWriter.write(I)
                else:
                    envs.render()

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(
                [[0.0] if done_ else [1.0] for done_ in done])
            bad_masks = torch.FloatTensor(
                [[0.0] if 'bad_transition' in info.keys() else [1.0]
                 for info in infos])
            rollouts.insert(obs, action,
                            action_log_prob, value, reward, masks, bad_masks)
        if OFFSCREEN and RENDER:
            VideoWriter.release()

        with torch.no_grad():
            next_value = agent.actor_critic.get_value(
                rollouts.states[-1], rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, GAMMA,
                                 GAE_LAMBDA)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        if j % LOG_INTERVAL == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * NUM_PROCESSES * NUM_STEPS
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss))
        if LOG_WANDB:
            wandb.log({
                "median": np.median(episode_rewards), 
                "min": np.min(episode_rewards),
                "max": np.max(episode_rewards),
                "action_loss": action_loss,
                "value_loss": value_loss
            })

    if LOG_WANDB and TUNING:
        run.finish()    
    
if __name__ == "__main__":
    if TUNING:
        for LR in lr_:
            for EPSILON in epsilon_:
                for GAMMA in gamma_:
                    for GAE_LAMBDA in gae_lambda_:
                        for NUM_MINI_BATCH in num_mini_batch_:
                            main()

    else:
        main()
