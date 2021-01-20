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
from args import get_args

from mujoco_py import GlfwContext
import cv2
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
    args = get_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.cuda and torch.cuda.is_available() and args.cuda_deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    log_dir = os.path.expanduser(args.log_dir)
    eval_log_dir = log_dir + "_eval"

    torch.set_num_threads(1)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    envs = make_vec_envs(args.env_name, args.seed, args.num_processes,
                         args.gamma, args.log_dir, device, False)

    actor_critic = Policy(
        envs.observation_space.shape,
        envs.action_space)
    actor_critic.to(device)

    agent = Agent(
        actor_critic,
        args.clip_param,
        args.ppo_epoch,
        args.num_mini_batch,
        args.value_loss_coef,
        args.entropy_coef,
        lr=args.lr,
        eps=args.eps,
        max_grad_norm=args.max_grad_norm)

    rollouts = Memory(args.num_steps, args.num_processes,
                              envs.observation_space.shape, envs.action_space)

    obs = envs.reset()
    rollouts.states[0].copy_(obs)
    rollouts.to(device)

    episode_rewards = deque(maxlen=10)

    start = time.time()
    num_updates = int(
        args.num_env_steps) // args.num_steps // args.num_processes
    for j in range(num_updates):
        if j % args.render_interval == 0:
            VideoWriter = cv2.VideoWriter(args.env_name + str(j//args.render_interval) + ".avi", fourcc, 50.0, (250, 250))
        elif j == num_updates - 1:
            VideoWriter = cv2.VideoWriter(args.env_name + "final.avi", fourcc, 50.0, (250, 250))
        update_linear_schedule(
            agent.optimizer, j, num_updates, args.lr)

        for step in range(args.num_steps):
            # Sample actions
            with torch.no_grad():
                value, action, action_log_prob = agent.actor_critic.act(
                    rollouts.states[step], rollouts.masks[step])

            # Obser reward and next obs
            obs, reward, done, infos = envs.step(action)
            if (j % args.render_interval == 0 and len(episode_rewards) > 1) or j == num_updates - 1:
                
                I = envs.render(mode='rgb_array')
                I = cv2.cvtColor(I, cv2.COLOR_RGB2BGR)
                I = cv2.resize(I, (250, 250))
                VideoWriter.write(I)

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
        VideoWriter.release()

        with torch.no_grad():
            next_value = agent.actor_critic.get_value(
                rollouts.states[-1], rollouts.masks[-1]).detach()

        rollouts.compute_returns(next_value, args.gamma,
                                 args.gae_lambda)

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

        rollouts.after_update()

        # save for every interval-th episode or for the last epoch
        if (j % args.save_interval == 0
                or j == num_updates - 1) and args.save_dir != "":
            save_path = os.path.join(args.save_dir, 'PPO')
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            #torch.save([
            #    actor_critic,
            #    getattr(get_vec_normalize(envs), 'ob_rms', None)
            #], os.path.join(save_path, args.env_name + ".pt"))

            #torch.save(agent.actor_critic.state_dict(), 
            #os.path.join(save_path, args.env_name + ".pt"))

            #torch.save(getattr(get_vec_normalize(envs), 'ob_rms', None),
            #os.path.join(save_path, args.env_name + ".vec"))
            #print('SAVED!')

        if j % args.log_interval == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            end = time.time()
            print(
                "Updates {}, num timesteps {}, FPS {} \n Last {} training episodes: mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}\n"
                .format(j, total_num_steps,
                        int(total_num_steps / (end - start)),
                        len(episode_rewards), np.mean(episode_rewards),
                        np.median(episode_rewards), np.min(episode_rewards),
                        np.max(episode_rewards), dist_entropy, value_loss,
                        action_loss))

if __name__ == "__main__":
    main()
