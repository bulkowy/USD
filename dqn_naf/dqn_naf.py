import torch
import torch.nn as nn
from torch.optim import Adam
import gym
import numpy as np
from collections import deque
import random
import copy
from torch.distributions import MultivariateNormal
import os
import time
import wandb
import reacheredited

from mujoco_py import GlfwContext
import cv2


lr_ = [1e-3, 3e-4, 1e-4]
layer_d_ = [128, 256]
batch_size_ = [64, 128, 256, 512]

TUNING = True

# -- CONFIG
ENV_NAME = 'Reacher-v2'
# ENV_NAME = 'reachere-v2'
EPISODES = 2000
RENDER = False
RENDER_EVERY = 100
LOG_WANDB = True

# -- TEST
IS_TEST = False
RENDER_IN_TEST = True
LOAD_PATH = 'checkpoint/Reacher-v2/_ep_1800.pt'

# -- NN
LAYER_D = 256

# -- HYPERPARAMETERS
BATCH_SIZE = 256
LR = 1e-3
BUFFER_SIZE = 100000

sigma = 0.3

OFFSCREEN = False

if OFFSCREEN:
    GlfwContext(offscreen=True)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')


class NAFdqn(nn.Module):
    def __init__(self, input_d, output_d, layer_d=256):
        super(NAFdqn, self).__init__()
        self.input_d = input_d
        self.output_d = output_d
        self.layer_d = layer_d

        self.v1 = nn.Linear(self.input_d, self.layer_d)
        self.vN1 = nn.BatchNorm1d(self.layer_d)          # reLU   -> layer_d (128)
        self.v2 = nn.Linear(self.layer_d, self.layer_d)
        self.vN2 = nn.BatchNorm1d(self.layer_d)          # reLU   -> layer_d (128)
        self.v3 = nn.Linear(self.layer_d, 1)             # Linear -> 1

        self.mu1 = nn.Linear(self.input_d, self.layer_d)
        self.muN1 = nn.BatchNorm1d(self.layer_d)         # reLU   -> layer_d (128)
        self.mu2 = nn.Linear(self.layer_d, self.layer_d)
        self.muN2 = nn.BatchNorm1d(self.layer_d)         # reLU   -> layer_d (128)
        self.mu3 = nn.Linear(self.layer_d, self.output_d)# tanh   -> output_d (2)

        self.l1 = nn.Linear(self.input_d, self.layer_d * 2)
        self.lN1 = nn.BatchNorm1d(self.layer_d * 2) 
        self.l2 = nn.Linear(self.layer_d * 2, self.layer_d * 2)
        self.lN2 = nn.BatchNorm1d(self.layer_d * 2)          # reLU   -> 2*layer_d (256)
        self.l3 = nn.Linear(self.input_d, ((self.output_d + 1) * self.output_d)//2) # linear

    def forward(self, input_, action=None):
        V = torch.relu(self.v1(input_))
        V = self.vN1(V)
        V = torch.relu(self.v2(V))
        V = self.vN2(V)
        V = self.v3(V)

        MU = torch.relu(self.mu1(input_))
        MU = self.muN1(MU)
        MU = torch.relu(self.mu2(MU))
        MU = self.muN2(MU)
        MU = torch.tanh(self.mu3(MU))

        MU = MU.unsqueeze(-1)
        L = self.l3(input_)

        Lmatrix = torch.zeros((input_.shape[0], self.output_d, self.output_d))#.to(device)
        indices = torch.tril_indices(row=self.output_d, col=self.output_d, offset=0)
        Lmatrix[:, indices[0], indices[1]] = L
        Lmatrix.diagonal(dim1=1, dim2=2).exp_()
        P = Lmatrix*Lmatrix.transpose(2, 1)

        Q = None

        if action is not None:
            sub_action_MU = action.unsqueeze(-1) - MU
            mul_a_MU_P = torch.matmul(sub_action_MU.transpose(2, 1), P)
            A = torch.matmul(mul_a_MU_P, sub_action_MU) * (-1.0/2.0)
            A = A.squeeze(-1)
            Q = A + V

        if not IS_TEST:
            noise = np.random.normal(loc=0, scale=sigma, size=2)
            idx = np.random.randint(MU.shape[0])
            action = MU.cpu().squeeze().detach().numpy()[idx] + noise
            action = action.reshape((2,))
            action = torch.from_numpy(np.clip(action, -1.0, 1.0)).float()
        else:
            action = MU.cpu().squeeze().detach().numpy()
            action = torch.from_numpy(action.reshape((2,))).float()
        return action, Q, V        

class Agent:
    def __init__(self, state_d, action_d, layer_d=256, batch_size=256, buffer_size=100000, lr=1e-3, episodes=1000):
        self.state_d = state_d
        self.action_d = action_d
        self.layer_d = layer_d
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.lr = lr 
        self.update_every = 4
        self.episodes = episodes

        self.checkpoint_path = f"./checkpoint/{ENV_NAME}/"

        self.dqn_l = NAFdqn(state_d, action_d, layer_d)
        self.dqn_t = NAFdqn(state_d, action_d, layer_d)
        self.opt = Adam(self.dqn_l.parameters(), lr=lr)

        self.memory = deque(maxlen=buffer_size)

        self.step_ = 0

    def _getBatch(self):
        miniBatch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, new_states, dones = map(np.array, zip(*miniBatch))
        states = torch.from_numpy(states).float()
        actions = torch.from_numpy(actions).float()
        rewards = torch.from_numpy(rewards).float()
        new_states = torch.from_numpy(new_states).float()
        dones = torch.from_numpy(dones).float()
        return (states, actions, rewards, new_states, dones)

    def step(self, state, action, reward, next_state, done):
        #                   0      1       2       3           4
        self.memory.append([state, action, reward, next_state, done])

        self.step_ = (self.step_ + 1) % self.update_every
        if self.step_ == 0:
            if len(self.memory) > self.batch_size * 4:
                exps = self._getBatch()
                self.learn(exps)

    def soft_update(self):
        for t_param, l_param in zip(self.dqn_t.parameters(), self.dqn_l.parameters()):
            t_param.data.copy_(0.01*l_param.data + 0.99*t_param.data)

    def hard_update(self):
        self.dqn_t.load_state_dict(self.dqn_l.state_dict())

    def act(self, state):
        state = torch.from_numpy(state).float()

        self.dqn_l.eval()
        with torch.no_grad():
            action, _, _ = self.dqn_l(state.unsqueeze(0))
        self.dqn_l.train()

        return action.cpu().squeeze().numpy().reshape((self.action_d,))

    def save_params(self, n_episode):
        params = {
            "dqn_l": self.dqn_l.state_dict(),
            "dqn_t": self.dqn_t.state_dict(),
            "opt": self.opt.state_dict()
        }
        os.makedirs(self.checkpoint_path, exist_ok=True)

        path = os.path.join(self.checkpoint_path + "_ep_" + str(n_episode) + ".pt")
        torch.save(params, path)

    
    def load_params(self, path):
        if not os.path.exists(path):
            raise Exception(
                f"[ERROR] the input path does not exist. Wrong path: {path}"
            )

        params = torch.load(path)
        self.dqn_l.load_state_dict(params["dqn_l"])
        self.dqn_t.load_state_dict(params["dqn_t"])
        self.opt.load_state_dict(params["opt"])
        print("[INFO] loaded the model and optimizer from", path)
        

    def learn(self, experiences):
        self.opt.zero_grad()
        states, actions, rewards, next_states, dones = experiences

        # get the Value for the next state from target model
        with torch.no_grad():
            _, _, V_ = self.dqn_l(next_states)

        # Compute Q targets for current states 
        V_targets = rewards.unsqueeze(-1) + (0.99 * V_ ) #* (1 - dones)
        
        # Get expected Q values from local model
        _, Q, _ = self.dqn_l(states, actions)


        # Compute loss
        loss = torch.nn.functional.l1_loss(Q, V_targets)
        
        # Minimize the loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.dqn_l.parameters(),1.)
        self.opt.step()

        self.soft_update()
        return loss.detach().cpu().numpy()

    def test(self, env):
        score_list = []
        for i_episode in range(self.episodes):
            state = env.reset()
            done = False
            score = 0
            step = 0

            while not done:
                if RENDER_IN_TEST:
                    env.render()

                action = self.act(state)
                next_state, reward, done, _ = env.step(action)

                state = next_state
                score += reward
                step += 1

            print("[INFO] test %d\tstep: %d\ttotal score: %d" % (i_episode, step, score))
            score_list.append(score)

                

def main():
    global sigma
    env = gym.make(ENV_NAME)
    action_d = env.action_space.shape[0]
    state_d = env.observation_space.shape[0]
    scores = deque(maxlen=100)
    red_sigma = (2*sigma)/EPISODES


    if LOG_WANDB:
        if TUNING:
            run = wandb.init(project=f'{ENV_NAME}_DQN-HP_SEARCH',
                    name=f'DQN/{ENV_NAME}-{LR}-{LAYER_D}-{BATCH_SIZE}', reinit=True)
        else:
            wandb.init(project=f'{ENV_NAME}_DQN',
                    name=f'DQN/{ENV_NAME}')

    agent = Agent(
        state_d=state_d,
        action_d=action_d,
        layer_d=LAYER_D,
        batch_size=BATCH_SIZE, 
        buffer_size=BUFFER_SIZE, 
        lr=LR, 
        episodes=EPISODES
    )
    wandb.watch([agent.dqn_l], log='parameters')
    if IS_TEST:
        agent.load_params(LOAD_PATH)
        agent.test(env)
    
    else:
        if OFFSCREEN and RENDER:
            VideoWriter = cv2.VideoWriter(ENV_NAME + ".avi", fourcc, 50.0, (250, 250))

        for ep in range(1, EPISODES+1):
            if OFFSCREEN and RENDER:
                if TUNING:
                    fn = f'{ENV_NAME}-{LR}-{LAYER_D}-{BATCH_SIZE}'
                else:
                    fn = f'{ENV_NAME}'
                if ep > 100 and ep % RENDER_EVERY == 0:
                    VideoWriter = cv2.VideoWriter(fn + str(ep//20) + ".avi", fourcc, 50.0, (250, 250))
                elif ep == EPISODES+1:
                    VideoWriter = cv2.VideoWriter(fn + "final.avi", fourcc, 50.0, (250, 250))
        
            sum_reward = 0
            done = False
            state = env.reset()

            sigma -= red_sigma
            if sigma < 0:
                sigma = 0

            while not done:
                
                action = agent.act(state)
                next_state, reward, done, _ = env.step(action)
                agent.step(state, action, reward, next_state, done)

                state = next_state
                sum_reward += reward

                if RENDER and ((ep > 100 and ep % RENDER_EVERY == 0) or ep == EPISODES):
                    if OFFSCREEN:
                        I = env.render(mode='rgb_array')
                        I = cv2.cvtColor(I, cv2.COLOR_RGB2BGR)
                        I = cv2.resize(I, (250, 250))
                        VideoWriter.write(I)
                    else:
                        env.render()

                if done:
                    scores.append(sum_reward)
                    print(
                        "[INFO] episode %d, total score: %d\n"
                        % (
                            ep,
                            sum_reward,
                        )  # actor loss  # critic loss
                    )
                    if LOG_WANDB:
                        wandb.log(
                            {
                                "score": sum_reward,
                            }
                        )
            
            if ep % 100 == 0 or ep == EPISODES:
                agent.save_params(ep)
            
            if OFFSCREEN and RENDER:
                VideoWriter.release()
    
    if LOG_WANDB and TUNING:
        run.finish()



if __name__=='__main__':
    if TUNING:
        for LR in lr_:
            for LAYER_D in layer_d_:
                for BATCH_SIZE in batch_size_:
                    main()
    else:
        main()