import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as U
from torch.optim import Adam
import numpy as np
from collections import namedtuple, deque
from ounoise import OUnoise
from ac import Actor, Critic

# config

BUFFER_SIZE         = 1000000
BATCH_SIZE          = 256
REPLAY_MIN_SIZE     = 10000
GAMMA               = 0.95
TAU                 = 0.001
LR                  = 0.0001
UPDATE_LOOP         = 20
UPDATE_EVERY        = 4
GRAD_CLIP_MAX       = 1.0
ADD_NOISE           = True

# end of config

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size, action_space,
                 reward_scale=False, seed=0):

        """Data Structure to store experience object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            reward_scale (flag): to scale reward down by 10
            seed (int): random seed
        """
        self.buffer_size = buffer_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.action_space = action_space
        self.reward_scale = reward_scale

        self.experience = namedtuple("Experience", field_names=["state", "action",
                                     "reward", "next_state", "done"])


    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory.
        state: (torch) shape: 1,state_space
        action: (torch) shape: 1,action_space
        reward: float
        next_state: (torch) shape: 1,state_space
        done: bool
        """

        e = self.experience(state, action, reward, next_state, done)

        self.memory.append(e)


    def sample(self):
        """Sample a batch of experiences from memory."""
        # get sample of index from the p distribution
        sample_ind = np.random.choice(len(self.memory), self.batch_size)

        # get the selected experiences: avoid using mid list indexing
        es, ea, er, en, ed = [], [], [], [], []

        i = 0
        while i < len(sample_ind): #while loop is faster
            self.memory.rotate(-sample_ind[i])
            e = self.memory[0]
            es.append(e.state)
            ea.append(e.action)
            er.append(e.reward)
            en.append(e.next_state)
            ed.append(e.done)
            self.memory.rotate(sample_ind[i])
            i += 1

        states = torch.stack(es).squeeze().float().to(device)
        actions = torch.stack(ea).float().to(device)
        rewards = torch.from_numpy(np.vstack(er)).float().to(device)
        next_states = torch.stack(en).squeeze().float().to(device)
        dones = torch.from_numpy(np.vstack(ed).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

class DDPG:
    def __init__(self, input_d, action_d, num_agents=20):
        self.input_d = input_d
        self.action_d = action_d
        self.num_agents = num_agents
        
        self.add_noise = ADD_NOISE

        self.l_critic = Critic(input_d, action_d).to(device)
        self.t_critic = Critic(input_d, action_d).to(device)
        self.critic_opt = Adam(self.l_critic.parameters(), lr=LR)

        self.l_actor = Actor(input_d, action_d).to(device)
        self.t_actor = Actor(input_d, action_d).to(device)
        self.actor_opt = Adam(self.l_actor.parameters(), lr=LR)

        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE, action_d)
        self.noise = OUnoise((num_agents, action_d), 123)

        self.step = 0
        self.is_training = False

        self.Q_history = deque(maxlen=1000)
        self.td_history = deque(maxlen=1000)
        self.noise_history = deque(maxlen=1000)

    def _toTorch(self, s):
        return torch.from_numpy(s).float().to(device)

    def step_(self, s, a, r, ns, d):
        if self.num_agents > 1:
            for i in range(0, self.num_agents):
                self.memory.add(self._toTorch(s[i,:]),
                                self._toTorch(a[i,:]), r[i],
                                self._toTorch(ns[i,:]), d[i])
        else:
            self.memory.add(self._toTorch(s),
                            self._toTorch(a), r,
                            self._toTorch(ns), d)

        self.step += 1

        if len(self.memory) >= REPLAY_MIN_SIZE * self.num_agents:
            if not self.is_training:
                print("Prefetch completed. Training starts\n")
                print(f"Num of agents:{self.num_agents}")
                print(f"Device: {device}")
                self.is_training = True

            for i in range(self.num_agents):
                exps = self.memory.sample()
                self._learn(exps, GAMMA)

            if self.step % UPDATE_EVERY == 0:
                self._soft_update(self.l_critic, self.t_critic, TAU)
                self._soft_update(self.l_actor, self.t_actor, TAU)

    def act(self, state, eps=0.99):
        self.l_actor.eval()
        with torch.no_grad():
            actions = self.l_actor(self._toTorch(state).unsqueeze(0)).cpu().numpy()
        self.l_actor.train()

        if self.add_noise and np.random.rand() < eps:
            noise = self.noise.sample()
            actions += noise
            self.noise_history.append(np.mean(np.abs(noise)))

        return np.clip(actions.squeeze(), -1, 1)

    def _learn(self, exps, gamma):
        s, a, r, ns, d = exps

        # new state Q
        ns_Q = self.t_critic(ns, self.t_actor(ns))

        # target Q
        t_Q = r + (1-d) * gamma * ns_Q.detach()
        #assert(t_Q.requires_grad == False)

        # current Q
        c_Q = self.l_critic(s, a)
        #assert(c_Q.requires_grad == False)

        c_loss = F.mse_loss(c_Q, t_Q)

        self.critic_opt.zero_grad()
        c_loss.backward()
        U.clip_grad_norm_(self.l_critic.parameters(), GRAD_CLIP_MAX)
        self.critic_opt.step()

        self.td_history.append(c_loss.detach())

        a_loss = -self.l_critic(s, self.l_actor(s)).mean()
        self.actor_opt.zero_grad()
        a_loss.backward()
        U.clip_grad_norm_(self.l_actor.parameters(), GRAD_CLIP_MAX)
        self.actor_opt.step()

        self.Q_history.append(-a_loss.detach())

    def _soft_update(self, l_model, t_model, tau):
        for t_p, l_p in zip(t_model.parameters(), l_model.parameters()):
            t_p.data.copy_(tau*l_p.data + (1.0-tau)*t_p.data)

    def getQAvg(self):
        return sum(self.Q_history)/len(self.Q_history)

    def getNoiseAvg(self):
        if self.add_noise: return sum(self.noise_history)/len(self.noise_history)

    def getTDAvg(self):
        return sum(self.td_history)/len(self.td_history)

    def reset(self):
        self.noise.reset()

