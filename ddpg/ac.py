import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam

import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def init_layer_uniform_(layer, init_weight=3e-3):
    layer.weight.data.uniform_(-init_weight, init_weight)
    layer.bias.data.uniform_(-init_weight, init_weight)

    return layer

# actor MLP - state / 256 / 256 / action
# critic - state+action / 256 / 256 / 1

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256]):
        super(Actor, self).__init__()

        self.hidden_dims = hidden_dims
        self.input_dim = state_dim
        self.output_dim = action_dim 
        
        self.hidden_layers = []
        start_dim = self.input_dim
        for i, next_size in enumerate(self.hidden_dims):
            fc = nn.Linear(start_dim, next_size)
            start_dim = next_size
            self.__setattr__(f"hidden_fc{i}", fc) # Needed for torch to see layers
            self.hidden_layers.append(fc)

        self.output_layer = nn.Linear(start_dim, self.output_dim)
        self.output_layer = init_layer_uniform_(self.output_layer)

    def forward(self, x): # x = state
        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))

        x = torch.tanh(self.output_layer(x))
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dims=[256, 256]):
        super(Critic, self).__init__()

        self.hidden_dims = hidden_dims
        self.input_dim = state_dim + action_dim
        self.output_dim = 1
        
        self.hidden_layers = []
        start_dim = self.input_dim
        for i, next_size in enumerate(self.hidden_dims):
            fc = nn.Linear(start_dim, next_size)
            start_dim = next_size
            self.__setattr__(f"hidden_fc{i}", fc) # Needed for torch to see layers
            self.hidden_layers.append(fc)

        self.output_layer = nn.Linear(start_dim, self.output_dim)
        self.output_layer = init_layer_uniform_(self.output_layer)

    def forward(self, x): # x = state + action
        for hidden_layer in self.hidden_layers:
            x = F.relu(hidden_layer(x))
        x = self.output_layer(x)
        return x

def soft_update(local, target, tau):
    for t_param, l_param in zip(target.parameters(), local.parameters()):
        t_param.data.copy_(tau * l_param.data + (1.0 - tau) * t_param.data)

class Learner:
    def __init__(
            self, 
            env, 
            env_name, 
            hidden_dims_a=[256, 256], 
            hidden_dims_c=[256, 256], 
            gradient_clip_a=0.5,
            gradient_clip_c=1.0,
            lr=1e-3,
            weight_decay=1e-6,
            gamma=0.99,
            tau=0.01,
            is_test=False,
            load_from=None):

        self.env = env
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.shape[0]
        self.hidden_dims_a = hidden_dims_a
        self.hidden_dims_c = hidden_dims_c
        self.gradient_clip_a = gradient_clip_a
        self.gradient_clip_c = gradient_clip_c
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.lr = lr
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.tau = tau
        self.load_from = load_from
        self.is_test = is_test

        if not is_test:
            self.checkpoint_path = f"./checkpoint/{env_name}/"
            os.makedirs(self.checkpoint_path, exist_ok=True)

        self._init_network()

    def _init_network(self):
        self.actor = Actor(self.state_dim, self.action_dim, self.hidden_dims_a).to(self.device)
        self.actor_target = Actor(self.state_dim, self.action_dim, self.hidden_dims_a).to(self.device)

        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(self.state_dim, self.action_dim, self.hidden_dims_c).to(self.device)
        self.critic_target = Critic(self.state_dim, self.action_dim, self.hidden_dims_c).to(self.device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optim = Adam(
            self.actor.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        self.critic_optim = Adam(
            self.critic.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )

        if self.load_from is not None and self.is_test:
            self.load_params(self.load_from)

    def update_model(self, experience):
        states, actions, rewards, next_states, dones = experience

        masks = 1 - dones
        next_actions = self.actor_target(next_states)
        next_values = self.critic_target(torch.cat((next_states, next_actions), dim=-1))
        curr_returns = rewards + self.gamma * next_values * masks
        curr_returns = curr_returns.to(self.device)

        values = self.critic(torch.cat((states, actions), dim=-1))
        critic_loss = F.mse_loss(values, curr_returns)
        self.critic_optim.zero_grad()
        critic_loss.backward()
        clip_grad_norm_(self.critic.parameters(), self.gradient_clip_c)
        self.critic_optim.step()

        actions = self.actor(states)
        actor_loss = -self.critic(torch.cat((states, actions), dim=-1)).mean()
        self.actor_optim.zero_grad()
        actor_loss.backward()
        clip_grad_norm_(self.actor.parameters(), self.gradient_clip_a)
        self.actor_optim.step()

        soft_update(self.actor, self.actor_target, self.tau)
        soft_update(self.critic, self.critic_target, self.tau)

        return actor_loss.item(), critic_loss.item()

    def save_params(self, n_episode):
        params = {
            "actor_state_dict": self.actor.state_dict(),
            "actor_target_state_dict": self.actor_target.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "critic_target_state_dict": self.critic_target.state_dict(),
            "actor_optim_state_dict": self.actor_optim.state_dict(),
            "critic_optim_state_dict": self.critic_optim.state_dict(),
        }
        os.makedirs(self.checkpoint_path, exist_ok=True)

        path = os.path.join(self.checkpoint_path + "_ep_" + str(n_episode) + ".pt")
        torch.save(params, path)

        print(f"[INFO] Saved the model and optimizer to {path} \n")

    def load_params(self, path):
        if not os.path.exists(path):
            raise Exception(
                f"[ERROR] the input path does not exist. Wrong path: {path}"
            )

        params = torch.load(path)
        self.actor.load_state_dict(params["actor_state_dict"])
        self.actor_target.load_state_dict(params["actor_target_state_dict"])
        self.critic.load_state_dict(params["critic_state_dict"])
        self.critic_target.load_state_dict(params["critic_target_state_dict"])
        self.actor_optim.load_state_dict(params["actor_optim_state_dict"])
        self.critic_optim.load_state_dict(params["critic_optim_state_dict"])
        print("[INFO] loaded the model and optimizer from", path)

    def get_state_dict(self):
        return (self.critic_target.state_dict(), self.actor.state_dict())

    def get_policy(self):
        return self.actor



