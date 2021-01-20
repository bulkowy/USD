import numpy as np
import torch
from ddpg import DDPG
import gym

env = gym.make('Reacher-v2')
input_d = env.observation_space.shape[0]
action_d = env.action_space.shape[0]
num_agents = 1

def saveModel(agent, path):
    state_dicts = {'m_c': agent.l_critic.state_dict(),
                   'm_a': agent.l_actor.state_dict()}

m_dir = 'saved_models/'
m_name = 'reacher-v2_agent.pt'

agent = DDPG(input_d, action_d, num_agents)

def train(n_ep=1000, i_ep=1, eps_start=1.0, eps_end=0.05, eps_decay=0.99):
    scores_history = []
    eps = eps_start

    while i_ep < n_ep + 1:
        agent.reset()
        s = env.reset()
        epi_scores = np.zeros(num_agents)

        d = False
        while not d:
            if i_ep % 10 == 0:
                env.render()
            a = agent.act(s, 1.0)
            a = np.clip(a, -1, 1)
            ns, r, d, _ = env.step(a)
            epi_scores += r
            agent.step_(s, a, r, ns, d)

            s = ns

        if agent.is_training == True:
            scores_history.append(np.mean(epi_scores))
            print('Episode: {}  Avg score(#{}): {:.2f}  steps: {}  Actor G: {:.2f}  '
                  'Critic L: {:1.2e}  noise: {:.2f}  eps: {:.2f}'.format(i_ep, num_agents, 
                                                                         np.mean(epi_scores),
                                                                         agent.step, 
                                                                         agent.getQAvg(), 
                                                                         agent.getTDAvg(), 
                                                                         agent.getNoiseAvg(),
                                                                         eps))
            
            if i_ep % 100 == 0:
                print('\nAverage Score: {:.2f}'.format(np.mean(scores_history)))

                saveModel(agent, m_dir + m_name)

            i_ep += 1                                         # episode increment
            eps = max(eps*eps_decay,eps_end)                       # min of eps
        else:                                                      # training not yet started
            print('\rFetching experiences... {} '.format(len(agent.memory.memory)), end="")

    env.close()
        
    return scores_history

train()