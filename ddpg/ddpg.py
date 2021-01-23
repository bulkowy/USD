import gym
import numpy as np
import torch
import wandb
import time

from ac import Learner
from noise import OUNoise
from memory import ReplayBuffer

from mujoco_py import GlfwContext
import cv2

OFFSCREEN = False

if OFFSCREEN:
    GlfwContext(offscreen=True)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')

def numpy2floattensor(arrays, device_):
    """Convert numpy type to torch FloatTensor.
        - Convert numpy array to torch float tensor.
        - Convert numpy array with Tuple type to torch FloatTensor with Tuple.
    """

    if isinstance(arrays, tuple):  # check Tuple or not
        tensors = []
        for array in arrays:
            tensor = torch.from_numpy(array).to(device_, non_blocking=True).float()
            tensors.append(tensor)
        return tuple(tensors)
    tensor = torch.from_numpy(arrays).to(device_, non_blocking=True).float()
    return tensor

class DDPG:
    def __init__(
            self,
            env,
            env_name,
            episode_num,
            interim_test_num,
            load_from=None,
            is_test=False,
            render=True,
            log=True,
            hidden_dims_a=[256, 256], 
            hidden_dims_c=[256, 256], 
            gradient_clip_a=0.5,
            gradient_clip_c=1.0,
            lr=1e-3,
            weight_decay=1e-6,
            gamma=0.99,
            tau=0.01,
            buffer_size=100000,
            batch_size=128,
            initial_random_action=10000,
            noise_sigma=0.0,
            noise_theta=0.0,
            tuning=False):

        self.env = env
        self.env_name = env_name

        self.episode_num = episode_num
        self.interim_test_num = interim_test_num

        self.curr_state = np.zeros((1,))
        self.total_step = 0
        self.episode_step = 0
        self.i_episode = 0

        self.load_from = load_from
        self.is_test = is_test

        self.render = render
        self.log = log
        self.tuning = tuning

        self.hidden_dims_a = hidden_dims_a
        self.hidden_dims_c = hidden_dims_c
        self.gradient_clip_a = gradient_clip_a
        self.gradient_clip_c = gradient_clip_c


        self.lr = lr
        self.weight_decay = weight_decay
        self.gamma = gamma
        self.tau = tau 
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.initial_random_action = initial_random_action
        self.noise_sigma = noise_sigma
        self.noise_theta = noise_theta

        self.noise = OUNoise(
            env.action_space.shape[0],
            theta=self.noise_theta,
            sigma=self.noise_sigma
        )

        self._initialize()

    def _initialize(self):
        if not self.is_test:
            self.memory = ReplayBuffer(
                self.buffer_size, self.batch_size
            )
        
        self.learner = Learner(
            self.env,
            self.env_name,
            self.hidden_dims_a,
            self.hidden_dims_c,
            self.gradient_clip_a,
            self.gradient_clip_c,
            self.lr,
            self.weight_decay,
            self.gamma,
            self.tau,
            self.is_test,
            self.load_from
        )

    def select_action(self, state):
        self.curr_state = state
        state = self._preprocess_state(state)
        if (self.total_step < self.initial_random_action and not self.is_test):
            return np.array(self.env.action_space.sample())
        
        with torch.no_grad():
            selected_action = self.learner.actor(state).detach().cpu().numpy()

        if not self.is_test:
            noise = self.noise.sample()
            selected_action = np.clip(selected_action + noise, -1.0, 1.0)

        return selected_action

    def _preprocess_state(self, state):
        state = numpy2floattensor(state, self.learner.device)
        return state

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)

        if not self.is_test:
            transition = (self.curr_state, action, reward, next_state, done)
            self._add_transition_to_memory(transition)

        return next_state, reward, done, info

    def _add_transition_to_memory(self, transition):
        self.memory.add(transition)

    def write_log(self, log_value: tuple):
        """Write log about loss and score"""
        i, loss, score, avg_time_cost = log_value
        total_loss = loss.sum()

        print(
            "[INFO] episode %d, episode step: %d, total step: %d, total score: %d\n"
            "total loss: %f actor_loss: %.3f critic_loss: %.3f (spent %.6f sec/step)\n"
            % (
                i,
                self.episode_step,
                self.total_step,
                score,
                total_loss,
                loss[0],
                loss[1],
                avg_time_cost,
            )  # actor loss  # critic loss
        )

        if self.log:
            log_value = {
                    "score": score,
                    "total loss": total_loss,
                    "actor loss": loss[0],
                    "critic loss": loss[1],
                    "time per each step": avg_time_cost,
                }
            wandb.log(log_value)

    def train(self):
        """Train the agent."""
        # logger
        if self.log:
            self.set_wandb()
            wandb.watch([self.learner.actor, self.learner.critic], log="parameters")
        
        if OFFSCREEN:
            VideoWriter = cv2.VideoWriter(self.env_name + ".avi", fourcc, 50.0, (250, 250))

        

        for self.i_episode in range(1, self.episode_num + 1):
            if OFFSCREEN and self.render:
                if self.tuning:
                    fn = f'{self.env_name}-{self.lr}-{self.weight_decay}-{self.gamma}-{self.batch_size}'
                else:
                    fn = f'{self.env_name}'
                if self.i_episode % 100 == 0:
                    VideoWriter = cv2.VideoWriter(fn + str(self.i_episode//20) + ".avi", fourcc, 50.0, (250, 250))
                elif self.i_episode == self.episode_num:
                    VideoWriter = cv2.VideoWriter(fn + "final.avi", fourcc, 50.0, (250, 250))
        
            state = self.env.reset()
            done = False
            score = 0
            self.episode_step = 0
            losses = list()

            t_begin = time.time()

            while not done:
                if self.render and (self.i_episode % 100 == 0 or self.i_episode == self.episode_num):
                    if OFFSCREEN:
                        I = self.env.render(mode='rgb_array')
                        I = cv2.cvtColor(I, cv2.COLOR_RGB2BGR)
                        I = cv2.resize(I, (250, 250))
                        VideoWriter.write(I)
                    else:
                        self.env.render()

                action = self.select_action(state)
                next_state, reward, done, _ = self.step(action)
                self.total_step += 1
                self.episode_step += 1

                if len(self.memory) >= self.batch_size:
                    experience = self.memory.sample()
                    experience = numpy2floattensor(experience, self.learner.device)
                    loss = self.learner.update_model(experience)
                    losses.append(loss)  # for logging

                state = next_state
                score += reward

            if self.render and OFFSCREEN:
                VideoWriter.release()

            t_end = time.time()
            avg_time_cost = (t_end - t_begin) / self.episode_step

            # logging
            if losses:
                avg_loss = np.vstack(losses).mean(axis=0)
                log_value = (self.i_episode, avg_loss, score, avg_time_cost)
                self.write_log(log_value)
                losses.clear()

            if self.i_episode % 200 == 0:
                self.learner.save_params(self.i_episode)
                self.interim_test()

        if self.tuning and self.log:
            self.run.finish()

        # termination
        self.env.close()
        self.learner.save_params(self.i_episode)
        self.interim_test()

    def set_wandb(self):
        if self.tuning:
            self.run = wandb.init(
                project=f'{self.env_name}_DDPG-HP_SEARCH',
                name=f'DDPG/{self.env_name}-{self.lr}-{self.weight_decay}-{self.gamma}-{self.batch_size}', reinit=True
            )
        else:
            wandb.init(project=f'{self.env_name}_DDPG',
                name=f'DDPG/{self.env_name}')
    
    def interim_test(self):
        self.is_test = True

        print()
        print("===========")
        print("Start Test!")
        print("===========")

        self._test(interim_test=True)

        print("===========")
        print("Test done!")
        print("===========")
        print()

        self.is_test = False

    def test(self):
        """Test the agent."""
        # logger
        if self.log:
            self.set_wandb()

        self._test()

        # termination
        self.env.close()

    def _test(self, interim_test=False):
        if interim_test:
            test_num = self.interim_test_num
        else:
            test_num = self.episode_num

        score_list = []

        for i_episode in range(test_num):
            state = self.env.reset()
            done = False
            score = 0
            step = 0

            while not done:
                if self.render:
                    self.env.render()

                action = self.select_action(state)
                next_state, reward, done, _ = self.step(action)

                state = next_state
                score += reward
                step += 1

            print(
                "[INFO] test %d\tstep: %d\ttotal score: %d" % (i_episode, step, score)
            )
            score_list.append(score)

