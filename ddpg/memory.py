from collections import deque
from typing import Any, Deque, List, Tuple

import numpy as np
import torch

def get_n_step_info(n_step_buffer, gamma):
    """Return n step reward, next state, and done."""
    # info of the last transition
    reward, next_state, done = n_step_buffer[-1][-3:]

    for transition in reversed(list(n_step_buffer)[:-1]):
        r, n_s, d = transition[-3:]

        reward = r + gamma * reward * (1 - d)
        next_state, done = (n_s, d) if d else (next_state, done)

    return reward, next_state, done

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples.
    Attributes:
        obs_buf (np.ndarray): observations
        acts_buf (np.ndarray): actions
        rews_buf (np.ndarray): rewards
        next_obs_buf (np.ndarray): next observations
        done_buf (np.ndarray): dones
        n_step_buffer (deque): recent n transitions
        n_step (int): step size for n-step transition
        gamma (float): discount factor
        max_len (int): size of buffers
        batch_size (int): batch size for training
        demo_size (int): size of demo transitions
        length (int): amount of memory filled
        idx (int): memory index to add the next incoming transition
    """

    def __init__(
        self,
        max_len: int,
        batch_size: int,
        gamma: float = 0.99,
        n_step: int = 1
    ):
        """Initialize a ReplayBuffer object.
        Args:
            max_len (int): size of replay buffer for experience
            batch_size (int): size of a batched sampled from replay buffer for training
            gamma (float): discount factor
            n_step (int): step size for n-step transition
        """
        assert 0 < batch_size <= max_len
        assert 0.0 <= gamma <= 1.0
        assert 1 <= n_step <= max_len

        self.obs_buf = None
        self.acts_buf = None
        self.rews_buf = None
        self.next_obs_buf = None
        self.done_buf = None

        self.n_step_buffer: Deque = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma

        self.max_len = max_len
        self.batch_size = batch_size
        self.length = 0
        self.idx = 0

    def add(self, transition):
        """Add a new experience to memory.
        If the buffer is empty, it is respectively initialized by size of arguments.
        """
        assert len(transition) == 5, "Inappropriate transition size"
        assert isinstance(transition[0], np.ndarray)
        assert isinstance(transition[1], np.ndarray)

        self.n_step_buffer.append(transition)

        # single step transition is not ready
        if len(self.n_step_buffer) < self.n_step:
            return ()

        if self.length == 0:
            state, action = transition[:2]
            self._initialize_buffers(state, action)

        # add a multi step transition
        reward, next_state, done = get_n_step_info(self.n_step_buffer, self.gamma)
        curr_state, action = self.n_step_buffer[0][:2]

        self.obs_buf[self.idx] = curr_state
        self.acts_buf[self.idx] = action
        self.rews_buf[self.idx] = reward
        self.next_obs_buf[self.idx] = next_state
        self.done_buf[self.idx] = done

        self.idx += 1
        self.length = min(self.length + 1, self.max_len)

        # return a single step transition to insert to replay buffer
        return self.n_step_buffer[0]

    def extend(self, transitions):
        """Add experiences to memory."""
        for transition in transitions:
            self.add(transition)

    def sample(self, indices: List[int] = None) -> Tuple[np.ndarray, ...]:
        """Randomly sample a batch of experiences from memory."""
        assert len(self) >= self.batch_size

        if indices is None:
            indices = np.random.choice(len(self), size=self.batch_size, replace=False)

        states = self.obs_buf[indices]
        actions = self.acts_buf[indices]
        rewards = self.rews_buf[indices].reshape(-1, 1)
        next_states = self.next_obs_buf[indices]
        dones = self.done_buf[indices].reshape(-1, 1)

        return states, actions, rewards, next_states, dones

    def _initialize_buffers(self, state: np.ndarray, action: np.ndarray) -> None:
        """Initialze buffers for state, action, resward, next_state, done."""
        # In case action of demo is not np.ndarray
        self.obs_buf = np.zeros([self.max_len] + list(state.shape), dtype=state.dtype)
        self.acts_buf = np.zeros(
            [self.max_len] + list(action.shape), dtype=action.dtype
        )
        self.rews_buf = np.zeros([self.max_len], dtype=float)
        self.next_obs_buf = np.zeros(
            [self.max_len] + list(state.shape), dtype=state.dtype
        )
        self.done_buf = np.zeros([self.max_len], dtype=float)

    def __len__(self) -> int:
        """Return the current size of internal memory."""
        return self.length