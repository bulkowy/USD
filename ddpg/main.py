import gym
from envs import create_env
from ddpg import DDPG
import reacheredited

lr_ = [10**j for j in range(-6, -3+1)]
weight_decay_ = [10**j for j in range(-7, -5+1)]
gamma_ = [10**-2*j for j in range(99, 99+1)]
batch_size_ = [2**j for j in range(5, 8+1)]

TUNING = True

# -- CONFIG
ENV_NAME = 'Reacher-v2'
#ENV_NAME = 'reachere-v2'
EPISODE_NUM = 100
INTERIM_TEST_NUM = 10

#    -- TEST
IS_TEST = False
LOAD_FROM = "checkpoint/Reacher-v2/_ep_2000.pt"

#    -- LOG
RENDER = True
LOG = True

#    -- NN DIMS
HIDDEN_DIMS_ACTOR = [256, 256]
HIDDEN_DIMS_CRITIC = [256, 256]

#    -- HYPERPARAMETERS
GRADIENT_CLIP_ACTOR = 0.5
GRADIENT_CLIP_CRITIC = 1.0
LR = 1e-3
WEIGHT_DECAY = 1e-6
GAMMA = 0.99
TAU = 0.01
BUFFER_SIZE = 100000
BATCH_SIZE = 128
INITIAL_RANDOM_ACTION = 10000
NOISE_SIGMA = 0.0
NOISE_THETA = 0.0


def main():
    env_name = ENV_NAME
    env = create_env(env_name)

    agent = DDPG(
        env=env,
        env_name=env_name,
        episode_num=EPISODE_NUM,
        interim_test_num=INTERIM_TEST_NUM,
        load_from=LOAD_FROM,
        is_test=IS_TEST,
        render=RENDER,
        log=LOG,
        hidden_dims_a=HIDDEN_DIMS_ACTOR, 
        hidden_dims_c=HIDDEN_DIMS_CRITIC, 
        gradient_clip_a=GRADIENT_CLIP_ACTOR,
        gradient_clip_c=GRADIENT_CLIP_CRITIC,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        gamma=GAMMA,
        tau=TAU,
        buffer_size=BUFFER_SIZE,
        batch_size=BATCH_SIZE,
        initial_random_action=INITIAL_RANDOM_ACTION,
        noise_sigma=NOISE_SIGMA,
        noise_theta=NOISE_THETA,
        tuning=TUNING
    )

    if IS_TEST:
        agent.test()
    else:
        agent.train()

if __name__ == "__main__":
    if TUNING:
        for LR in lr_:
            for WEIGHT_DECAY in weight_decay_:
                for GAMMA in gamma_:
                    for BATCH_SIZE in batch_size_:
                        main()
    else:
        main()
