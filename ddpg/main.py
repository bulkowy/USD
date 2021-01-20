import gym
from envs import create_env
from ddpg import DDPG

# -- CONFIG
EPISODE_NUM=20000
INTERIM_TEST_NUM=10

#    -- TEST
IS_TEST = True
LOAD_FROM = "checkpoint/Reacher-v2/_ep_2000.pt"

#    -- LOG
RENDER = True
LOG = False

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
    env_name = "Reacher-v2"
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
        noise_theta=NOISE_THETA
    )

    if IS_TEST:
        agent.test()
    else:
        agent.train()

if __name__ == "__main__":
    main()
