from gym.envs.registration import register

register(
    id='reachere-v2',
    entry_point='reacheredited.envs:ReacherEditedEnv',
    max_episode_steps=50,
    reward_threshold=-3.75,
)