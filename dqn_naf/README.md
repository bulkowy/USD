# PPO

## How to run
`python dqn_naf.py`

## How to tune parameters in Algorithm
First lines of `dqn_naf.py` file contains config for parameters.  
If any change of parameters is required, it has to be done in `dqn_naf.py` file, before running algorithm.

## What parameters are able to be tuned

`ENV_NAME` - Name (ID) of environment to train algo in (tested for `Reacher-v2` and `reachere-v2`)  
`EPISODES` - Number of episodes  
`RENDER` - Flag stating render mode (`True` - render env, `False` - don't render env)  
`RENDER_EVERY` - Value stating interval in episodes between rendering environment  
`LOG_WANDB` - Flag stating logging to WANDB (`True` - log to WANDB, `False` - don't log to WANDB)  
`IS_TEST` - Load previously trained model (from `LOAD_FROM` path) to test it  
`RENDER_IN_TEST` - Render during test  
`LOAD_PATH` - Path to previously trained model to test  
`LAYER_D` - Size for hidden layers in DQN (number)  
`BATCH_SIZE`   
`LR`   
`BUFFER_SIZE`  