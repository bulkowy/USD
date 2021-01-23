# DDPG

## How to run
`python main.py`

## How to tune parameters in Algorithm
First lines of `main.py` file contains config for parameters.  
If any change of parameters is required, it has to be done in `main.py` file, before running algorithm.

## What parameters are able to be tuned

`ENV_NAME` - Name (ID) of environment to train algo in (tested for `Reacher-v2` and `reachere-v2`)  
`EPISODE_NUM` - Number of episodes  
`INTERIM_TEST_NUM` - Number of episodes for interim test  
`IS_TEST` - Load previously trained model (from `LOAD_FROM` path) to test it  
`LOAD_FROM` - Path to previously trained model to test  
`RENDER` - Flag stating render mode (`True` - render env, `False` - don't render env)  
`LOG` - Flag stating logging to WANDB (`True` - log to WANDB, `False` - don't log to WANDB)  
`HIDDEN_DIMS_ACTOR` - sizes of hidden layers of actor (List of sizes, default `[256, 256]`)  
`HIDDEN_DIMS_CRITIC` - sizes of hidden layers of critic (List of sizes, default `[256, 256]`)  
`GRADIENT_CLIP_ACTOR` - value of gradient clipping for actor  
`GRADIENT_CLIP_CRITIC` = value of gradient clipping for critic  
`LR` - Learning rate  
`WEIGHT_DECAY` - Optimizer weight decay  
`GAMMA`  
`TAU`  
`BUFFER_SIZE`  
`BATCH_SIZE`  
`INITIAL_RANDOM_ACTION` - number of initial random sampled steps for populating memory  
`NOISE_SIGMA` - sigma of OUNoise  
`NOISE_THETA` - theta of OUNoise  