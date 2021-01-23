# PPO

## How to run
`python main.py`

## How to tune parameters in Algorithm
First lines of `main.py` file contains config for parameters.  
If any change of parameters is required, it has to be done in `main.py` file, before running algorithm.

## What parameters are able to be tuned

`ENV_NAME` - Name (ID) of environment to train algo in (tested for `Reacher-v2` and `reachere-v2`)  
`ENV_STEPS` - Number of environment steps  
`NUM_STEPS` - Number of steps per epoch  
`NUM_MINI_BATCH`  - value of how many minibatches there are in epoch
`RENDER` - Flag stating render mode (`True` - render env, `False` - don't render env)  
`RENDER_INTERVAL` - Value stating interval in episodes between rendering environment  
`LOG_WANDB` - Flag stating logging to WANDB (`True` - log to WANDB, `False` - don't log to WANDB)  
`LOG_INTERVAL` - Value stating interval in episodes between logging  
`LR` - Learning rate  
`EPSILON`  
`GAMMA`  
`GAE_LAMBDA`  
`SEED`
`VALUE_LOSS_COEF` - scale for how value loss impacts update  
`MAX_GRAD_NORM` - value for stating clipping limit for AC parameters   
`PPO_EPOCH` - epochs of updating PPO  
`CLIP_PARAM` - clip parameter for policy    