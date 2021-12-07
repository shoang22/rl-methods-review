# RL-methods-review
Review of Fundamental RL Algorithms for Game Control

## Running instruction:
DQN: Open the DQN.py to modify the environment name on line 159, then run it in terminal: "python DQN.py".

DDPG: Open the DDPG.py to modify the environment name on line 134, then run it in terminal: "python DDPG.py".

PPO:
For continuous games, open the ppo_main_cont.py to modify the environment name on line 10, then run it in terminal: "ppo_main_cont.py".
For discrete games, open the ppo_main.py to modify the environment name on line 10, then run it in terminal: "python ppo_main.py".

A3C:
For continuous games, open the continuous_A3C.py to modify the environment name on line 25, then run it in terminal: "python continuous_A3C.py".
For discrete games, open the discrete_A3C.py to modify the environment name on line 24, then run it in terminal: "python discrete_A3C.py.py".

## Dependencies:
### pytorch
```
pip install pytorch
pip install torchvision  
```
### numpy
```
pip install numpy  
```
### gym
For macOS
```
Brew install cmake  
Brew install swig   
pip install gym  
git clone https://github.com/openai/gym  
cd gym  
pip install -e .  
pip install -e '.[all]'  
pip install atari-py  
pip install box2d box2d-kengz  
```
                
## Demo file:
DQN:
You just need to run DQN.py  

## Downloading data:
You do not need to download data, because all the environments are in gym library.  

## Credit:
### DQN: https://zhuanlan.zhihu.com/p/137787080  
Difference: 
* rewrite the code from tensorflow to pytorch  
* change replay_size from 2000 to 10000  
* change learning rate from 0.001 to 0.01  
* change the way of randomly selecting actions  
* add 'done' to replay buffer  
* delete policy modification  
* change the training process (including target Q value and online Q value)  

### DDPG: https://zhuanlan.zhihu.com/p/65931777?ivk_sa=1024320u  
Difference: 
* change the network structure  
* change y_true  
* add 'done' to replay buffer  
* change learning rate from 0.001 to 0.01  
* add stochastic action selection mechanism  

### PPO: https://github.com/philtabor/Youtube-Code-Repository/tree/master/ReinforcementLearning/PolicyGradient/PPO/torch
Difference:
* modify to run for contunuous games
* change NN architecture for apples to apples comparision with other methods
* change learning rate from 0.0003 to 0.01
* change number of episodes from 300 to 10000
* add render function

### A3C: https://github.com/MorvanZhou/pytorch-A3C
Difference:
* change number of episodes from 3000 to 10000
* change discount rate from 0.9 to 0.99
* add function to record time
* change learning rate from .0001 to .01
                  
