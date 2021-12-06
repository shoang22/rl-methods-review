# RL-methods-review
Review of Fundamental RL Algorithms for Game Control

## Running instruction:
DQN: Open the DQN.py to modify the environment name on line 159, then run it in terminal: "python DQN.py".

DDPG: Open the DDPG.py to modify the environment name on line 134, then run it in terminal: "python DDPG.py".




## Dependencies:
pytorch: 1. pip install pytorch 2. pip install torchvision  
numpy: pip install numpy  
gym: For macOS, Brew install cmake  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&nbsp;Brew install swig   
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&nbsp;pip install gym  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&nbsp;git clone https://github.com/openai/gym  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&nbsp;cd gym  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&nbsp;pip install -e .  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&nbsp;pip install -e '.[all]'  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&nbsp;pip install atari-py  
&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&nbsp;pip install box2d box2d-kengz  
                
## Demo file:
You can just run the DQN.py  

## Downloading data:
You do not need to download data, because all the environments are in gym library.  

## Credit:
DQN: https://zhuanlan.zhihu.com/p/137787080  
&emsp;&emsp;&ensp;&emsp;Difference: rewrite the code from tensorflow to pytorch  
                 change replay_size from 2000 to 10000  
                 change learning rate from 0.001 to 0.01  
                 change the way of randomly selecting actions  
                 add 'done' to replay buffer  
                 delete policy modification  
                 change the training process (including target Q value and online Q value)  
              
DDPG: https://zhuanlan.zhihu.com/p/65931777?ivk_sa=1024320u  
      Difference: change the network structure  
                  change y_true  
                  add 'done' to replay buffer  
                  change learning rate from 0.001 to 0.01  
                  add stochastic action selection mechanism  
                  
