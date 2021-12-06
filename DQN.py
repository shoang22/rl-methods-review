from collections import deque
import random
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time


import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
##The net is easy, the input is the observation (which involves the statements of the game), the output is the score array
# of actions such as [0.4000,0.6000] for two actions)
class net(nn.Module):
    def __init__(self, STATE_DIM,ACTION_DIM):
        super(net, self).__init__()
        self.input = nn.Sequential(nn.Linear(STATE_DIM,100), nn.ReLU())
        self.output = nn.Sequential(nn.Linear(100, ACTION_DIM))

    def forward(self, x):
        x=self.input(x)
        x=self.output(x)
        return x
# class net(nn.Module):
#     def __init__(self, STATE_DIM,ACTION_DIM):
#         super(net, self).__init__()
#         self.input = nn.Sequential(nn.Linear(STATE_DIM,128), nn.LeakyReLU(negative_slope=0.2))
#         self.hidden1 = nn.Sequential(nn.Linear(128,256), nn.LeakyReLU(negative_slope=0.2))
#         self.hidden2 = nn.Sequential(nn.Linear(256,512), nn.LeakyReLU(negative_slope=0.2))
#         self.output = nn.Sequential(nn.Linear(512, ACTION_DIM))
#
#     def forward(self, x):
#         x=self.input(x)
#         x = self.hidden1(x)
#         x = self.hidden2(x)
#         x=self.output(x)
#         return x

class DQN(object):
    def __init__(self,STATE_DIM, ACTION_DIM):
        self.step = 0
        self.update_freq = 100  # the frequency of refreshing the model
        self.replay_size = 10000  # training set size
        self.replay_queue = deque(maxlen=self.replay_size)
        self.model = self.create_model(STATE_DIM, ACTION_DIM)
        self.target_model = self.create_model(STATE_DIM, ACTION_DIM)
        self.criterion=torch.nn.MSELoss()
        self.optimizer=optim.AdamW(self.model.parameters(), lr=0.01)
        self.STATE_DIM=STATE_DIM
        self.ACTION_DIM=ACTION_DIM

    def create_model(self,STATE_DIM, ACTION_DIM):
        model=net(STATE_DIM, ACTION_DIM)
        # model = torch.nn.DataParallel(model).cuda()
        return model

    def act(self, s, epsilon=1):
        """predict the action"""
        # At the beginning, add a little random component to generate more states
        if np.random.rand() < epsilon:
            return np.random.choice(np.arange(self.ACTION_DIM))

        return torch.argmax(self.model(torch.Tensor(s)).detach()).item()
        # if np.random.uniform() < epsilon:
        #     return np.argmax(self.model(torch.Tensor(s)).detach()).numpy()
        # return np.random.choice(np.arange(self.ACTION_DIM))
        # if np.random.uniform() < epsilon:
        #     return np.argmax(self.model(torch.Tensor(s)))
        # return np.random.choice(np.arange(self.ACTION_DIM))


    def save_model(self, file_path):
        print('model saved')
        torch.save(self.model.state_dict(), file_path)

    def remember(self, s, a, next_s, reward,done):
        """History recording, where the reward can be adjusted, e.g. reward+=1 or reward-=2 or something
        to converge quickly when next_s[n]>x or some cases"""
        # if next_s[0] >= 0.4:
        #     reward += 1

        self.replay_queue.append((s, a, next_s, reward,done))

    def train(self,batch_size=64, lr=1, gamma=0.95):
        #Before training we need to fill up the replay queue
        if len(self.replay_queue) < self.replay_size:
            return True if len(self.replay_queue)/self.replay_size==1 else False

        self.step += 1
        # Every update_freq step, assign the weights of the model to target_model
        # hard copy
        if self.step % self.update_freq == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        replay_batch = random.sample(self.replay_queue, batch_size)
        """replay_batch:[[s_1,a_1,next_s_1,r_1],
                        [s_2,a_2,next_s_2,r_2],
                        ...,
                        [s_64,a_64,next_s_64,r_64]]"""
        s_batch = torch.tensor([replay[0] for replay in replay_batch]).detach()
        """s_batch:[[s_1],
                    [s_2],
                    ...,
                    [s_64]]"""
        next_s_batch = torch.tensor([replay[2] for replay in replay_batch]).detach()
        """s_batch:[[next_s_1],
                    [next_s_2],
                    ...,
                    [next_s_64]]"""


        action_batch=torch.tensor([[replay[1] for replay in replay_batch]]).t().detach()
        reward_batch=torch.tensor([replay[3] for replay in replay_batch],dtype=torch.float32).detach()
        done_batch=torch.tensor([replay[4] for replay in replay_batch]).float().detach()
        # Q is the score array of each observation of a batch
        Q = self.model(s_batch).gather(1,action_batch)
        """Q:    action1    action2
        state1    2.132        0     (if I choose action1 for state1)
        state2      0        3.817
        state3      0        4.123
        """
        # Q_next = self.target_model(next_s_batch).detach()
        Q_next = self.target_model(next_s_batch).detach()
        # Q_next[done_batch]=0.0
        # print(Q_next)
        Q_target=(reward_batch + (1-done_batch)*gamma * torch.max(Q_next, dim=1)[0]).view(batch_size, 1)
        # Q = self.model(s_batch)
        # """Q:  action1   action2
        # state1  0.5923    0.2601
        # state2  0.5556    0.5214
        # .....
        # state64 0.4116    0.1151"""
        # Q_next = self.target_model(next_s_batch).detach()
        #
        # Q_next[done_batch] = 0.0
        #
        # Q_target=Q.clone()
        # # Update the Q values in the training set using the formula
        #
        # for i, replay in enumerate(replay_batch):
        #     _, a, _, reward, _= replay
        #     """a is 0 or 1 here"""
        #     Q_target[i][a] = (1 - lr) * Q[i][a] + lr * (reward + gamma * Q_next[i].max(-1)[0])
        #     """reward + factor * Q_next[i].max(-1)[0] is what we should actually receive
        #     getted by using the Bellman equation and the greedy strategy
        #     , our goal is to keep our Q-value close to this value(target Q-value)."""
        # Train
        loss = self.criterion(Q, Q_target)
        # backward
        self.optimizer.zero_grad()
        loss.backward(loss)
        self.optimizer.step()
        return True


# for different game, we need to modify the 'envname'
envname='CartPole-v0'
env = gym.make(envname)
STATE_DIM, ACTION_DIM=env.observation_space.shape[0],env.action_space.n
# print(env.action_space)
# #> Discrete(2)
print(env.observation_space)
print(env.action_space)
#> Box(4,)
# print(env.observation_space.high)
# #> array([ 2.4       ,         inf,  0.20943951,         inf])
# print(env.observation_space.low)
# #> array([-2.4       ,        -inf, -0.20943951,        -inf])
episodes = 10000  # train for 500 times
score_list = np.zeros(episodes)  # record all the scores
agent = DQN(STATE_DIM, ACTION_DIM)
# print("Start filling up the replay queue")
# while True:
#     s = env.reset()
#     # start_pos=s[0]
#     # start_speed=s[1]
#     end=False
#     while True:
#         env.render()
#         a = agent.act(s)
#         next_s, reward, done, _ = env.step(a)
#         agent.remember(s, a, next_s, reward,done)
#         ratio=agent.train(done)
#         s = next_s
#         if ratio==True:
#             end=True
#             break
#         if done:
#             break
#     if end==True:
#         break
# print("End up")
start=time.time()
for i in range(episodes):
    s = env.reset()
    score = 0
    while True:
        # env.render()
        epsilon=torch.tensor(max(1-i/500,0.01))
        a = agent.act(s,epsilon)
        next_s, reward, done, _ = env.step(a)
        score += reward
        reward = reward if not done else -10
        agent.remember(s, a, next_s, reward,done)
        agent.train()
        s = next_s
        if done:
            score_list[i]=score
            print('episode:', i, 'score:', score, 'average_score:',np.mean(score_list[:i+1]))
            break
end=time.time()
print("total time is:",end-start)

# agent.save_model(file_path=envname+'_DQN_reward.h5')
score_list=np.array(score_list)
np.save(envname+ '_DQN_BetterPolicy.npy',score_list)
env.close()