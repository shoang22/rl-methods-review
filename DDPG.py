import gym
from collections import deque
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class Actor(nn.Module):
    def __init__(self, STATE_DIM, ACTION_DIM):
        super(Actor, self).__init__()
        self.input = nn.Sequential(nn.Linear(STATE_DIM, 100), nn.ReLU())
        ##use Tanh to make output within [-1,1]
        self.output = nn.Sequential(nn.Linear(100, ACTION_DIM),nn.Tanh())

    def forward(self, s):
        x = self.input(s)
        ##here to project to the range of the environment
        x = self.output(x)
        return x


class Critic(nn.Module):
    def __init__(self, STATE_DIM, ACTION_DIM):
        super(Critic, self).__init__()
        self.input = nn.Sequential(nn.Linear(STATE_DIM+ACTION_DIM, 100), nn.ReLU())
        self.output = nn.Sequential(nn.Linear(100, ACTION_DIM))

    def forward(self, s,a):
        x = torch.cat([s, a], 1)
        x = self.input(x)
        x = self.output(x)
        return x


class Agent(object):
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

        s_dim = self.env.observation_space.shape[0]
        a_dim = self.env.action_space.shape[0]
        print(self.env.action_space)
        self.a_high= self.env.action_space.high[0]
        self.a_low = self.env.action_space.low[0]

        self.actor = Actor(s_dim, a_dim)
        self.actor_target = Actor(s_dim, a_dim)
        self.critic = Critic(s_dim, a_dim)
        self.critic_target = Critic(s_dim, a_dim)
        self.actor_optim = optim.AdamW(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optim = optim.AdamW(self.critic.parameters(), lr=self.critic_lr)
        self.replay_queue = deque(maxlen=self.replay_size)

        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

    def act(self, s0):
        s0 = torch.tensor(s0.copy(), dtype=torch.float).unsqueeze(0)
        a0 = self.actor(s0).squeeze(0).detach().numpy()
        ##When the model has been trained, delete this line.
        a0 = np.clip(np.random.normal(a0,self.VAR),self.a_low,self.a_high)

        return a0

    def remember(self, *transition):
        self.replay_queue.append(transition)

    def updateVAR(self):
        if self.VAR>0.2:
            self.VAR-=0.2

    def save_model(self, actor_path,critic_path):
        print('model saved')
        torch.save(self.actor.state_dict(), actor_path)
        torch.save(self.critic.state_dict(), critic_path)

    def train(self):
        if len(self.replay_queue) < self.replay_size:
            return True if len(self.replay_queue)/self.replay_size==1 else False

        replay_batch = random.sample(self.replay_queue, self.batch_size)

        s0, a0, s1, r1, done = zip(*replay_batch)

        s0 = torch.tensor(s0, dtype=torch.float)
        a0 = torch.tensor(a0, dtype=torch.float)
        s1 = torch.tensor(s1, dtype=torch.float)
        r1 = torch.tensor(r1, dtype=torch.float).view(self.batch_size, -1)
        done=torch.tensor(done,dtype=torch.float)
        #r1.shape=[64,1]

        def critic_learn():
            a1 = self.actor_target(s1).detach()
            y_true = r1 + (1-done).unsqueeze(1).clone().detach() * self.gamma * self.critic_target(s1, a1).detach()
            y_pred = self.critic(s0, a0)

            loss_fn = nn.MSELoss()
            loss = loss_fn(y_true, y_pred)
            self.critic_optim.zero_grad()
            loss.backward()
            self.critic_optim.step()

        def actor_learn():
            loss = -torch.mean(self.critic(s0, self.actor(s0)))
            self.actor_optim.zero_grad()
            loss.backward()
            self.actor_optim.step()

        def soft_update(net_target, net, tau):
            for target_param, param in zip(net_target.parameters(), net.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

        critic_learn()
        actor_learn()
        soft_update(self.critic_target, self.critic, self.tau)
        soft_update(self.actor_target, self.actor, self.tau)
        return True
# def START_seed():
#     seed = 42
#     torch.manual_seed(seed)
#     torch.cuda.manual_seed(seed)
#     torch.cuda.manual_seed_all(seed)
#     np.random.seed(seed)
#     random.seed(seed)
#     torch.backends.cudnn.benchmark = False
#     torch.backends.cudnn.deterministic = True
# START_seed()
envname='LunarLanderContinuous-v2'
env = gym.make(envname)

params = {
    'env': env,
    'gamma': 0.95,
    'actor_lr': 0.01,
    'critic_lr': 0.01,
    'tau': 0.02,
    'replay_size': 10000,
    'batch_size': 64,
    'VAR':2,
}
agent = Agent(**params)
score_list = []
# print("Start filling up the replay queue")
# while True:
#     s = env.reset()
#     end=False
#     while True:
#         env.render()
#         a = agent.act(s)
#         next_s, reward, done, _ = env.step(a)
#         agent.remember(s, a, next_s, reward)
#         ratio=agent.train()
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
episodes=10000
for episode in range(episodes):
    s = env.reset()
    score = 0
    while True:
        # env.render()
        a = agent.act(s)
        next_s, reward, done, _ = env.step(a)
        score += reward
        agent.remember(s, a, next_s, reward,done)
        agent.train()
        s = next_s
        if done:
            score_list.append(score)
            print('episode:', episode, 'score:', score, 'average_score:',np.mean(score_list[:episode+1]))
            break
    if episode % 1000==0:
        agent.updateVAR()
end=time.time()
print("total time is:",end-start)
# agent.save_model(actor_path=envname + '_DDPG_actor.h5',critic_path=envname + '_DDPG_critic.h5')
score_list=np.array(score_list)
np.save(envname+ '_DDPG.npy',score_list)
env.close()
    # for step in range(500):
    #     env.render()
    #     a0 = agent.act(s0)
    #
    #     s1, r1, done, _ = env.step(a0)
    #     agent.put(s0, a0, r1, s1)
    #
    #     episode_reward += r1
    #     s0 = s1
    #
    #     agent.learn()

    # print(episode, ': ', episode_reward)