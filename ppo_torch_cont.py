import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions import MultivariateNormal


class PPOMemory:
    def __init__(self, batch_size):
        # storing information for states
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        # generate batches indices for gradient descent
        # will create a list of indices to randomly sample steps
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + self.batch_size] for i in batch_start]

        return np.array(self.states), \
               np.array(self.actions), \
               np.array(self.probs), \
               np.array(self.vals), \
               np.array(self.rewards), \
               np.array(self.dones), \
               batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    # clear trajectory data after policy update
    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []


class ActorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, alpha, chkpt_dir='tmp/ppo'):
        super(ActorNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'actor_torch_ppo')
        self.input = nn.Sequential(nn.Linear(state_dim, 100),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.output = nn.Sequential(nn.Linear(100, action_dim),
                                    nn.Tanh())

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    # draw from probabilities to get action
    def forward(self, s):
        if isinstance(s, np.ndarray):
            s = T.tensor(s, dtype=T.float)
        x = self.input(s)
        x = self.output(x) * 2
        return x

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


# create critic function approximator for V
class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, alpha, chkpt_dir='tmp/ppo'):
        super(CriticNetwork, self).__init__()

        self.checkpoint_file = os.path.join(chkpt_dir, 'critic_torch_ppo')

        # no activation layer for output
        self.input = nn.Sequential(nn.Linear(state_dim + action_dim, 100),
                                   nn.LeakyReLU(negative_slope=0.2))
        self.output = nn.Sequential(nn.Linear(100, action_dim))

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)

        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, s, a):
        x = T.cat([s, a], 1).float()
        x = self.input(x)
        x = self.output(x)
        return x

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class Agent:
    def __init__(self, env, gamma=0.99, alpha=0.0001, gae_lambda=0.95,
                 policy_clip=0.2, batch_size=64, n_epochs=10):

        self.env = env
        self.gamma = gamma
        # self.alpha = alpha
        self.policy_clip = policy_clip
        self.n_epochs = n_epochs
        self.gae_lambda = gae_lambda  # smoothing parameter

        self.s_dim = self.env.observation_space.shape[0]
        self.a_dim = self.env.action_space.shape[0]

        self.actor = ActorNetwork(self.s_dim, self.a_dim, alpha)
        self.critic = CriticNetwork(self.s_dim, self.a_dim, alpha)
        self.memory = PPOMemory(batch_size)

        self.cov_var = T.full(size=(self.a_dim,), fill_value=0.5)  # get covariance matrix
        self.cov_mat = T.diag(self.cov_var)  # compute standard deviation along diagonals

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def save_models(self):
        print('... saving models ...')
        self.actor.save_checkpoint()
        self.critic.save_checkpoint()

    def load_models(self):
        print('... loading models ...')
        self.actor.load_checkpoint()
        self.critic.load_checkpoint()

    def choose_action(self, obs):
        # create and sample from policy distribution to choose action
        state = T.tensor([obs], dtype=T.float).to(self.actor.device)
        mean = self.actor(state)
        dist = MultivariateNormal(mean, self.cov_mat.to(self.critic.device))
        action = dist.sample().to(self.critic.device)
        value = self.critic(state, action) # create baseline for advantage function
        probs = T.squeeze(dist.log_prob(action)).item()

        # if len(action[0]) > 1:
        #     action = T.squeeze(action).detach().numpy()
        #     # print('the actions im sampling are:', action)
        #     value = T.squeeze(value).detach()
        # else:
        #     action = T.squeeze(action).item()
        #     value = T.squeeze(value).item()

        return action, probs, value

    def save_agent(self, actor_path, critic_path):
        print('model saved')
        T.save(self.actor.state_dict(), actor_path)
        T.save(self.critic.state_dict(), critic_path)

    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr, \
            reward_arr, dones_arr, batches = \
                self.memory.generate_batches()

            # print('my batches look like:', batches)

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            # calculate return (cumulative discounted reward) for each trajectory
            # and use return - baseline to calculate advantage
            for t in range(len(reward_arr) - 1):
                discount = .99
                a_t = 0
                for k in range(t, len(reward_arr) - 1):
                    a_t += discount * (reward_arr[k] + self.gamma * values[k + 1] * \
                                       (1 - int(dones_arr[k])) - values[k])
                    discount *= self.gamma * self.gae_lambda
                advantage[t] = a_t
            advantage = T.tensor(advantage).to(self.actor.device)
            values = T.tensor(values).to(self.actor.device)

            # finding the optimal policy parameters using the clipped surrogate loss function
            for batch in batches:

                states = T.tensor(state_arr[batch], dtype=T.float).to(self.actor.device)
                old_probs = T.tensor(old_prob_arr[batch]).to(self.actor.device)
                actions = T.tensor(action_arr[batch]).unsqueeze(1).to(
                    self.actor.device).float()  # need to increase dimension of tensor from 1 to 2 to match state

                mean = self.actor(states)
                dist = MultivariateNormal(mean, self.cov_mat.to(self.critic.device))
                critic_value = self.critic(states, actions)
                critic_value = T.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                prob_ratio = new_probs.exp() / old_probs.exp()
                # prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1 - self.policy_clip,
                                                 1 + self.policy_clip) * advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns - critic_value) ** 2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5 * critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()
