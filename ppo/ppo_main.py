import gym
import numpy as np
from ppo_torch_discrete import Agent
import time

if __name__ == '__main__':
    start_time = time.time()

    # initialize environment and initial conditions
    env_name = 'LunarLander-v2'
    env = gym.make(env_name)
    N = 20
    batch_size = 5
    n_epochs = 4
    alpha = 0.0001
    agent = Agent(n_actions=env.action_space.n, batch_size=batch_size,
                    alpha=alpha, n_epochs=n_epochs,
                    input_dims=env.observation_space.shape)
    n_games = 10000

    figure_file = f'plots/{env_name}.png'

    best_score = env.reward_range[0]
    score_history = []

    learn_iters = 0
    avg_score = 0
    n_steps = 0

    for i in range(n_games):
        observation = env.reset()
        done = False
        score = 0
        while not done:
            env.render()
            action, prob, val = agent.choose_action(observation)
            observation_, reward, done, info = env.step(action)
            n_steps += 1


            score += reward
            agent.remember(observation, action, prob, val, reward, done)
            if n_steps % N == 0:
                agent.learn()
                learn_iters += 1
            observation = observation_
        score_history.append(score)
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_models()

        print('episode', i, 'score %.1f' % score, 'avg score %.1f' % avg_score,
                'time_steps', n_steps, 'learning_steps', learn_iters)

    print("--- %s seconds to train---" % (time.time() - start_time))

    agent.save_agent(actor_path=env_name + '_PPO_actor.h5',critic_path=env_name + '_PPO_critic.h5')
    np.save(env_name+'_PPO.npy',score_history)
    x = [i+1 for i in range(len(score_history))]
    env.close()

    end_time = time.time() - start_time
    time_text = f'the time it takes to run is {end_time}'
    print('time recorded')
    with open('times/'+env_name+'time.txt', 'w') as f:
        f.write(time_text)

