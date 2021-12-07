import gym
import numpy as np
from ppo_torch_cont import Agent
# from ppo_torch import Agent
import time

if __name__ == '__main__':
    # initialize environment and initial conditions
    start_time = time.time()
    env_name = 'MountainCarContinuous-v0'
    env = gym.make(env_name)
    N = 20  # trajectories to store in memory
    batch_size = 5  # batchsize for update
    n_epochs = 4
    alpha = 0.001

    # initialize agent
    agent = Agent(env=env, batch_size=batch_size,
                    alpha=alpha, n_epochs=n_epochs)
    n_games = 10000

    figure_file = f'plots/{env_name}_cont.png'

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
            observation_, reward, done, info = env.step([action])  # need to make into array to convert to tensor
            n_steps += 1

            score += reward
            agent.remember(observation, action, val, prob, reward, done)  # store trajectory in PPO memory

            # update parameters when memory is full
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


    agent.save_agent(actor_path=env_name + '_PPO_actor.h5',critic_path=env_name + '_PPO_critic.h5')
    np.save(env_name+'_PPO.npy',score_history)
    x = [i+1 for i in range(len(score_history))]
    env.close()
    
    end_time = time.time() - start_time
    time_text = f'the time it takes to run is {end_time}'
    print('time recorded')
    with open('times/'+env_name+'time.txt', 'w') as f:
        f.write(time_text)
