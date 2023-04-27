# -*- coding: utf-8 -*-


!pip install git+https://github.com/MattChanTK/gym-maze.git


import os
os.environ["SDL_VIDEODRIVER"] = "dummy"

import gym
import gym_maze
import matplotlib.pyplot as plt
import numpy as np

env = gym.make("maze-sample-10x10-v0")
height = width = 10
# env = gym.make("maze-random-10x10-plus-v0")
# height = width = 10
num_actions = 4

state = env.reset()
state = env.step('E')
state = env.step('S')
image = env.render('rgb_array')

plt.imshow(image.copy())
plt.show()

"""### Q-Learning

In the maze environment, a reward of $1$ is given when the agent reaches the goal. For every step in the maze, the agent recieves a penalty reward of $-0.1/\text{(number of cells)}$. We want to maximize the cumulative returns of the agent over time. One way to define the objective is $ R_N = \frac1N \mathbb{E} \sum_{n=1}^{N} \sum_{t=1}^{T} r_{it} $ where $N$ is the number of episode and each episode has $T$ steps.

"""

num_episodes = 1000
max_steps_per_episode = 500
learning_rate = 0.1

q_values = np.zeros((height, width, num_actions))

### YOUR IMPLEMENTATION HERE ###
epsilon = 0.05
alpha = 0.1
def get_action_with_eps_greedy(state):
  p = np.random.binomial(1,1-epsilon)
  if p == 1:
    matrix = state[0]
    row = state[1]
    max = np.argmax(q_values[matrix,row])
    return int(max)
  else:
    return int(np.random.randint(0,4))
  


def update_q_values(state, action, reward, next_state):
  Q = q_values[state[0]][state[1]][action]

  #for next state 
  row = q_values[next_state[0]][next_state[1]]
  index = int(np.argmax(row))

  #update value 

  q_values[state[0]][state[1]][action] = Q + alpha*( reward + q_values[next_state[0],next_state[1],index] - Q )

total_return = 0.0
episode_returns = np.zeros(num_episodes)

for i in range(num_episodes):
    state = env.reset()
    state = tuple(state.astype(np.int32))
    episode_return = 0.0
    for t in range(max_steps_per_episode):
        # get action according to current policy
        action = get_action_with_eps_greedy(state)
        action_name = 'NESW'[action]  # convert action index to action name
        
        # run one simulation step
        next_state, reward, done, _ = env.step(action_name)
        next_state = tuple(next_state.astype(np.int32))
        episode_return += reward
        
        # update the Q values
        update_q_values(state, action, reward, next_state)

        if done: break
        state = next_state

    total_return += episode_return
    episode_returns[i] = episode_return
    if i % 100 == 0 or i == num_episodes-1:
        print(f'{i:5d}: average_episode_return = {total_return / (i+1):.3f}, last_episode_return = {episode_return:.3f}')

plt.plot(episode_returns)
plt.xlabel('#episode'); plt.ylabel('return')
plt.show()

"""### Double Q-Learning
"""

learning_rate = 0.2

q_values_1 = np.zeros((height, width, num_actions))
q_values_2 = np.zeros((height, width, num_actions))

### YOUR IMPLEMENTATION HERE ###
epsilon = 0.05
alpha = 0.1
def get_action_with_eps_greedy(state):
  p = np.random.binomial(1,1-epsilon)
  if p == 1:
    matrix = state[0]
    row = state[1]
    max = np.argmax(q_values_1[matrix,row] + q_values_2[matrix,row])
    return int(max)
  else:
    return int(np.random.randint(0,4))
  


def update_q_values(state, action, reward, next_state):
  Q1 = q_values_1[state[0]][state[1]][action]
  Q2 = q_values_2[state[0]][state[1]][action]
  matrix = state[0]
  row = state[1]
  nmatrix = next_state[0]
  nrow = next_state[1]
  q1prime = np.argmax(q_values_1[next_state[0],next_state[1]])
  q2prime = np.argmax(q_values_2[next_state[0],next_state[1]])


  if np.random.binomial(1,.5) == 1:
    #Q1 update 
    q_values_1[state[0]][state[1]][action] = Q1 + alpha*(reward+ q_values_2[next_state[0],next_state[1],q1prime]-Q1 )

  else:
    #Q2 update
    q_values_2[state[0]][state[1]][action] = Q2 + alpha*(reward+ q_values_1[next_state[0],next_state[1],q1prime]-Q2 )

total_return = 0.0
episode_returns = np.zeros(num_episodes)

for i in range(num_episodes):
    state = env.reset()
    state = tuple(state.astype(np.int32))
    episode_return = 0.0
    for t in range(max_steps_per_episode):
        # get action according to current policy
        action = get_action_with_eps_greedy(state)
        action_name = 'NESW'[action]  # convert action index to action name
        
        # run one simulation step
        next_state, reward, done, _ = env.step(action_name)
        next_state = tuple(next_state.astype(np.int32))
        episode_return += reward
        
        # update the Q values
        update_q_values(state, action, reward, next_state)

        if done: break
        state = next_state

    total_return += episode_return
    episode_returns[i] = episode_return
    if i % 100 == 0 or i == num_episodes-1:
        print(f'{i:5d}: average_episode_return = {total_return / (i+1):.3f}, last_episode_return = {episode_return:.3f}')

plt.plot(episode_returns)
plt.xlabel('#episode'); plt.ylabel('return')
plt.show()



from IPython import display
env = gym.make("maze-sample-10x10-v0")
height = width = 10
# env = gym.make("maze-random-10x10-plus-v0")
# height = width = 10
num_actions = 4

state = env.reset()
state = tuple(state.astype(np.int32))
episode_return = 0.0
image = env.render('rgb_array').copy()
avg_image = image

for t in range(max_steps_per_episode):
    # get greedy action
    action = np.argmax(q_values[state[0], state[1]])
    action_name = 'NESW'[action]

    # run one simulation step
    next_state, reward, done, _ = env.step(action_name)
    next_state = tuple(next_state.astype(np.int32))
    episode_return += reward

    image = env.render('rgb_array').copy()
    avg_image = image * 0.05 + np.minimum(image, avg_image) * 0.95
    
    #NOTE: commented this out because this causes google colab to crash. 
    '''
    display.clear_output(wait=True)
    plt.clf()
    plt.imshow(image)
    plt.show()
    plt.pause(0.01)
    '''
    if done: break
    state = next_state

plt.imshow(avg_image.astype(np.uint8))
plt.show()

# env.close()
# del env
