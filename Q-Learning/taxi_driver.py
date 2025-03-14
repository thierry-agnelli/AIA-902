import gymnasium as gym
import random
import numpy as np
from tqdm import tqdm


seed = random.randint(0, 1000)
np.random.seed(seed)

def define_QTable(env):
    states = env.observation_space.n
    actions = env.action_space.n

    Qtable = np.zeros((states, actions))

    return Qtable

def epsilon_greedy_policy(QTable, state, epsilon):
    random_int = random.uniform(0,1)

    if(random_int > epsilon):
        action = np.argmax(QTable[state])
    else:
        action = env.action_space.sample()

    return action
    
def train(env, qtable, epsilon, max_epsilon, min_epsilon, learning_rate, discount_rate, decay_rate, max_episode, max_steps):
    win_count = 0
    
    for episode in tqdm(range(max_episode)):
        
        state, _ = env.reset()

        for step in range(max_steps):
            action = epsilon_greedy_policy(qtable, state, epsilon)
            
            new_state, reward, done, _, _ = env.step(action)
            
            qtable[state, action] = qtable[state, action] + learning_rate * (reward + discount_rate * np.max(qtable[new_state,:]) - qtable[state, action])
            
            # Mettre à jour l'état
            state = new_state
         
            if done == True:
                win_count+=1
                break

        epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)

    print("nb win:", win_count)

max_episode = 1000
# max_episode = 10
max_steps = 99

learning_rate = 0.9
discount_rate = 0.8

max_epsilon = 1.0
min_epsilon = 0.05

decay_rate = 0.005

epsilon = max_epsilon


env = gym.make("Taxi-v3")


qtable = define_QTable(env)

train(env, qtable, epsilon, max_epsilon, min_epsilon, learning_rate, discount_rate, decay_rate, max_episode, max_steps)
env.close()

# watch trained agent
state, _ = env.reset(seed=seed)
done = False
rewards = 0

print(f"TRAINED AGENT")

env = gym.make("Taxi-v3", render_mode="human")
state, _ = env.reset(seed=seed)

for s in range(max_steps):
    action = np.argmax(qtable[state,:])
    new_state, reward, done, info, _ = env.step(action)
    rewards += reward
    env.render()
    
    state = new_state
    # env.render()
    if done == True:
        print("JAYJAY")
        print("score: ", rewards)
        break

env.close()
