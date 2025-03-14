import numpy as np
import random
import gymnasium as gym
from tqdm import tqdm
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

def initialize_Qtable(env):
    state_space = env.observation_space.n
    action_space = env.action_space.n
    Qtable = np.zeros((state_space, action_space))
    return Qtable

def epsilon_greedy_policy(Qtable, state, epsilon):
    random_int = random.uniform(0,1)
    
    if random_int > epsilon:
        action = np.argmax(Qtable[state])
    else:
        action = env.action_space.sample()

    return action

def train(env, Qtable, n_training_episodes, min_epsilon, max_epsilon, decay_rate, max_steps):
    win_count = 0
    lose_count = 0
    no_ending = 0
    episode = 0
    win_needs = 50
    epsilon = max_epsilon

    with tqdm(total=win_needs, desc="Win Count Progress") as pbar:
        # for episode in tqdm(range(n_training_episodes)):
        while win_count < win_needs:
            state, env_info = env.reset()
            
            for step in range(max_steps):
                action = epsilon_greedy_policy(Qtable, state, epsilon)
                new_state, reward, done, truncated, info =  env.step(action)

                Qtable[state][action] = Qtable[state][action] + learning_rate * (reward + gamma * np.max(Qtable[new_state]) - Qtable[state][action])

                state = new_state
                
                # If game ended (Hole or Goal)
                if done:
                    if reward > 0:
                        # print("win ++",win_count)
                        pbar.update(1)
                        win_count+=1
                    else:
                        lose_count+=1
                    break
                    
        
            if not done:
                no_ending+=1

            epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
            episode += 1


    print("nb win:", win_count)
    print("nb lose:", lose_count)
    print("no_ending", no_ending)
    return Qtable


# Initialization

# Hyperparameters
n_training_episodes = 20000
max_steps = 300             

learning_rate = 0.25  
gamma = 0.95

# Exploration vs exploitation
max_epsilon = 1.0           
min_epsilon = 0.05
# Less cheers exploration      
decay_rate = 0.00005

# Map génération
size=6
desc=generate_random_map(size=size)
map_name=f"{size}x{size}"

# Show map
demo_env = gym.make("FrozenLake-v1",desc=desc, map_name=map_name, is_slippery=False,render_mode="human")
demo_state, _ = demo_env.reset()

# Training
env = gym.make("FrozenLake-v1",desc=desc, map_name=map_name, is_slippery=False)

Qtable = initialize_Qtable(env)

result_Qtable = train(env, Qtable, n_training_episodes, min_epsilon, max_epsilon, decay_rate, max_steps)
print(result_Qtable)

# Demonstration
print("TRAINING DONE")

is_done = False

while not is_done:
    action = np.argmax(result_Qtable[demo_state])
    new_state, reward, done,truncated, info = demo_env.step(action)
    demo_state = new_state
    is_done = done

    if done:
        if reward==1.0:
            print("JAY JAY !!")
        else:
            print("PWND !!")
        break
