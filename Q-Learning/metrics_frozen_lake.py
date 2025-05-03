import numpy as np
import random
import gymnasium as gym
from tqdm import tqdm
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import matplotlib.pyplot as plt

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

def train(env, Qtable, n_training_episodes, min_epsilon, max_epsilon, decay_rate, max_steps, win_need):
    win_count = 0
    lose_count = 0
    no_ending = 0
    episode = 0
    epsilon = max_epsilon
    epsilon_evaluate_episode = 0

    with tqdm(total=win_need, desc="Win Count Progress") as pbar:
        # for episode in tqdm(range(n_training_episodes)):
        while win_count < win_need:
            state, env_info = env.reset()

            epsilon_evaluate_episode += 1
            
            for step in range(max_steps):
                action = epsilon_greedy_policy(Qtable, state, epsilon)
                new_state, reward, done, truncated, info =  env.step(action)


                # if done and reward == 0.0:
                #     reward -=0.5

                Qtable[state][action] = Qtable[state][action] + learning_rate * (reward + gamma * np.max(Qtable[new_state]) - Qtable[state][action])

                state = new_state

                # if(episode_count_since_last_win > 10_000_000 and epsilon < 0.1):               
                if(win_count < 8 and epsilon < 0.4):               
                    print("reevaluate espilon")
                    print(Qtable)
                    epsilon_evaluate_episode = 15000
                
                # If game ended (Hole or Goal)
                if done:
                    if reward > 0:
                        pbar.update(1)
                        win_count+=1
                    else:
                        lose_count+=1
                    break
                    
        
            if not done:
                no_ending+=1

            epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*epsilon_evaluate_episode)
            episode += 1

            if(episode%100000==0):
                print(Qtable)

            if(episode%5000==0):
                print(episode,epsilon_evaluate_episode, epsilon,win_count)

    # print("nb win:", win_count)
    # print("nb lose:", lose_count)
    # print("no_ending", no_ending)
    return Qtable, episode


# Initialization
seed = random.randint(0, 1000)
# np.random.seed(seed)
np.random.seed(705)

# Hyperparameters
n_training_episodes = 20000
max_steps = 300             

learning_rate = 0.9
gamma = 0.9

# Exploration vs exploitation https://chat.mistral.ai/chat/0d9ab896-b498-4001-bfeb-ecd5627ffcd9
max_epsilon = 1.0           
min_epsilon = 0.05
# Less cheers exploration
decay_rate = 0.000005

# Map génération
size=9
desc=generate_random_map(size=size)
# desc = [
#     "SFHHFF",
#     "HFFHFF",
#     "HHFHFH",
#     "FFFFFH",
#     "FHHHHH",
#     "FFFFFG"
# ]
map_name=f"{size}x{size}"
win_need = 10


# Show map
demo_env = gym.make("FrozenLake-v1",desc=desc, map_name=map_name, is_slippery=False,render_mode="human")
demo_state, _ = demo_env.reset()

# Training
env = gym.make("FrozenLake-v1",desc=desc, map_name=map_name, is_slippery=False)

Qtable = initialize_Qtable(env)

metrics = []

for i in range(1):
    result_Qtable, episode = train(env, Qtable, n_training_episodes, min_epsilon, max_epsilon, decay_rate, max_steps, win_need)
    metrics.append(episode)
    print(f"End of Learning #{i} moyenne: ", np.mean(np.array(metrics)))

# print(result_Qtable)

# Demonstration
# print("TRAINING DONE in:")
# print(metrics)

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

# x_length = len(metrics)

# x = list(range(0, x_length))
# y = metrics

# fig, ax = plt.subplots()
# ax.plot(x, y, label="steps")

# plt.show()