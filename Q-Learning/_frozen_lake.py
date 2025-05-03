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
    episode_count_since_last_win = 0

    metrics = {
        "win_count": [],
        "lose_count": [],
        "no_ended_count": [],
        "win_rate": [],
        "steps": []
    }

    # with tqdm(total=win_need, desc="Win Count Progress") as pbar:
    for episode in tqdm(range(n_training_episodes)):
    # while win_count < win_need:
        state, env_info = env.reset()
        
        for step in range(max_steps):
            action = epsilon_greedy_policy(Qtable, state, epsilon)
            new_state, reward, done, truncated, info =  env.step(action)

            # if done and reward == 0.0:
            #     reward -=0.1

            Qtable[state][action] = Qtable[state][action] + learning_rate * (reward + gamma * np.max(Qtable[new_state]) - Qtable[state][action])

            state = new_state
            episode_count_since_last_win += 1

            if(episode_count_since_last_win > 10_000_000 and epsilon < 0.1):                
                print("epsilon:",win_count,episode,episode_count_since_last_win, epsilon)
                print("no-ending",no_ending)
                episode = 0
                episode_count_since_last_win = 0
            
            # If game ended (Hole or Goal)
            if done:
                if reward > 0:
                    # pbar.update(1)
                    win_count+=1
                    episode_count_since_last_win = 0
                else:
                    lose_count+=1
                break
             
    
        if not done:
            no_ending+=1
        
        metrics["win_count"].append(win_count)
        metrics["lose_count"].append(lose_count)
        metrics["no_ended_count"].append(no_ending)
        metrics["steps"].append(step)
        
        metrics["win_rate"].append(win_count/n_training_episodes)
        epsilon = min_epsilon + (max_epsilon - min_epsilon)*np.exp(-decay_rate*episode)
        # episode += 1

    print("nb win:", win_count)
    print("nb lose:", lose_count)
    print("no_ending", no_ending)
    return Qtable, metrics


# Initialization


seed = random.randint(0, 1000)
# np.random.seed(seed)
np.random.seed(705)

# Hyperparameters
n_training_episodes = 10000
max_steps = 100             

learning_rate = 0.5
gamma = 0.9

# Exploration vs exploitation
max_epsilon = 1.0           
min_epsilon = 0.05
# Less cheers exploration
decay_rate = 0.000005

# Map génération
size=4
# desc=generate_random_map(size=size)
desc = [
    "SFHF",
    "HFFF",
    "HHFH",
    "FFFG"
]


map_name=f"{size}x{size}"
win_need = 10


# Show map
demo_env = gym.make("FrozenLake-v1",desc=desc, map_name=map_name, is_slippery=False,render_mode="human")
demo_state, _ = demo_env.reset()

# Training
env = gym.make("FrozenLake-v1",desc=desc, map_name=map_name, is_slippery=False)

Qtable = initialize_Qtable(env)

result_Qtable, metrics = train(env, Qtable, n_training_episodes, min_epsilon, max_epsilon, decay_rate, max_steps, win_need)
print(result_Qtable)

# Demonstration
print("TRAINING DONE")

is_done = False

x_length = n_training_episodes

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()
x = list(range(0, x_length))


yWin = metrics["win_count"]
yLose = metrics["lose_count"]
ySteps = metrics["steps"]
yWinRate = metrics["win_rate"]

# ax1.plot(x, yWin, label="win_count")
# ax1.plot(x, yLose, label="lose_count")
ax1.plot(x, ySteps, label="steps")
ax2.plot(x, yWinRate, label="Win rate", color="red")

ax1.legend(loc='upper left')
ax2.legend(loc='lower right')

plt.show()


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