import gymnasium as gym
import random
import numpy as np
from tqdm import tqdm


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
    
    # episode = -1
    # while(win_count < 100):
    #     episode+=1

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

    return win_count/max_episode


max_episode = 1000
# max_episode = 10
max_steps = 99

learning_rate = 0.9
discount_rate = 0.8

max_epsilon = 1.0
min_epsilon = 0.05

decay_rate = 0.005

epsilon = max_epsilon


training_number = 250

metrics = {
    "rewards": np.array([]),
    "win_rate": np.array([])
}

for t in range(training_number):
    seed = random.randint(0, 1000)
    np.random.seed(seed)

    env = gym.make("Taxi-v3")
    env.reset()

    rewards = []
    win_count = []

    qtable = define_QTable(env)

    win_rate = train(env, qtable, epsilon, max_epsilon, min_epsilon, learning_rate, discount_rate, decay_rate, max_episode, max_steps)
    metrics["win_rate"] = np.append(metrics["win_rate"], win_rate)
    env.close()

    done = False
    total_reward = 0

    # env = gym.make("Taxi-v3", render_mode="human")
    state, _ = env.reset(seed=seed)

    for s in range(max_steps):
        action = np.argmax(qtable[state,:])
        new_state, reward, done, info, _ = env.step(action)
        total_reward += reward
        # env.render()
        
        state = new_state

        if done == True:
            break


    # watch trained agent
    state, _ = env.reset(seed=seed)

    print(t, "---> reward :", total_reward)
    metrics["rewards"] = np.append(metrics["rewards"], total_reward)
    # metrics["rewards"].append(total_reward)

print(np.mean(metrics["rewards"]))
print(np.mean(metrics["win_rate"]))

env.close()
