"""
Dans ce TP, nous allons implémenter un agent qui apprend à jouer au jeu Taxi-v3
de OpenAI Gym. Le but du jeu est de déposer un passager à une destination
spécifique en un minimum de temps. Le jeu est composé d'une grille de 5x5 cases
et le taxi peut se déplacer dans les 4 directions (haut, bas, gauche, droite).
Le taxi peut prendre un passager sur une case spécifique et le déposer à une
destination spécifique. Le jeu est terminé lorsque le passager est déposé à la
destination. Le jeu est aussi terminé si le taxi prend plus de 200 actions.

Vous devez implémenter un agent qui apprend à jouer à ce jeu en utilisant
les algorithmes Q-Learning et SARSA.

Pour chaque algorithme, vous devez réaliser une vidéo pour montrer que votre modèle fonctionne.
Vous devez aussi comparer l'efficacité des deux algorithmes en termes de temps
d'apprentissage et de performance.

A la fin, vous devez rendre un rapport qui explique vos choix d'implémentation
et vos résultats (max 1 page).
"""

import typing as t
import gymnasium as gym
import numpy as np
import os

from qlearning import QLearningAgent
from qlearning_eps_scheduling import QLearningAgentEpsScheduling
from sarsa import SarsaAgent
from gym.wrappers.monitoring.video_recorder import VideoRecorder

env = gym.make("Taxi-v3", render_mode="rgb_array")
n_actions = env.action_space.n

records = {
    'qlearning': {
        'best_mean_reward': 0,
        'best_i': 0
    },
    'qlearning_eps_scheduling': {
        'best_mean_reward': 0,
        'best_i': 0
    },
    'sarsa': {
        'best_mean_reward': 0,
        'best_i': 0
    }
}

#################################################
# 1. video tools
#################################################
video = True

# Define a function to capture videos during training
def create_video(agent, env, iteration, folder="./videos"):
    video_recorder = VideoRecorder(env, path=f"{folder}/training_episode_{iteration}.mp4")
    s, _ = env.reset()
    video_recorder.capture_frame()
    
    done = False
    epoch = 0
    while not done:
        dict_action = agent._qvalues[s]
        action = np.argmax(list(dict_action.values()))
        s, _, done, _, _ = env.step(action)
        video_recorder.capture_frame()
        epoch += 1
    video_recorder.close()



#################################################
# 1. Play with QLearningAgent
#################################################

agent = QLearningAgent(
    learning_rate=0.1, epsilon=0.05, gamma=0.9, legal_actions=list(range(n_actions))
)

def play_and_train(env: gym.Env, agent: QLearningAgent, t_max=int(1e8)) -> float:
    """
    This function should
    - run a full game, actions given by agent.getAction(s)
    - train agent using agent.update(...) whenever possible
    - return total rewardb
    """
    total_reward: t.SupportsFloat = 0.0
    s, _ = env.reset()
    done = False

    while not done:
        a = agent.get_action(s)
        next_s, r, done, _, _ = env.step(a)

        # Train agent for state s
        agent.update(s, a, r, next_s)
        s = next_s

        total_reward += r
    return total_reward


rewards = []
best_mean_reward = 0
best_i = 0
for i in range(1, 10001):
    rewards.append(play_and_train(env, agent))
    if i % 100 == 0:
        os.system("clear")
        print(f"Episode {i}")
        mean_reward = np.mean(rewards[-100:])
        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            best_i = i
        print("Mean reward qlearn", mean_reward)
    
    records['qlearning']['best_mean_reward'] = best_mean_reward
    records['qlearning']['best_i'] = best_i

assert np.mean(rewards[-100:]) > 0.0


"""Evaluate agent's performance after Q-learning"""

total_epochs, total_penalties = 0, 0
episodes = 100

for _ in range(episodes):
    state, _ = env.reset()
    epochs, penalties, reward = 0, 0, 0
    
    done = False
    
    while not done: # and epochs < 500:
        dict_action = agent._qvalues[state]
        action = np.argmax(list(dict_action.values()))
        state, reward, done, info, _ = env.step(action)

        if reward == -10:
            penalties += 1

        epochs += 1

    total_penalties += penalties
    total_epochs += epochs

print(f"Evaluating Agent, results after {episodes} episodes:")
print(f"Average epochs per episode: {total_epochs / episodes}")
print(f"Average penalties per episode: {total_penalties / episodes}")


# Create video for QLearningAgent
if video:
    create_video(agent=agent, folder="./videos/qlearning", env=env, iteration=5000)


#################################################
# 2. Play with QLearningAgentEpsScheduling
#################################################

agent = QLearningAgentEpsScheduling(
    learning_rate=0.05, epsilon=0.25, gamma=0.99, legal_actions=list(range(n_actions))
)

rewards = []
best_mean_reward = 0
best_i = 0
for i in range(1, 10001):
    rewards.append(play_and_train(env, agent))
    if i % 100 == 0:
        os.system("clear")
        print(f"Episode {i}")
        mean_reward = np.mean(rewards[-100:])
        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            best_i = i
        print("Mean reward qlearn", mean_reward)
    
    records['qlearning_eps_scheduling']['best_mean_reward'] = best_mean_reward
    records['qlearning_eps_scheduling']['best_i'] = best_i

assert np.mean(rewards[-100:]) > 0.0

# Create video for QLearningAgentEpsScheduling
if video:
    create_video(agent=agent, folder="./videos/qlearning_eps_scheduling", env=env, iteration=5000)




####################
# 3. Play with SARSA
####################


agent = SarsaAgent(learning_rate=0.05, gamma=0.99, legal_actions=list(range(n_actions)))

rewards = []
best_mean_reward = 0
best_i = 0
for i in range(1, 10001):
    rewards.append(play_and_train(env, agent))
    if i % 100 == 0:
        os.system("clear")
        print(f"Episode {i}")
        mean_reward = np.mean(rewards[-100:])
        if mean_reward > best_mean_reward:
            best_mean_reward = mean_reward
            best_i = i
        print("Mean reward qlearn", mean_reward)
    
    records['sarsa']['best_mean_reward'] = best_mean_reward
    records['sarsa']['best_i'] = best_i

# Create video for SarsaAgent
if video:
    create_video(agent=agent, folder="./videos/sarsa", env=env, iteration=5000)

print(records)