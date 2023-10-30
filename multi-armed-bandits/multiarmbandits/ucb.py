import numpy as np
import math
import matplotlib.pyplot as plt

def simulate_UCB(MAB, T):
    """
    Simulate a MAB experiment
    Args:
        MAB (list): list of arms
        T (int): number of rounds
    Returns:
        (arm index, score)
    """
    X = [0 for _ in range(len(MAB))]
    B = [math.inf for _ in range(len(MAB))]
    count_arms_chosen = [0 for _ in range(len(MAB))]
    score = 0
    for t in range(1,T):
        arm = np.argmax([X[i] + B[i] for i in range(len(MAB))])
        count_arms_chosen[arm] += 1
        B[arm] = np.sqrt(2*np.log(t)/count_arms_chosen[arm])
        result = int(MAB[arm].sample())
        score += result
        X[arm] = X[arm] + result/count_arms_chosen[arm]
    return np.argmax(X), score

def regret_UCB(MAB, T, nb_sim=50):
    """
    Compute the regret of the UCB algorithm on a Bernoulli MAB
    Args:
        MAB (list): list of arms
        T (int): number of rounds
        nb_sim (int): number of simulations
    Returns:
        (float): regret
    """
    best_arm_mean = np.max([a.mean for a in MAB])
    score = 0
    for _ in range(nb_sim):
        _, reward = simulate_UCB(MAB, T)
        score += reward
    return T*best_arm_mean - score/nb_sim

def visualize_regret_UCB(MAB, T, nb_sim=20):
    """
    Visualize the regret of the UCB algorithm on a Bernoulli MAB
    Args:
        MAB (list): list of arms
        T (int): number of rounds
        nb_sim (int): number of simulations
    """
    records = []
    for t in range(1, T, 5):
        records.append((t,np.sum([regret_UCB(MAB, t) for _ in range(nb_sim)])/nb_sim))
    plt.plot(*zip(*records))
    plt.xlabel("t")
    plt.ylabel("Regret")
    plt.show()

def plot_UCB_values(MAB, T, arm_idx):
    """
    Plot the evolution of X and B for a given arm
    Args:
        MAB (list): list of arms
        T (int): number of rounds
        arm_idx (int): index of the arm to plot
    """
    X = [0 for _ in range(len(MAB))]
    x = []
    B = [math.inf for _ in range(len(MAB))]
    b = []
    count_arms_chosen = [0 for _ in range(len(MAB))]
    score = 0
    for t in range(1,T):
        arm = np.argmax([X[i] + B[i] for i in range(len(MAB))])
        count_arms_chosen[arm] += 1
        B[arm] = np.sqrt(2*np.log(t)/count_arms_chosen[arm])
        result = int(MAB[arm].sample())
        score += result
        X[arm] = X[arm] + result/count_arms_chosen[arm]
        if arm == arm_idx:
            x.append((t,X[arm]))
            b.append((t,B[arm]))
    
    # Plot x and b side to side for arm_idx
    plt.plot(*zip(*x), label="X", color='r', linestyle='dotted')
    plt.plot(*zip(*b), label="B", color='b', linestyle='dotted')
    plt.legend(loc='upper right', frameon=False)
    plt.title(f"X and B evolution for arm {arm_idx}")
    plt.xlabel("t")
    plt.ylabel("X and B")
    plt.show()