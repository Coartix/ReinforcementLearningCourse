import numpy as np
import matplotlib.pyplot as plt

def simulate(MAB, T, N):
    """
    Simulate a MAB experiment
    Args:
        MAB (list): list of arms
        T (int): number of all patients
        N (int): number of test patients
    
    Returns:
        (arm index, score): best arm index and score
    """
    # Training phase
    patients_per_arm = N//len(MAB)
    rewards = []
    for t in range(len(MAB)):
        score = 0 
        for _ in range(patients_per_arm):
            score += int(MAB[t].sample())
        rewards.append(score)
    best_arm_idx = np.argmax(rewards)
    score = rewards[best_arm_idx]
    # Exploitation phase
    for _ in range(T-N):
        score += int(MAB[best_arm_idx].sample())
    return best_arm_idx, score

def regret(MAB, T, N):
    best_arm_mean = np.max([a.mean for a in MAB])
    sim = 100
    score = 0
    for _ in range(sim):
        _, reward = simulate(MAB, T, N)
        score += reward
    return T*best_arm_mean - score/sim

def visualize_regret(MAB, T, N):
    records = []
    for t in range(1, T, 5):
        records.append((t,np.sum([regret(MAB, t, N) for _ in range(50)])/50))
    plt.plot(*zip(*records))
    plt.axvline(x=N, color='r', linestyle='--', label='N')
    plt.legend(loc='upper left', frameon=False)
    plt.xlabel("t")
    plt.ylabel("Regret")
    plt.show()