import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
import seaborn as sns

class ThompsonSampling:
    '''
    Thompson Sampling algorithm for Bernoulli bandits   
    '''
    def __init__(self, num_arms):
        self.num_arms = num_arms
        self.alpha = np.ones(num_arms)
        self.beta = np.ones(num_arms)
        
    def select_arm(self):
        sampled_means = np.random.beta(self.alpha, self.beta)
        chosen_arm = np.argmax(sampled_means)
        return chosen_arm

    def update(self, arm, reward):
        if reward:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1

def simulate_thomson(MAB, T):
    '''
    Simulate the Thompson Sampling algorithm on a Bernoulli MAB

    Parameters
    ----------
    MAB : list of ArmBernoulli
        The list of arms to use for the simulation
    T : int
        The number of rounds to simulate

    Returns
    -------
    total_rewards : int
        The total rewards of the simulation
    '''
    thompson_sampler = ThompsonSampling(len(MAB))

    total_rewards = 0
    for t in range(T):
        chosen_arm = thompson_sampler.select_arm()
        reward = MAB[chosen_arm].sample()
        thompson_sampler.update(chosen_arm, reward)
        total_rewards += reward
    return total_rewards

def regret_thomson(MAB, T, nb_sim=20):
    '''
    Compute the regret of the Thompson Sampling algorithm on a Bernoulli MAB

    Parameters
    ----------
    MAB : list of ArmBernoulli
        The list of arms to use for the simulation
    T : int
        The number of rounds to simulate
    nb_sim : int
        The number of simulations to run to estimate the regret

    Returns
    -------
    regret : float
        The regret of the Thompson Sampling algorithm on the MAB
    '''
    best_arm_mean = np.max([a.mean for a in MAB])
    score = 0
    for _ in range(nb_sim):
        score += simulate_thomson(MAB, T)
    return T*best_arm_mean - score/nb_sim

def visualize_regret_thomson(MAB, T, nb_sim=20):
    '''
    Visualize the regret of the Thompson Sampling algorithm on a Bernoulli MAB

    Parameters
    ----------
    MAB : list of ArmBernoulli
        The list of arms to use for the simulation
    T : int
        The number of rounds to simulate
    nb_sim : int
        The number of simulations to run to estimate the regret
    '''
    records = []
    for t in range(1, T, 5):
        records.append((t,np.sum([regret_thomson(MAB, t) for _ in range(nb_sim)])/nb_sim))
    plt.plot(*zip(*records))
    plt.xlabel("t")
    plt.ylabel("Regret")
    plt.show()

def plot_beta_distributions(alpha_history, beta_history):
    '''
    Plot the beta distributions of the arms over time (patients)

    Parameters
    ----------
    alpha_history : list of float
        The list of alpha parameters over time
    beta_history : list of float
        The list of beta parameters over time
    '''
    x = np.linspace(0, 1, 100)
    plt.figure(figsize=(10, 6))
    for a, b, i in zip(alpha_history, beta_history, np.linspace(1, 5, len(alpha_history))):
        pdf = beta.pdf(x, a, b)
        plt.plot(x, pdf, label=f'Fonction de densité Beta pour le vaccin {int(i)}')
    plt.title('Évolution des distributions Beta')
    plt.xlabel('Probabilité de succès')
    plt.ylabel('Densité de probabilité')
    plt.legend()
    plt.show()

def visualize_beta_probabilities(MAB, T):
    '''
    Visualize the beta probabilities of the arms over time (patients)

    Parameters
    ----------
    MAB : list of ArmBernoulli
        The list of arms to use for the simulation
    T : int
        The number of rounds to simulate
    '''
    beta_probabilities = [[] for _ in range(len(MAB))]

    for t in range(T):
        thompson_sampler = ThompsonSampling(len(MAB))
        chosen_arm = thompson_sampler.select_arm()
        reward = MAB[chosen_arm].sample()
        thompson_sampler.update(chosen_arm, reward)

        for arm in range(len(MAB)):
            beta_probabilities[arm].append(np.random.beta(thompson_sampler.alpha[arm], thompson_sampler.beta[arm], 100).mean())
        
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=beta_probabilities)
    plt.xlabel("Arm")
    plt.ylabel("Beta Probabilities")
    plt.title("Beta Probabilities for Arms Over Time (patients -> t)")
    plt.xticks(np.arange(len(MAB)), [f'Arm {i}' for i in range(len(MAB))])
    plt.show()