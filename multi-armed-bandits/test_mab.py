import multiarmbandits as mab
import numpy as np
import pytest


# Create a bandit with 5 arms
K = 5
T = 100
np.random.seed(1)
means = np.random.random(K)
MAB = [mab.ArmBernoulli(m) for m in means]

def test_init():
    assert(len(MAB) == 5)

@pytest.fixture
def naive_simulation():
    return [(1, 65), (1,67)]

def test_naive(naive_simulation):
    sim1 = mab.simulate(MAB, T, 20)
    sim2 = mab.simulate(MAB, T, 10)
    assert([sim1, sim2] == naive_simulation)

@pytest.fixture
def UCB_simulation():
    return 1,72

def test_UCB(UCB_simulation):
    sim = mab.simulate_UCB(MAB, T)
    assert(sim == UCB_simulation)

@pytest.fixture
def thomson_simulation():
    return 58

def test_thomson(thomson_simulation):
    sim = mab.simulate_thomson(MAB, T)
    assert(sim == thomson_simulation)


@pytest.fixture
def naive_regret():
    return 21.242449344215807

def test_naive_regret(naive_regret):
    reg = mab.regret(MAB, T, 20)
    assert(reg == naive_regret)

@pytest.fixture
def UCB_regret():
    return 15.252449344215805

def test_UCB_regret(UCB_regret):
    reg = mab.regret_UCB(MAB, T)
    assert(reg == UCB_regret)

@pytest.fixture
def thomson_regret():
    return 12.832449344215803

def test_thomson_regret(thomson_regret):
    reg = mab.regret_thomson(MAB, T)
    assert(reg == thomson_regret)