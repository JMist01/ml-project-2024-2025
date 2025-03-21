import random
from abc import ABC, abstractmethod

import numpy as np


class QLearningPolicy(ABC):
    @abstractmethod
    def _select_action(self, q_values) -> int:
        pass

    @abstractmethod
    def get_action_probabilities(self, q_values) -> list[int]:
        pass

    @abstractmethod
    def update_step(self, rewards, q_val_p1, q_val_p2, lr):
        pass

    def restart(self):
        pass


class EpsilonGreedyPolicy(QLearningPolicy):
    def __init__(self, epsilon):
        self.epsilon = epsilon
        self.temperature = 1

    def _select_action(self, q_values):
        if random.random() < self.epsilon:
            return random.randint(0, 1)
        else:
            return int(np.argmax(q_values))

    def update_step(self, rewards, q_val_p1, q_val_p2, lr):
        action_p1 = self._select_action(q_val_p1)
        action_p2 = self._select_action(q_val_p2)
        # TODO: there is some peroblem with choeing acton

        reward_p1, reward_p2 = rewards[action_p1][action_p2]

        q_val_p1[action_p1] += lr * (reward_p1 - q_val_p1[action_p1])
        q_val_p2[action_p2] += lr * (reward_p2 - q_val_p2[action_p2])

    # TODO: somre problem here with epsilon
    def get_action_probabilities(self, q_values):
        return q_values / np.sqrt(np.sum(q_values[0] ** 2 + q_values[1] ** 2))
        probabilities = [self.epsilon] * 2
        best_action = int(np.argmax(q_values))
        probabilities[best_action] = 1 - self.epsilon
        return probabilities


class BoltzmannPolicy(QLearningPolicy):
    def __init__(self, temperature):
        self.temperature = temperature

    def _boltzmann_probs(self, q_values):
        q_values = np.array(q_values)
        exp_values = np.exp(q_values / self.temperature)
        return exp_values / np.sum(exp_values)

    def _select_action(self, q_values):
        probs = self._boltzmann_probs(q_values)
        return np.random.choice(len(q_values), p=probs)

    def get_action_probabilities(self, q_values):
        return self._boltzmann_probs(q_values).tolist()

    def update_step(self, rewards, q_val_p1, q_val_p2, lr):
        action_p1 = self._select_action(q_val_p1)
        action_p2 = self._select_action(q_val_p2)

        reward_p1, reward_p2 = rewards[action_p1][action_p2]

        # TODO: is this correct way to implment Q_leraning?
        q_val_p1[action_p1] += lr * (reward_p1 - q_val_p1[action_p1])
        q_val_p2[action_p2] += lr * (reward_p2 - q_val_p2[action_p2])


class LenientBoltzmannPolicy(QLearningPolicy):
    def __init__(self) -> None:
        self.K = 25  # TODO: ite seems taht K does not affect anything... -> look at the paper
        self.BETA = 0.9
        self.INIT_TEMPT = 3
        self.temps = np.ones((2, 2)) * self.INIT_TEMPT
        self.temperature = 0.3

    def restart(self):
        self.temps = np.ones((2, 2)) * self.INIT_TEMPT

    def _get_leniency(self, action1, action2):
        leniency = 1 - np.exp(-self.K * self.temps[action1][action2])
        self.temps[action1][action2] *= self.BETA
        return leniency

    def _boltzmann_probs(self, q_values):
        q_values = np.array(q_values)
        exp_values = np.exp(q_values / self.temperature)
        return exp_values / np.sum(exp_values)

    def _select_action(self, q_values):
        probs = self._boltzmann_probs(q_values)
        return np.random.choice(len(q_values), p=probs)

    def get_action_probabilities(self, q_values):
        return self._boltzmann_probs(q_values).tolist()

    def update_step(self, rewards, q_val_p1, q_val_p2, lr):
        action_p1 = self._select_action(q_val_p1)
        action_p2 = self._select_action(q_val_p2)
        reward_p1, reward_p2 = rewards[action_p1][action_p2]

        leniency = self._get_leniency(action_p1, action_p2)
        x = random.random()

        # TODO: mb use the same q-valeu for each player
        td_error_p1 = reward_p1 - q_val_p1[action_p1]
        # Apply leniency
        if not (x < leniency and td_error_p1 <= 0):
            q_val_p1[action_p1] += lr * td_error_p1
        else:
            print("lency not app,iled")

        td_error_p2 = reward_p2 - q_val_p2[action_p2]
        # Apply leniency
        if not (x < leniency and td_error_p2 <= 0):
            q_val_p2[action_p2] += lr * td_error_p2
