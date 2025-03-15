import random
from abc import ABC, abstractmethod

import numpy as np


class QLearningPolicy(ABC):
    @abstractmethod
    def select_action(self, q_values) -> int:
        """Return an action based on the provided q_values."""
        pass

    @abstractmethod
    def get_action_probabilities(self, q_values) -> list[int]:
        """Return the action probabilities based on the provided q_values."""
        pass


class EpsilonGreedyPolicy(QLearningPolicy):
    def __init__(self, epsilon):
        self.epsilon = epsilon

    def select_action(self, q_values):
        if random.random() < self.epsilon:
            return random.randint(0, len(q_values) - 1)
        else:
            return int(np.argmax(q_values))

    def get_action_probabilities(self, q_values):
        num_actions = len(q_values)
        probabilities = [self.epsilon / num_actions] * num_actions
        best_action = int(np.argmax(q_values))
        probabilities[best_action] += 1 - self.epsilon
        return probabilities


class BoltzmannPolicy(QLearningPolicy):
    def __init__(self, temperature):
        self.temperature = temperature

    def _boltzmann_probs(self, q_values):
        q_values = np.array(q_values)
        exp_values = np.exp(q_values / self.temperature)
        return exp_values / np.sum(exp_values)

    def select_action(self, q_values):
        probs = self._boltzmann_probs(q_values)
        return np.random.choice(len(q_values), p=probs)

    def get_action_probabilities(self, q_values):
        return self._boltzmann_probs(q_values).tolist()
