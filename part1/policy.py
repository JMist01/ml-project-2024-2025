import random
from abc import ABC, abstractmethod

import numpy as np


class QLearningPolicy(ABC):
    name: str

    @abstractmethod
    def _select_action(self, q_values) -> int:
        pass

    @abstractmethod
    def get_action_probabilities(self, q_values) -> list[int]:
        pass

    @abstractmethod
    def update_step(self, rewards, q_val_p1, q_val_p2):
        pass

    def restart(self):
        pass


class EpsilonGreedyPolicy(QLearningPolicy):
    name = "epsilon_greedy"

    def __init__(self, epsilon, lr):
        self.LR = lr
        self.epsilon = epsilon

    def _select_action(self, q_values):
        if random.random() < self.epsilon:
            return random.randint(0, 1)
        else:
            return int(np.argmax(q_values))

    def update_step(self, rewards, q_val_p1, q_val_p2):
        action_p1 = self._select_action(q_val_p1)
        action_p2 = self._select_action(q_val_p2)
        # TODO: there is some peroblem with choeing acton

        reward_p1, reward_p2 = rewards[action_p1][action_p2]

        q_val_p1[action_p1] += self.LR * (reward_p1 - q_val_p1[action_p1])
        q_val_p2[action_p2] += self.LR * (reward_p2 - q_val_p2[action_p2])

    # TODO: somre problem here with epsilon
    def get_action_probabilities(self, q_values):
        return q_values / np.sqrt(np.sum(np.array(q_values) ** 2))
        probabilities = [self.epsilon] * 2
        best_action = int(np.argmax(q_values))
        probabilities[best_action] = 1 - self.epsilon
        return probabilities


class BoltzmannPolicy(QLearningPolicy):
    name = "boltzmann"

    def __init__(self, temperature, lr):
        self.TEMPERATURE = temperature
        self.LR = lr

    def _boltzmann_probs(self, q_values):
        q_values = np.array(q_values)
        exp_values = np.exp(q_values / self.TEMPERATURE)
        return exp_values / np.sum(exp_values)

    def _select_action(self, q_values):
        probs = self._boltzmann_probs(q_values)
        return np.random.choice(len(q_values), p=probs)

    def get_action_probabilities(self, q_values):
        return self._boltzmann_probs(q_values).tolist()

    def update_step(self, rewards, q_val_p1, q_val_p2):
        action_p1 = self._select_action(q_val_p1)
        action_p2 = self._select_action(q_val_p2)

        reward_p1, reward_p2 = rewards[action_p1][action_p2]

        q_val_p1[action_p1] += self.LR * (reward_p1 - q_val_p1[action_p1])
        q_val_p2[action_p2] += self.LR * (reward_p2 - q_val_p2[action_p2])


class LenientBoltzmannPolicy(BoltzmannPolicy):
    name = "lenient_boltzmann"

    def __init__(self, temperature, lr) -> None:
        super().__init__(temperature, lr)
        self.K = 2

    def update_step(self, rewards, q_val_p1, q_val_p2):
        for action_p1 in range(2):
            max_reward_p1 = float("-inf")
            for _ in range(self.K):
                action_p2 = self._select_action(q_val_p2)
                reward_p1 = rewards[action_p1][action_p2][0]

                max_reward_p1 = max(max_reward_p1, reward_p1)

            q_val_p1[action_p1] += self.LR * (max_reward_p1 - q_val_p1[action_p1])

        for action_p2 in range(2):
            max_reward_p2 = float("-inf")
            for _ in range(self.K):
                action_p1 = self._select_action(q_val_p1)
                reward_p2 = rewards[action_p1][action_p2][1]
                max_reward_p2 = max(max_reward_p2, reward_p2)

            q_val_p2[action_p2] += self.LR * (max_reward_p2 - q_val_p2[action_p2])
