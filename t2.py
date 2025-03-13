import random
from abc import ABC, abstractmethod

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("TkAgg")


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


def train(
    rewards_mat,
    qlearning: QLearningPolicy,
    episodes=100,
    learning_rate=0.6,
    save_interval=100,
):
    q_val_p1 = [0.0, 0.0]
    q_val_p2 = [0.0, 0.0]

    action_counts_p1 = [0, 0]
    action_counts_p2 = [0, 0]

    history = {
        "q_values_p1": [],
        "q_values_p2": [],
        "policy_p1": [],  # Probabilities of selecting each action
        "policy_p2": [],
        "empirical_p1": [],  # Empirical frequencies of actions
        "empirical_p2": [],
        "episodes": [],
    }

    for episode in range(episodes):
        action_p1 = qlearning.select_action(q_val_p1)
        action_p2 = qlearning.select_action(q_val_p2)
        action_counts_p1[action_p1] += 1
        action_counts_p2[action_p2] += 1

        reward_p1, reward_p2 = rewards_mat[action_p1][action_p2]

        q_val_p1[action_p1] += learning_rate * (reward_p1 - q_val_p1[action_p1])
        q_val_p2[action_p2] += learning_rate * (reward_p2 - q_val_p2[action_p2])

        if episode % save_interval == 0 or episode == episodes - 1:
            history["episodes"].append(episode)
            history["q_values_p1"].append(q_val_p1.copy())
            history["q_values_p2"].append(q_val_p2.copy())

            policy_p1 = qlearning.get_action_probabilities(q_val_p1)
            policy_p2 = qlearning.get_action_probabilities(q_val_p2)

            history["policy_p1"].append(policy_p1)
            history["policy_p2"].append(policy_p2)

            total_p1 = sum(action_counts_p1)
            total_p2 = sum(action_counts_p2)
            empirical_p1 = [count / total_p1 for count in action_counts_p1]
            empirical_p2 = [count / total_p2 for count in action_counts_p2]

            history["empirical_p1"].append(empirical_p1)
            history["empirical_p2"].append(empirical_p2)

    return (q_val_p1, q_val_p2, history)


def visualize_policy_traces(history, game_name, action_names):
    import matplotlib.pyplot as plt

    p1_policies = history["policy_p1"]
    p2_policies = history["policy_p2"]

    plt.figure(figsize=(10, 8))

    p1_probs = [policy[0] for policy in p1_policies]
    p2_probs = [policy[0] for policy in p2_policies]

    plt.plot(p1_probs, p2_probs, "k-", linewidth=2, label="Learning trajectory")
    plt.scatter(
        p1_probs, p2_probs, c=range(len(p1_probs)), cmap="viridis", s=50, label="Steps"
    )

    plt.xlabel(f"P1 probability of selecting action {action_names[0]}")
    plt.ylabel(f"P2 probability of selecting action {action_names[0]}")
    plt.title(f"Empirical policy traces {game_name}")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.grid(True)
    plt.legend()
    plt.show()


def visualize_training(history, action_names, game_name):
    episodes = history["episodes"]
    q_values_p1 = np.array(history["q_values_p1"])
    q_values_p2 = np.array(history["q_values_p2"])

    plt.figure(figsize=(15, 12))

    plt.subplot(2, 1, 1)
    for i in range(2):
        plt.plot(episodes, q_values_p1[:, i], label=action_names[i])
    plt.title(f"Player 1 Q-Values Over Time ({game_name})")
    plt.xlabel("Episodes")
    plt.ylabel("Q-Value")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(2, 1, 2)
    for i in range(2):
        plt.plot(episodes, q_values_p2[:, i], label=action_names[i])
    plt.title(f"Player 2 Q-Values Over Time ({game_name})")
    plt.xlabel("Episodes")
    plt.ylabel("Q-Value")
    plt.legend()
    plt.grid(True, alpha=0.3)

    final_diff_p1 = q_values_p1[-1, 1] - q_values_p1[-1, 0]
    final_diff_p2 = q_values_p2[-1, 1] - q_values_p2[-1, 0]

    print("Final Q-value differences :" + str(action_names))
    print(f"Player 1: {final_diff_p1:.4f}")
    print(f"Player 2: {final_diff_p2:.4f}")


def prisoners_dilema():
    # defect == testify -> if both testify
    # Cooperate == both stay silent and don't talk to police
    rewards = [
        [(3, 3), (0, 5)],
        [(5, 0), (1, 1)],
    ]
    action_names = ["Cooperate", "Defect"]
    game_name = "Prisoner's Dilemma"
    run_experiment(rewards, action_names, game_name)


def stag_hunt():
    rewards = [
        [(1, 1), (0, 2 / 3)],
        [(2 / 3, 0), (2 / 3, 2 / 3)],
    ]
    print(rewards)
    action_names = ["Stag", "Hare"]
    game_name = "Stag Hunt"
    run_experiment(rewards, action_names, game_name)


def mathching_pennies():
    rewards = [
        [(0, 1), (0, 1)],
        [(1, 0), (0, 1)],
    ]
    action_names = ["Heads", "Tails"]
    game_name = "Matching Pennies"
    run_experiment(rewards, action_names, game_name)


def subsidy_game():
    rewards = [
        [(12, 12), (0, 11)],
        [(11, 0), (10, 10)],
    ]
    action_names = ["S1", "S2"]
    game_name = "Subsidy Game"
    run_experiment(rewards, action_names, game_name)


def run_experiment(rewards, action_names, game_name):
    # qlearning = EpsilonGreedyPolicy(0.1)
    qlearning = BoltzmannPolicy(0.3)
    q_val_p1, q_val_p2, history = train(
        rewards_mat=rewards,
        qlearning=qlearning,
        episodes=1000,
        learning_rate=0.1,
        save_interval=10,
    )

    visualize_training(history=history, action_names=action_names, game_name=game_name)

    visualize_policy_traces(history, game_name, action_names)

    print("\nFinal Q-values:")
    print(f"Player 1: {[q for q in q_val_p1]}")
    print(f"Player 2: {[q for q in q_val_p2]}")

    # final_policy_p1 = calculate_policy_from_q(q_val_p1, 0.1)
    # final_policy_p2 = calculate_policy_from_q(q_val_p2, 0.1)
    # print("\nFinal Policies (probability of selecting each action):")
    # print(f"Player 1: {[round(p, 3) for p in final_policy_p1]}")
    # print(f"Player 2: {[round(p, 3) for p in final_policy_p2]}")


if __name__ == "__main__":
    prisoners_dilema()
    # stag_hunt()
    # mathching_pennies()
    plt.show()
    print("end")
