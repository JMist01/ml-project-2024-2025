import random

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("TkAgg")
import numpy as np


def epsilon_q_learing_action(q_value, epsilon, type_of_problem):
    if random.random() < epsilon:
        action_p = random.randint(0, 1)
    else:
        if type_of_problem == "max":
            action_p = q_value.index(max(q_value))
        else:
            action_p = q_value.index(min(q_value))
    return action_p


def train(
    rewards_mat,
    type_of_problem,
    episodes=100,
    learning_rate=0.1,
    epsilon=0.3,
    save_interval=100,
):
    q_val_p1 = [0.0, 0.0]
    q_val_p2 = [0.0, 0.0]

    history = {
        "q_values_p1": [],
        "q_values_p2": [],
        "episodes": [],
    }

    for episode in range(episodes):
        action_p1 = epsilon_q_learing_action(q_val_p1, epsilon, type_of_problem)
        action_p2 = epsilon_q_learing_action(q_val_p2, epsilon, type_of_problem)

        reward_p1, reward_p2 = rewards_mat[action_p1][action_p2]

        # update step
        q_val_p1[action_p1] += learning_rate * (reward_p1 - q_val_p1[action_p1])
        q_val_p2[action_p2] += learning_rate * (reward_p2 - q_val_p2[action_p2])

        if episode % save_interval == 0 or episode == episodes - 1:
            history["episodes"].append(episode)
            history["q_values_p1"].append(q_val_p1.copy())
            history["q_values_p2"].append(q_val_p2.copy())

    return (q_val_p1, q_val_p2, history)


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
    # defect == testify -> if both testify -> each 3 years
    # Cooperate == both stay silent and don't talk to police -> each gets 1 year
    # TODO: It is rather a minimization problem, not maximization, so what do we do about that?
    rewards = [
        [(3, 3), (0, 5)],
        [(5, 0), (1, 1)],
    ]
    action_names = ["Cooperate", "Defect"]
    game_name = "Prisoner's Dilemma"
    type_of_problem = "min"
    run_experiment(rewards, action_names, game_name, type_of_problem)


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


def run_experiment(rewards, action_names, game_name, type_of_problem="max"):
    q_val_p1, q_val_p2, history = train(
        rewards,
        episodes=10000,
        learning_rate=0.1,
        epsilon=0.1,
        save_interval=10,
        type_of_problem=type_of_problem,
    )

    visualize_training(history=history, action_names=action_names, game_name=game_name)

    print("\nFinal Q-values:")
    print(f"Player 1: {[q for q in q_val_p1]}")
    print(f"Player 2: {[q for q in q_val_p2]}")


if __name__ == "__main__":
    # prisoners_dilema()
    stag_hunt()
    # mathching_pennies()
    plt.show()
