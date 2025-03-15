import random

import matplotlib
import matplotlib.pyplot as plt

from game import Game, StagHunt
from policy import BoltzmannPolicy, EpsilonGreedyPolicy, QLearningPolicy
from visualization import visualize_policy_traces

matplotlib.use("TkAgg")


def train(
    rewards_mat,
    qlearning: QLearningPolicy,
    episodes,
    learning_rate,
    save_interval,
    training_runs,
):
    histories = []
    q_values_p1 = []
    q_values_p2 = []
    for i in range(training_runs):
        print(f"Training run:{i}")
        action_counts_p1 = [0, 0]
        action_counts_p2 = [0, 0]
        q_val_p1 = [random.random() - 0.5, random.random() - 0.5]
        q_val_p2 = [random.random() - 0.5, random.random() - 0.5]
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
            if episode % 100 == 0:
                print(f"Episode {episode} of training run {i}")
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

        histories.append(history)
        q_values_p1.append(q_val_p1)
        q_values_p2.append(q_val_p2)

    return (q_values_p1, q_values_p2, histories)


def run_experiment(game: Game):
    iqlearning = EpsilonGreedyPolicy(0.1)
    qlearning = BoltzmannPolicy(0.3)
    q_val_p1, q_val_p2, history = train(
        rewards_mat=game.rewards,
        qlearning=qlearning,
        episodes=10000,
        learning_rate=0.01,
        save_interval=1,
        training_runs=20,
    )

    # visualize_training(
    #     histories=history, action_names=game.action_names, game_name=game.name
    # )

    visualize_policy_traces(history, game.name, game.action_names)

    print("\nFinal Q-values:")
    print(f"Player 1: {[q for q in q_val_p1]}")
    print(f"Player 2: {[q for q in q_val_p2]}")

    # final_policy_p1 = calculate_policy_from_q(q_val_p1, 0.1)
    # final_policy_p2 = calculate_policy_from_q(q_val_p2, 0.1)
    # print("\nFinal Policies (probability of selecting each action):")
    # print(f"Player 1: {[round(p, 3) for p in final_policy_p1]}")
    # print(f"Player 2: {[round(p, 3) for p in final_policy_p2]}")


if __name__ == "__main__":
    run_experiment(StagHunt())
    # stag_hunt()
    # mathching_pennies()
    plt.show()
    print("end")
