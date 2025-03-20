import random

import matplotlib
import matplotlib.pyplot as plt
from game import Game, MatchingPennies
from policy import BoltzmannPolicy, EpsilonGreedyPolicy, QLearningPolicy
from visualization import plot_vector_field, visualize_policy_traces

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
        lr = learning_rate
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
                lr = lr * 0.9
                print(f"Episode {episode} of training run {i}")
            action_p1 = qlearning.select_action(q_val_p1)
            action_p2 = qlearning.select_action(q_val_p2)
            action_counts_p1[action_p1] += 1
            action_counts_p2[action_p2] += 1

            reward_p1, reward_p2 = rewards_mat[action_p1][action_p2]

            q_val_p1[action_p1] += lr * (reward_p1 - q_val_p1[action_p1])
            q_val_p2[action_p2] += lr * (reward_p2 - q_val_p2[action_p2])

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
        episodes=1000,
        learning_rate=0.03,
        save_interval=1,
        training_runs=20,
    )

    fig, ax = plt.subplots(figsize=(12, 8))

    ax, q = plot_vector_field(
        game.rewards,
        grid_size=20,
        ax=ax,
        fig=fig,
    )

    ax = visualize_policy_traces(history, game.name, game.action_names, ax=ax)

    ax.legend()

    plt.tight_layout()
    plt.show()
    print("\nFinal Q-values:")
    print(f"Player 1: {[q for q in q_val_p1]}")
    print(f"Player 2: {[q for q in q_val_p2]}")


if __name__ == "__main__":
    # run_experiment(StagHunt())
    # run_experiment(PrisonersDilemma())
    run_experiment(MatchingPennies())
