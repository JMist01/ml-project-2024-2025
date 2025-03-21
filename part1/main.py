import random

import matplotlib
import matplotlib.pyplot as plt
from game import *
from policy import (
    BoltzmannPolicy,
    EpsilonGreedyPolicy,
    LenientBoltzmannPolicy,
    QLearningPolicy,
)
from visualization import configure_figure, plot_vector_field, visualize_policy_traces

matplotlib.use("TkAgg")


def train(
    rewards_mat,
    qlearning: QLearningPolicy,
    episodes,
    # learning_rate,
    training_runs,
):
    histories = []
    q_values_p1 = []
    q_values_p2 = []

    for i in range(training_runs):
        qlearning.restart()
        # lr = learning_rate
        print(f"Training run: {i}")
        q_val_p1 = [random.random(), random.random()]
        q_val_p2 = [random.random(), random.random()]
        history = {
            "policy_p1": [],  # Probabilities of selecting each action
            "policy_p2": [],
        }

        for episode in range(episodes):
            # if episode % 100 == 0:
            # print(f"Episode {episode} of training run {i}")
            # action_p1 = qlearning._select_action(q_val_p1)
            # action_p2 = qlearning._select_action(q_val_p2)
            #
            # reward_p1, reward_p2 = rewards_mat[action_p1][action_p2]
            #
            # q_val_p1[action_p1] += lr * (reward_p1 - q_val_p1[action_p1])
            # q_val_p2[action_p2] += lr * (reward_p2 - q_val_p2[action_p2])
            qlearning.update_step(rewards_mat, q_val_p1, q_val_p2)

            if episode % 1 == 0 or episode == episodes - 1:
                policy_p1 = qlearning.get_action_probabilities(q_val_p1)
                policy_p2 = qlearning.get_action_probabilities(q_val_p2)
                history["policy_p1"].append(policy_p1)
                history["policy_p2"].append(policy_p2)

        histories.append(history)
        q_values_p1.append(q_val_p1)
        q_values_p2.append(q_val_p2)

    return (q_values_p1, q_values_p2, histories)


def run_experiment(game: Game, qlearning: QLearningPolicy):
    q_val_p1, q_val_p2, history = train(
        rewards_mat=game.rewards,
        qlearning=qlearning,
        episodes=1000,
        training_runs=20,
    )

    _, ax = plt.subplots(figsize=(12, 8))

    plot_vector_field(qlearning, game.rewards, ax=ax, grid_size=20)

    visualize_policy_traces(history, ax=ax)
    configure_figure(qlearning, game, ax)
    # print("\nFinal Q-values:")
    # print(f"Player 1: {[q for q in q_val_p1]}")
    # print(f"Player 2: {[q for q in q_val_p2]}")


def run_all_policies(game: Game, lr):
    run_experiment(game, EpsilonGreedyPolicy(0.2, lr))
    run_experiment(game, BoltzmannPolicy(0.3, lr))
    run_experiment(game, LenientBoltzmannPolicy(0.3, lr))


def run_all_games(lr):
    run_all_policies(StagHunt(), lr)
    run_all_policies(PrisonersDilemma(), lr)
    run_all_policies(MatchingPennies(), lr)
    run_all_policies(SubsidyGame(), lr)


if __name__ == "__main__":
    lr = 0.01
    # run_all_games(lr)
    learning = EpsilonGreedyPolicy(0.2, lr)
    qlearning = BoltzmannPolicy(0.3, lr)
    learning = LenientBoltzmannPolicy(0.3, lr)
    run_experiment(StagHunt(), qlearning)
    # run_experiment(PrisonersDilemma())
    # run_experiment(MatchingPennies())
    # run_experiment(SubsidyGame())
    # plt.show()
    plt.show()
