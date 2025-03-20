import matplotlib.pyplot as plt
import numpy as np
from replicator_dynamic import compute_vector_field


def visualize_policy_traces(histories, ax):
    labels = [f"Run {i + 1}" for i in range(len(histories))]
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i / len(histories)) for i in range(len(histories))]
    for i, history in enumerate(histories):
        p1_policies = history["policy_p1"]
        p2_policies = history["policy_p2"]
        p1_probs = [policy[0] for policy in p1_policies]
        p2_probs = [policy[0] for policy in p2_policies]
        ax.plot(p1_probs, p2_probs, linewidth=2, color=colors[i], label=labels[i])


def plot_vector_field(policy, rewards, ax, grid_size=20):
    X, Y, U, V = compute_vector_field(policy, rewards, grid_size)
    magnitudes = np.sqrt(U**2 + V**2)

    ax.quiver(X, Y, U, V, magnitudes, width=0.002, pivot="tail")


# def visualize_training(histories, action_names, game_name):
#     plt.figure(figsize=(15, 12))
#     labels = [f"Run {i + 1}" for i in range(len(histories))]
#     cmap = plt.get_cmap("tab10")
#     colors = [cmap(i / len(histories)) for i in range(len(histories))]
#
#     # Player 1 Q-values
#     plt.subplot(2, 1, 1)
#     for run_idx, history in enumerate(histories):
#         episodes = history["episodes"]
#         q_values_p1 = np.array(history["q_values_p1"])
#
#         for i in range(2):
#             linestyle = "-" if i == 0 else "--"
#             plt.plot(
#                 episodes,
#                 q_values_p1[:, i],
#                 linestyle=linestyle,
#                 color=colors[run_idx],
#                 label=f"{labels[run_idx]}: {action_names[i]}",
#             )
#
#     plt.title(f"Player 1 Q-Values Over Time ({game_name})")
#     plt.xlabel("Episodes")
#     plt.ylabel("Q-Value")
#     plt.legend()
#     plt.grid(True, alpha=0.3)
#
#     # Player 2 Q-values
#     plt.subplot(2, 1, 2)
#     for run_idx, history in enumerate(histories):
#         episodes = history["episodes"]
#         q_values_p2 = np.array(history["q_values_p2"])
#
#         for i in range(2):
#             linestyle = "-" if i == 0 else "--"
#             plt.plot(
#                 episodes,
#                 q_values_p2[:, i],
#                 linestyle=linestyle,
#                 color=colors[run_idx],
#                 label=f"{labels[run_idx]}: {action_names[i]}",
#             )
#
#     plt.title(f"Player 2 Q-Values Over Time ({game_name})")
#     plt.xlabel("Episodes")
#     plt.ylabel("Q-Value")
#     plt.legend()
#     plt.grid(True, alpha=0.3)
