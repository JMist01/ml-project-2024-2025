import matplotlib.pyplot as plt
import numpy as np
from replicator_dynamic import compute_vector_field


def visualize_policy_traces(histories, ax):
    # Mark beginning and end of traces with different symbols
    labels = [f"Run {i + 1}" for i in range(len(histories))]
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i / len(histories)) for i in range(len(histories))]

    for i, history in enumerate(histories):
        p1_policies = history["policy_p1"]
        p2_policies = history["policy_p2"]
        p1_probs = [policy[0] for policy in p1_policies]
        p2_probs = [policy[0] for policy in p2_policies]

        # Plot the trajectory line
        ax.plot(p1_probs, p2_probs, linewidth=2, color=colors[i], label=labels[i])

        # Mark beginning of trajectory with a circle
        ax.scatter(
            p1_probs[0],
            p2_probs[0],
            s=100,
            marker="o",
            color=colors[i],
            edgecolor="black",
            linewidth=1.5,
            zorder=10,
        )

        # Mark end of trajectory with a star
        ax.scatter(
            p1_probs[-1],
            p2_probs[-1],
            s=150,
            marker="*",
            color=colors[i],
            edgecolor="black",
            linewidth=1.5,
            zorder=10,
        )


def plot_vector_field(policy, rewards, ax, grid_size, lr):
    X, Y, U, V = compute_vector_field(policy, rewards, grid_size, lr * 10)
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
