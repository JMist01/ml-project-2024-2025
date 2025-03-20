import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap
from replicator_dynamic import compute_vector_field


def visualize_policy_traces(histories, game_name, action_names, ax=None):
    # Use provided axes or create new ones if none provided
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 8))

    labels = [f"Run {i + 1}" for i in range(len(histories))]
    cmap = plt.get_cmap("tab10")
    colors = [cmap(i / len(histories)) for i in range(len(histories))]
    for i, history in enumerate(histories):
        p1_policies = history["policy_p1"]
        p2_policies = history["policy_p2"]
        p1_probs = [policy[0] for policy in p1_policies]
        p2_probs = [policy[0] for policy in p2_policies]
        ax.plot(p1_probs, p2_probs, linewidth=2, color=colors[i], label=labels[i])

    ax.set_xlabel(f"P1 probability of selecting action {action_names[0]}")
    ax.set_ylabel(f"P2 probability of selecting action {action_names[0]}")
    ax.set_title(f"Empirical policy traces {game_name}")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True)
    # ax.legend()

    return ax


def plot_vector_field(
    rewards,
    grid_size=20,
    title="Replicator Dynamics Vector Field",
    cmap="viridis",
    ax=None,
    fig=None,
):
    X, Y, U, V = compute_vector_field(rewards, grid_size)
    # Calculate vector magnitudes for coloring
    magnitudes = np.sqrt(U**2 + V**2)

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))

    norm = mpl.colors.Normalize(vmin=0, vmax=np.max(magnitudes))
    q = ax.quiver(
        X,
        Y,
        U,
        V,
        magnitudes,
        cmap=get_cmap(cmap),
        norm=norm,
        scale=20,
        width=0.002,
        pivot="mid",
    )

    if fig is not None:
        cbar = fig.colorbar(q, ax=ax)
        cbar.set_label("Vector Magnitude")

    ax.set_xlabel("Player 1 - Probability of Action 1")
    ax.set_ylabel("Player 2 - Probability of Action 1")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, linestyle="--", alpha=0.7)

    return ax, q


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
