import numpy as np
from policy import *


def compute_vector_field(policy, rewards, grid_size=20, lr=0.1):
    """
    Compute the vector field for learning dynamics.

    Args:
        rewards: The reward matrix
        policy: The policy object (BoltzmannPolicy or LenientBoltzmannPolicy)
        grid_size: Resolution of the grid
        lr: Learning rate

    Returns:
        X, Y, X_derivative, Y_derivative: Grid points and vectors
    """
    # Grid setup
    x = np.linspace(0.01, 0.99, grid_size)
    y = np.linspace(0.01, 0.99, grid_size)
    X, Y = np.meshgrid(x, y)

    # Output vectors
    X_derivative = np.zeros_like(X)
    Y_derivative = np.zeros_like(Y)

    # For each point in the grid
    for i in range(grid_size):
        for j in range(grid_size):
            # Current mixed strategies (as probabilities of playing action 0)
            p1_prob_action0 = X[i, j]
            p2_prob_action0 = Y[i, j]

            # Setup Q-values that would produce these probabilities
            # For Boltzmann policies, we need to derive Q-values that give our desired probabilities
            # if p1_prob_action0 > 0.99:
            #     p1_prob_action0 = 0.99
            # if p1_prob_action0 < 0.01:
            #     p1_prob_action0 = 0.01
            # if p2_prob_action0 > 0.99:
            #     p2_prob_action0 = 0.99
            # if p2_prob_action0 < 0.01:
            #     p2_prob_action0 = 0.01

            temp = policy.temperature

            # For a Boltzmann policy with probability p of choosing action 0:
            # p = exp(Q[0]/temp) / (exp(Q[0]/temp) + exp(Q[1]/temp))
            # Solving for Q[0] - Q[1]:
            # Q[0] - Q[1] = temp * log(p/(1-p))
            # TODO: here  might be some problme with how you get q-valeu from probablitesi
            q_diff_p1 = temp * np.log(p1_prob_action0 / (1 - p1_prob_action0))
            q_diff_p2 = temp * np.log(p2_prob_action0 / (1 - p2_prob_action0))

            # Set arbitrary base value and derive Q-values
            q_val_p1 = [0, -q_diff_p1]
            q_val_p2 = [0, -q_diff_p2]

            # Track policy change over many simulated updates
            num_samples = 200
            new_p1_probs = np.zeros(2)
            new_p2_probs = np.zeros(2)

            policy.restart()
            for _ in range(num_samples):
                # Clone Q-values for this sample
                q1_sample = q_val_p1.copy()
                q2_sample = q_val_p2.copy()

                # Perform update step according to policy type
                # if isinstance(policy, LenientBoltzmannPolicy):
                # Simulate lenient update
                policy.update_step(rewards, q1_sample, q2_sample, lr)
                # else:
                #     # Standard Boltzmann update
                #     action_p1 = np.random.choice(
                #         2, p=[p1_prob_action0, 1 - p1_prob_action0]
                #     )
                #     action_p2 = np.random.choice(
                #         2, p=[p2_prob_action0, 1 - p2_prob_action0]
                #     )
                #     reward_p1, reward_p2 = rewards[action_p1][action_p2]

                #     q1_sample[action_p1] += lr * (reward_p1 - q1_sample[action_p1])
                #     q2_sample[action_p2] += lr * (reward_p2 - q2_sample[action_p2])

                # Get updated probabilities
                updated_probs_p1 = policy._boltzmann_probs(q1_sample)
                updated_probs_p2 = policy._boltzmann_probs(q2_sample)

                new_p1_probs += updated_probs_p1
                new_p2_probs += updated_probs_p2

            # Average probabilities after updates
            new_p1_probs /= num_samples
            new_p2_probs /= num_samples

            # Calculate derivative (change in probability)
            X_derivative[i, j] = new_p1_probs[0] - p1_prob_action0
            Y_derivative[i, j] = new_p2_probs[0] - p2_prob_action0

    return X, Y, X_derivative, Y_derivative


# TODO: it's wrong, for boltzmann and lenianct boltzmann plots will look different...
# def compute_vector_field(rewards, grid_size=20):
#     A = np.zeros((2, 2))
#     B = np.zeros((2, 2))
#
#     for i in range(2):
#         for j in range(2):
#             A[i, j] = rewards[i][j][0]
#             B[i, j] = rewards[i][j][1]
#     # Grid
#     x = np.linspace(0.01, 0.99, grid_size)
#     y = np.linspace(0.01, 0.99, grid_size)
#     X, Y = np.meshgrid(x, y)
#
#     # Output vectors
#     X_derivative = np.zeros_like(X)
#     Y_derivative = np.zeros_like(Y)
#
#     # Calculate derivatives
#     for i in range(grid_size):
#         for j in range(grid_size):
#             x_policy = np.array([X[i, j], 1 - X[i, j]])
#             y_policy = np.array([Y[i, j], 1 - Y[i, j]])
#             x_policy = np.transpose(x_policy)
#
#             dx = x_policy[0] * (
#                 np.dot(A, y_policy)[0] - np.dot(np.dot(x_policy, A), y_policy)
#             )
#             dy = y_policy[0] * (
#                 np.dot(x_policy, B)[0] - np.dot(np.dot(x_policy, B), y_policy)
#             )
#
#             X_derivative[i, j] = dx
#             Y_derivative[i, j] = dy
#
#     return X, Y, X_derivative, Y_derivative
