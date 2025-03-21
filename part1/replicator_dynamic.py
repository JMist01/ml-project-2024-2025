import numpy as np
from policy import *


def compute_replicator_field(rewards, grid_size=20, temp=0.3):
    x_policy = np.linspace(0.01, 0.99, grid_size)
    y_policy = np.linspace(0.01, 0.99, grid_size)
    X, Y = np.meshgrid(x_policy, y_policy)

    A = np.zeros((2, 2))
    B = np.zeros((2, 2))

    for x in range(2):
        for y in range(2):
            A[x, y] = rewards[x][y][0]
            B[x, y] = rewards[x][y][1]

    DX = np.zeros_like(X)
    DY = np.zeros_like(Y)

    for x in range(grid_size):
        for y in range(grid_size):
            x_prob = X[x, y]
            y_prob = Y[x, y]

            x_policy = np.array([x_prob, 1 - x_prob])
            y_policy = np.array([y_prob, 1 - y_prob])

            kappa = 2
            additional_term = [0, 0]
            for i in range(2):
                for j in range(2):
                    a_ij = A[i, j]
                    y_j = y_policy[j]

                    S_leq = sum(y_policy[k] for k in range(2) if A[i, k] <= a_ij)
                    S_lt = sum(y_policy[k] for k in range(2) if A[i, k] < a_ij)
                    S_eq = sum(y_policy[k] for k in range(2) if A[i, k] == a_ij)

                    if S_eq != 0:
                        diff_term = (S_leq**kappa - S_lt**kappa) / S_eq
                        additional_term[i] += a_ij * y_j * diff_term
            Ay = A @ y_policy
            Ay = additional_term
            dx = x_prob / temp * ((Ay)[0] - (x_policy.T @ Ay)) - x_prob * 1 * (
                np.log(x_policy[0])
                - x_policy[0] * np.log(x_policy[0])
                - x_policy[1] * np.log(x_policy[1])
            )
            additional_term = [0, 0]
            for i in range(2):
                for j in range(2):
                    b_ji = B[j, i]
                    x_j = x_policy[j]

                    S_leq = sum(x_policy[k] for k in range(2) if B[k, i] <= b_ji)
                    S_lt = sum(x_policy[k] for k in range(2) if B[k, i] < b_ji)
                    S_eq = sum(x_policy[k] for k in range(2) if B[k, i] == b_ji)

                    if S_eq != 0:
                        diff_term = (S_leq**kappa - S_lt**kappa) / S_eq
                        additional_term[i] += b_ji * x_j * diff_term
            Bx = x_policy @ B
            Bx = additional_term
            dy = y_prob / temp * ((Bx)[0] - (Bx @ y_policy)) - y_prob * 1 * (
                np.log(y_policy[0])
                - y_policy[0] * np.log(y_policy[0])
                - y_policy[1] * np.log(y_policy[1])
            )

            DX[x, y] = dx
            DY[x, y] = dy

    return X, Y, DX, DY


#
# def compute_vector_field(policy, rewards, grid_size=20, lr=0.1):
#     x = np.linspace(0.01, 0.99, grid_size)
#     y = np.linspace(0.01, 0.99, grid_size)
#     X, Y = np.meshgrid(x, y)
#
#     X_derivative = np.zeros_like(X)
#     Y_derivative = np.zeros_like(Y)
#
#     for i in range(grid_size):
#         for j in range(grid_size):
#             p1_prob_action0 = X[i, j]
#             p2_prob_action0 = Y[i, j]
#
#             temp = policy.temperature
#
#             # For a Boltzmann policy with probability p of choosing action 0:
#             # p = exp(Q[0]/temp) / (exp(Q[0]/temp) + exp(Q[1]/temp))
#             # Solving for Q[0] - Q[1]:
#             # Q[0] - Q[1] = temp * log(p/(1-p))
#             # TODO: here  might be some problme with how you get q-valeu from probablitesi
#             q_diff_p1 = temp * np.log(p1_prob_action0 / (1 - p1_prob_action0))
#             q_diff_p2 = temp * np.log(p2_prob_action0 / (1 - p2_prob_action0))
#
#             q_val_p1 = [0, -q_diff_p1]
#             q_val_p2 = [0, -q_diff_p2]
#
#             num_samples = 1000
#             new_p1_probs = np.zeros(2)
#             new_p2_probs = np.zeros(2)
#
#             policy.restart()
#             for _ in range(num_samples):
#                 q1_sample = q_val_p1.copy()
#                 q2_sample = q_val_p2.copy()
#
#                 policy.update_step(rewards, q1_sample, q2_sample, lr)
#                 updated_probs_p1 = policy._boltzmann_probs(q1_sample)
#                 updated_probs_p2 = policy._boltzmann_probs(q2_sample)
#
#                 new_p1_probs += updated_probs_p1
#                 new_p2_probs += updated_probs_p2
#
#             new_p1_probs /= num_samples
#             new_p2_probs /= num_samples
#
#             # TODO: what if i used the q-values as payoff martix?
#             # x_policy = new_p1_probs
#             # y_policy = new_p2_probs
#             # x_policy = np.transpose(x_policy)
#             # A = q1_sample
#             # B = q2_sample
#             # aaa = np.dot(A, y_policy)
#             # np.dot(np.dot(x_policy, A), y_policy)
#             # dx = x_policy[0] * (
#             #     np.dot(A, y_policy)[0] - np.dot(np.dot(x_policy, A), y_policy)
#             # )
#             # dy = y_policy[0] * (
#             #     np.dot(x_policy, B)[0] - np.dot(np.dot(x_policy, B), y_policy)
#             # )
#
#             # X_derivative[i, j] = dx
#             # Y_derivative[i, j] = dy
#
#             X_derivative[i, j] = new_p1_probs[0] - p1_prob_action0
#             Y_derivative[i, j] = new_p2_probs[0] - p2_prob_action0
#
#     return X, Y, X_derivative, Y_derivative


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
