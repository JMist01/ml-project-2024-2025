import numpy as np


# TODO: it's wrong, for boltzmann and lenianct boltzmann plots will look different...
def compute_vector_field(rewards, grid_size=20):
    A = np.zeros((2, 2))
    B = np.zeros((2, 2))

    for i in range(2):
        for j in range(2):
            A[i, j] = rewards[i][j][0]
            B[i, j] = rewards[i][j][1]
    # Grid
    x = np.linspace(0.01, 0.99, grid_size)
    y = np.linspace(0.01, 0.99, grid_size)
    X, Y = np.meshgrid(x, y)

    # Output vectors
    X_derivative = np.zeros_like(X)
    Y_derivative = np.zeros_like(Y)

    # Calculate derivatives
    for i in range(grid_size):
        for j in range(grid_size):
            x_policy = np.array([X[i, j], 1 - X[i, j]])
            y_policy = np.array([Y[i, j], 1 - Y[i, j]])
            x_policy = np.transpose(x_policy)

            dx = x_policy[0] * (
                np.dot(A, y_policy)[0] - np.dot(np.dot(x_policy, A), y_policy)
            )
            dy = y_policy[0] * (
                np.dot(x_policy, B)[0] - np.dot(np.dot(x_policy, B), y_policy)
            )

            X_derivative[i, j] = dx
            Y_derivative[i, j] = dy

    return X, Y, X_derivative, Y_derivative
