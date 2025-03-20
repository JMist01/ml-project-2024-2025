import numpy as np


class ReplicatorDynamics:
    def __init__(self, payoff_matrix_A, payoff_matrix_B):
        """
        Initialize replicator dynamics with payoff matrices for two players.

        Parameters:
        -----------
        payoff_matrix_A : numpy.ndarray
            Payoff matrix for player 1 (row player)
        payoff_matrix_B : numpy.ndarray
            Payoff matrix for player 2 (column player)
        """
        self.A = payoff_matrix_A
        self.B = payoff_matrix_B

        # Ensure matrices have compatible dimensions
        assert self.A.shape == self.B.shape, "Payoff matrices must have the same shape"

        # Number of strategies for each player
        self.n_strategies = self.A.shape[0]

    def dynamics(self, t, policy_prob):
        """
        Replicator dynamics equations for two-player games.

        Parameters:
        -----------
        t : float
            Time variable (not used directly, but required for ODE solvers)
        state : numpy.ndarray
            Current state of the system, where state[:n_strategies] represents
            player 1's strategy distribution and state[n_strategies:] represents
            player 2's strategy distribution

        Returns:
        --------
        numpy.ndarray
            The derivatives of the state variables
        """
        # Split state into two players' strategy distributions
        x = policy_prob[: self.n_strategies]
        y = policy_prob[self.n_strategies :]

        # Ensure probabilities sum to 1 (due to numerical errors)
        x = x / np.sum(x)
        y = y / np.sum(y)

        # Calculate fitness for each strategy
        fitness_x = np.dot(self.A, y)
        fitness_y = np.dot(x, self.B)

        # Calculate average fitness
        avg_fitness_x = np.dot(x, fitness_x)
        avg_fitness_y = np.dot(fitness_y, y)

        # Calculate derivatives according to replicator dynamics
        dx = x * (fitness_x - avg_fitness_x)
        dy = y * (fitness_y - avg_fitness_y)

        return np.concatenate([dx, dy])


def compute_vector_field(rewards, grid_size=20):
    """
    Compute vector field for 2x2 games.

    Parameters:
    -----------
    grid_size : int
        Number of grid points along each axis

    Returns:
    --------
    tuple
        (X, Y, U, V) where X, Y are meshgrid coordinates and U, V are vector components
    """
    A = np.zeros((2, 2))
    B = np.zeros((2, 2))

    for i in range(2):
        for j in range(2):
            A[i, j] = rewards[i][j][0]
            B[i, j] = rewards[i][j][1]

    # Create a grid of points
    x = np.linspace(0.01, 0.99, grid_size)
    y = np.linspace(0.01, 0.99, grid_size)
    X, Y = np.meshgrid(x, y)

    # Initialize arrays for vector components
    U = np.zeros_like(X)
    V = np.zeros_like(Y)

    # Compute vector field
    for i in range(grid_size):
        for j in range(grid_size):
            x_strat = np.array([X[i, j], 1 - X[i, j]])
            y_strat = np.array([Y[i, j], 1 - Y[i, j]])

            # Calculate derivatives
            fitness_x = np.dot(A, y_strat)
            fitness_y = np.dot(x_strat, B)

            avg_fitness_x = np.dot(x_strat, fitness_x)
            avg_fitness_y = np.dot(y_strat, fitness_y)

            dx = x_strat[0] * (fitness_x[0] - avg_fitness_x)
            dy = y_strat[0] * (fitness_y[0] - avg_fitness_y)

            U[i, j] = dx
            V[i, j] = dy

    return X, Y, U, V
