"""Function(s) for cleaning the data set(s)."""

import numpy as np


def produce_grids(economic_params, numerical_params):
    """Obtain capital and employment grids based on numerical and economic parameters.

    Args:
        numerical_params (dict): Dictionary containing numerical parameters.
        economic_params (dict): Dictionary containing economic parameters.

    Returns:
        dict: Dictionary containing the updated economic parameters, including the
              employment grid, capital grid, and capital and employment meshes
              as well as some economic relevant variables such as the
              erdodic distribution, the aggregate labor supply, the share of unemployed
              and the required tax rate.

    """
    ergodic_dist = get_ergodic_dist(economic_params["transition_mat"])

    aggregate_labor_sup, employed_share, unemployed_share = get_aggregate_labor(
        ergodic_dist,
        economic_params["productivity"],
    )

    tax_rate = get_tax_rate(
        economic_params["unemp_benefit"],
        unemployed_share,
        ergodic_dist,
        economic_params["productivity"],
    )

    state_grid = get_state_grid(
        numerical_params["n_points_income_grid"],
        economic_params["unemp_benefit"],
        tax_rate,
        economic_params["productivity"],
    )

    capital_grid = get_k_grid(
        numerical_params["n_points_capital_grid"],
        numerical_params["max_value_capital_grid"],
        numerical_params["min_value_capital_grid"],
    )

    capital_mesh, state_mesh = get_meshes(capital_grid, state_grid)

    # Add subfunctions to the economic_params dictionary
    economic_params["aggregate_labor_sup"] = aggregate_labor_sup
    economic_params["employed_share"] = employed_share
    economic_params["unemployed_share"] = unemployed_share
    economic_params["tax_rate"] = tax_rate
    economic_params["state_grid"] = state_grid
    economic_params["capital_grid"] = capital_grid
    economic_params["capital_mesh"] = capital_mesh
    economic_params["state_mesh"] = state_mesh

    return economic_params


def get_ergodic_dist(transition_mat):
    """Calculate the ergodic distribution of a transition matrix.

    Args:
        transition_mat (numpy.ndarray): Transition matrix for a Markov chain.

    Returns:
        numpy.ndarray: Ergodic distribution of the Markov chain.

    """
    if not _is_irreducible(transition_mat):
        error_message = "The transition matrix is not irreducible."
        raise ValueError(error_message)
    ergodic_dist_mat = np.linalg.matrix_power(transition_mat, 1000)
    ergodic_dist = ergodic_dist_mat[0]
    return ergodic_dist


def get_aggregate_labor(ergodic_dist, productivity):
    """Calculate the aggregate labor supply based on the ergodic distribution and
    productivity.

    Args:
        ergodic_dist (numpy.ndarray): Ergodic distribution of the Markov chain.
        productivity (array): Labor productivity parameter.

    Returns:
        tuple: Tuple containing the aggregate labor supply, employed share, and
        unemployed share.

    """
    employed_share = ergodic_dist[1:].sum().round(5)
    aggregate_labor_sup = np.dot(ergodic_dist[1:], productivity[1:])
    unemployed_share = 1 - employed_share
    return aggregate_labor_sup, employed_share, unemployed_share


def get_tax_rate(unemp_benefit, unemployed_share, ergodic_dist, productivity):
    """Calculate the tax rate required to finance unemployment insurance benefits.

    Args:
        unemp_benefit (float): Unemployment insurance benefit parameter.
        employed_share (numpy.ndarray): Array of employed share over the states
        of the Markov chain.
        unemployed_share (numpy.ndarray): Array of unemployed share over the states
        of the Markov chain.
        ergodic_dist (numpy.ndarray): Ergodic distribution of the Markov chain.
        productivity (array): Labor productivity parameters.

    Returns:
        float: Tax rate required to finance the unemployment insurance benefits.

    """
    # tax rate to finance UIB.
    return unemp_benefit * unemployed_share / np.dot(ergodic_dist[1:], productivity[1:])


def get_state_grid(n_points_z, unemp_benefit, tax_rate, productivity):
    """Calculate the employment grid.

    Args:
        n_points_z (int): Number of grid points.
        unemp_benefit (float): Unemployment insurance benefit parameter.
        tax_rate (float): Tax rate required to finance the unemployment insurance
        benefits.
        productivity (array): Labor productivity array.

    Returns:
        numpy.ndarray: Array of employment grid points.

    """
    state_grid = np.zeros(n_points_z)
    state_grid[0] = unemp_benefit
    state_grid[1:] = productivity[1:] * (1 - tax_rate)
    return state_grid


def get_k_grid(n_points_k, max_k, min_k):
    """Calculates the capital grid.

    Args:
        n_points_k (int): Number of points on the capital grid.
        max_k (float): Maximum value of capital.
        min_k (float): Minimum value of capital.

    Returns:
        numpy.ndarray: Array of capital grid points.

    """
    return np.exp(np.linspace(0, np.log(max_k - min_k + 1), n_points_k)) - 1 + min_k


def get_meshes(capital_grid, state_grid):
    """Creates two two-dimensional arrays representing a meshgrid of the given one-
    dimensional arrays.

    Args:
    - capital_grid: numpy array, one-dimensional array of capital grid values
    - state_grid: numpy array, one-dimensional array of income grid values

    Returns:
    A tuple of two two-dimensional numpy arrays: (capital_mesh, state_mesh).

    - capital_mesh: numpy array, two-dimensional array of capital grid values
        paired with income grid values
    - state_mesh: numpy array, two-dimensional array of income grid values
        paired with capital grid values

    """
    capital_mesh, state_mesh = np.meshgrid(
        capital_grid,
        state_grid,
        indexing="ij",
    )
    return capital_mesh, state_mesh


def _is_irreducible(transition_mat):
    """Check if the Markov chain represented by the transition matrix is irreducible."""
    num_zeros = np.sum(transition_mat == 0, axis=0)
    return not np.any(num_zeros == transition_mat.shape[0])
