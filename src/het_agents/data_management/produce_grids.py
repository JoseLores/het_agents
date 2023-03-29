"""Function(s) for cleaning the data set(s)."""

import numpy as np
import pandas as pd


def produce_grids(numerical_params, economic_params):
    """
    Calculate capital and employment grids based on numerical and economic parameters.

    Args:
        numerical_params (dict): Dictionary containing numerical parameters.
        economic_params (dict): Dictionary containing economic parameters.

    Returns:
        dict: Dictionary containing the updated economic parameters, including the
              employment grid, capital grid, and capital and employment meshes.
    """

    def get_ergodic_dist(transition_mat):
        """
        Calculate the ergodic distribution of a transition matrix.

        Args:
            transition_mat (numpy.ndarray): Transition matrix for a Markov chain.

        Returns:
            numpy.ndarray: Ergodic distribution of the Markov chain.
        """
        # if the matrix is symmetric return the same matrix
        ergodic_dist = (
            np.linalg.matrix_power(transition_mat, 1000)
        )
        return ergodic_dist  # TO DO: test this!! Some properties of markov chains so it always converges

    def get_aggregate_labor(ergodic_dist, productivity):
        """
        Calculate the aggregate labor supply based on the ergodic distribution and productivity.

        Args:
            ergodic_dist (numpy.ndarray): Ergodic distribution of the Markov chain.
            productivity (float): Labor productivity parameter.

        Returns:
            tuple: Tuple containing the aggregate labor supply, employed share, and unemployed share.
        """

        employed_share = ergodic_dist[0, 1:].sum().round(5)
        aggregate_labor_sup = np.sum(employed_share)*productivity
        unemployed_share = 1 - employed_share
        return aggregate_labor_sup, employed_share, unemployed_share

    def get_tax_rate(unemp_benefit, employed_share, unemployed_share):
        """
        Calculate the tax rate required to finance unemployment insurance benefits.

        Args:
            unemp_benefit (float): Unemployment insurance benefit parameter.
            employed_share (numpy.ndarray): Array of employed share over the states of the Markov chain.
            unemployed_share (numpy.ndarray): Array of unemployed share over the states of the Markov chain.

        Returns:
            float: Tax rate required to finance the unemployment insurance benefits.
        """

        tax_rate = unemp_benefit * unemployed_share/employed_share; # tax rate to finance UIB. 
        return tax_rate


    def get_income_grid(n_points_z, unemp_benefit, tax_rate, productivity):
        """
        Calculate the employment grid.

        Args:
            n_points_z (int): Number of grid points.
            unemp_benefit (float): Unemployment insurance benefit parameter.
            tax_rate (float): Tax rate required to finance the unemployment insurance benefits.
            productivity (float): Labor productivity parameter.

        Returns:
            numpy.ndarray: Array of employment grid points.
        """

        income_grid = np.zeros(n_points_z)
        income_grid[0] = unemp_benefit * productivity
        income_grid[1:] = productivity * (1 - tax_rate)
        return income_grid

    def get_k_grid(n_points_k, max_k, min_k):
        """
        Calculates the capital grid.

        Args:
        - n_points_k (int): The number of points on the capital grid.
        - max_k (float): The maximum value of capital.
        - min_k (float): The minimum value of capital.

        Returns:
        - The capital grid.
        """
        capital_grid = np.exp(np.linspace(0, np.log(max_k - min_k + 1), n_points_k)) - 1 + min_k
        return capital_grid

    def get_meshes(capital_grid, income_grid):
        '''
        Creates two two-dimensional arrays representing a meshgrid of the given one-dimensional arrays.
        
        Args:
        - capital_grid: numpy array, one-dimensional array of capital grid values
        - income_grid: numpy array, one-dimensional array of income grid values
        
        Returns:
        A tuple of two two-dimensional numpy arrays: (capital_mesh, income_mesh).
        - capital_mesh: numpy array, two-dimensional array of capital grid values paired with income grid values
        - income_mesh: numpy array, two-dimensional array of income grid values paired with capital grid values
        '''

        capital_mesh, income_mesh = np.meshgrid(capital_grid, income_grid, indexing='ij')
        return capital_mesh, income_mesh

    # Compute subfunctions and store them in separate variables
    
    ergodic_dist = get_ergodic_dist(economic_params['transition_mat'])

    aggregate_labor_sup, employed_share, unemployed_share = get_aggregate_labor(ergodic_dist, economic_params['productivity'])

    tax_rate = get_tax_rate(economic_params['unemp_benefit'], employed_share, unemployed_share)

    income_grid = get_income_grid(numerical_params['n_points_income_grid'], economic_params['unemp_benefit'], tax_rate, economic_params['productivity'])

    capital_grid = get_k_grid(numerical_params['n_points_capital_grid'], numerical_params['max_value_capital_grid'], numerical_params['min_value_capital_grid'])

    capital_mesh, income_mesh = get_meshes(capital_grid, income_grid)

    # Add subfunctions to the economic_params dictionary
    economic_params['aggregate_labor_sup'] = aggregate_labor_sup
    economic_params['employed_share'] = employed_share
    economic_params['unemployed_share'] = unemployed_share
    economic_params['tax_rate'] = tax_rate
    economic_params['income_grid'] = income_grid
    economic_params['capital_grid'] = capital_grid
    economic_params['capital_mesh'] = capital_mesh
    economic_params['income_mesh'] = income_mesh

    return economic_params
    # TODO: compute each funtion & TEST