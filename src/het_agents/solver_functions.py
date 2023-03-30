"""Functions for solving small heterogeneous agents models."""

import estimagic as em
import numpy as np
from scipy.interpolate import interp1d


def excess_demand(capital):
    """This function calculates the difference between the equilibrium capital demand
    and the given capital stock. The equilibrium capital demand is calculated using the
    function aiyagari() with inputs that depend on the given capital stock.

    Args:
    capital (float): the given capital stock

    Returns:
    eq_capital_demand - capital (float): the difference between the equilibrium capital
    demand and the given capital stock.

    """
    eq_capital_demand, *_ = aiyagari(
        rate(Capital),
        wage(Capital),
        economic_params,
        numerical_params,
    )
    return eq_capital_demand - capital


def capital_demand(Return_rate):
    """This function calculates the aggregate capital demand for a given interest rate
    (return rate). It uses the capital share of income and the depreciation rate to
    determine the factor by which the marginal product of capital needs to exceed the
    rental rate to justify investment in capital.

    Args:
    Return_rate (float): the given return rate (interest rate)

    Returns:
    Capital Demand (float): the aggregate capital demand assuming a representative firm.

    """
    return economic_params["aggregate_labor_sup"] * (
        economic_params["alpha"] / (Return_rate + economic_params["delta"])
    ) ** (1 / (1 - economic_params["alpha"]))


def solve_steady_state(economic_params, numerical_params, algorithm="scipy_lbfgsb"):
    """This function solves the steady-state equilibrium of an Aiyagari model with
    heterogeneous agents and incomplete markets, using the specified numerical
    optimization algorithm.

    Args:
        economic_params (dict): A dictionary containing economic parameters such as the discount
         factor, return rate, borrowing limit, and a transition matrix for employment states.
        numerical_params (dict): A dictionary containing numerical parameters such as the number
         of grid points for capital and employment.
        algorithm (str, optional): A string indicating the numerical optimization algorithm to use.
         Defaults to "scipy_lbfgsb".

    Returns:
        K_star (float): The aggregate capital stock in the steady state.
        saving_star (numpy.ndarray): A two-dimensional array with shape (n_points_capital_grid, n_points_income_grid)
         containing the savings decisions for each combination of capital and income in the steady state.
        marginal_k (numpy.ndarray): A one-dimensional array of length n_points_capital_grid representing
         the marginal distribution of capital in the steady state.
        StDist (numpy.ndarray): A one-dimensional array of length n_points_capital_grid times
         n_points_income_grid representing the steady-state distribution of households.
        Gamma (numpy.ndarray): A two-dimensional array with shape (n_points_capital_grid, n_points_income_grid)
         representing the steady-state transition matrix for households.
        c_star (numpy.ndarray): A one-dimensional array of length n_points_capital_grid times
         n_points_income_grid containing the consumption decisions for each combination of capital
          and income in the steady state.

    """

    def rate(Capital):
        """This subfunction calculates the interest rate as a function of capital, using
        the Cobb-Douglas production function and the specified economic parameters.

        Args:
            Capital (float): The aggregate capital stock in the economy.

        Returns:
            float: The interest rate corresponding to the given level of capital.

        """
        return (
            economic_params["alpha"]
            * economic_params["aggregate_labor_sup"] ** (1 - economic_params["alpha"])
            * Capital ** (economic_params["alpha"] - 1)
            - economic_params["delta"]
        )

    def wage(Capital):
        """This subfunction calculates the wage rate as a function of capital, using the
        Cobb-Douglas production function and the specified economic parameters.

        Args:
            Capital (float): The aggregate capital stock in the economy.

        Returns:
            float: The wage rate corresponding to the given level of capital.

        """
        return (
            (1 - economic_params["alpha"])
            * economic_params["aggregate_labor_sup"] ** (-economic_params["alpha"])
            * Capital ** (economic_params["alpha"])
        )

    # Solve for Rstar and Kstar
    res = em.minimize(
        criterion=excess_demand,
        params=4,
        algorithm=algorithm,
        lower_bounds=float(
            capital_demand(0.052),
        ),  # recall incomplete markets -> solution will be a bound
        upper_bounds=float(
            capital_demand(-0.01),
        ),  # Effective Lower Bound for interest rates
    )
    Kstar = res.params

    Rate = rate(Kstar)
    Wage = wage(Kstar)
    # Calculate other variables using K_Agg function
    _, saving_star, marginal_k, StDist, Gamma, c_star = aiyagari(
        Rate,
        Wage,
        economic_params,
        numerical_params,
    )

    return K_star, saving_star, marginal_k, StDist, Gamma, c_star


def aiyagari(Rate, Wage, economic_params, numerical_params):
    """This function simulates the Aiyagari model of a dynamic economy with
    heterogeneous agents and (in)complete markets, and returns the steady-state capital
    stock and other relevant variables.

    Args:
        Rate (float): The interest rate in the economy.
        Wage (float): The wage rate in the economy.
        economic_params (dict): A dictionary containing economic parameters such as the discount
         factor, return rate, borrowing limit, and a transition matrix for employment states.
        numerical_params (dict): A dictionary containing numerical parameters such as the number
         of grid points for capital and employment.

    Returns:
        Capital_stock (float): The aggregate capital stock in the steady state.
        kprime (numpy.ndarray): A two-dimensional array with shape
        (n_points_capital_grid, n_points_income_grid) containing the savings
        decisions for each combination of capital and income.
        marginal_k (numpy.ndarray): A one-dimensional array of length n_points_capital_grid
         representing the marginal distribution of capital in the steady state.
        StDist (numpy.ndarray): A one-dimensional array of length n_points_capital_grid times
         n_points_income_grid representing the steady-state distribution of households' savings.
        Gamma (numpy.ndarray): A two-dimensional array with shape
         (n_points_capital_grid, n_points_income_grid) representing the steady-state
          transition matrix for households.
        consumption_grid (numpy.ndarray): A one-dimensional array of length n_points_capital_grid
        times n_points_income_grid containing the consumption decisions for each combination
         of capital and income in the steady state.

    """
    economic_params["income_mesh"] = economic_params["state_mesh"] * Wage
    old_consumption_grid = (
        economic_params["income_mesh"] + Rate * economic_params["capital_mesh"]
    )
    old_consumption_grid = old_consumption_grid.flatten(order="F")  # reshape

    distEG = 1  # Initialize distance
    while distEG > numerical_params["critical_value"]:
        # Update consumption policy by EGM
        consumption_grid, _ = endogenous_grid_method(
            old_consumption_grid,
            1 + Rate,
            1 + Rate,
            economic_params,
            numerical_params,
        )
        # Calculate distance
        distEG = np.max(
            np.abs(
                consumption_grid.flatten(order="F")
                - old_consumption_grid.flatten(order="F"),
            ),
        )
        old_consumption_grid = consumption_grid.copy()  # Replace old policy

    # Calculate capital policy
    _, kprime = endogenous_grid_method(
        consumption_grid,
        1 + Rate,
        1 + Rate,
        economic_params,
        numerical_params,
    )

    # Calculate distribution of wealth and aggregate capital
    Gamma, StDist = young(kprime, economic_params, numerical_params)
    marginal_k = np.sum(
        np.reshape(
            StDist,
            (
                numerical_params["n_points_capital_grid"],
                numerical_params["n_points_income_grid"],
            ),
            order="F",
        ),
        axis=1,
    )
    Capital_stock = np.dot(marginal_k, economic_params["capital_grid"])

    return Capital_stock, kprime, marginal_k, StDist, Gamma, consumption_grid


def young(kprime, economic_params, numerical_params):
    """This function calculates a transition matrix from policy functions and uses it to
    obtain the stationary distribution of income and wealth. The method used is known as
    Young's method.

    Args:
        kprime: array of size (n_points_capital_grid, n_points_income_grid) representing the policy functions
        economic_params: object containing economic parameters
        numerical_params: object containing numerical parameters
    Outputs:

        Gamma: array of size (mpar.nkmpar.nz, mpar.nkmpar.nz) representing the transition matrix
        StDist: array of size (mpar.nk*mpar.nz,) representing the stationary distribution of income and wealth
    Algorithm Steps:

        Find the next lowest point on the grid for policy and remain in the index set.
        Calculate linear interpolation weights for the asset values.
        Fill the transition matrix with probabilities using these weights and current income states.
        Reshape the resulting 4-dimensional array into a transition matrix by stacking two dimensions.
        Solve the equation x* = Gamma' x* to obtain the eigenvector with the largest eigenvalue.
        Normalize the eigenvector to sum(x) = 1 to obtain the stationary distribution of income and wealth.
        Return the transition matrix Gamma and the stationary distribution StDist.

    """
    idk = (
        np.searchsorted(economic_params["capital_grid"], kprime, side="right") - 1
    )  # find the next lowest point on grid for policy
    idk[kprime <= economic_params["capital_grid"][0]] = 0  # remain in the index set
    idk[kprime >= economic_params["capital_grid"][-1]] = (
        numerical_params["n_points_capital_grid"] - 2
    )  # remain in the index set

    # Calculate linear interpolation weights
    distance = kprime - economic_params["capital_grid"][idk]
    weightright = distance / (
        economic_params["capital_grid"][idk + 1] - economic_params["capital_grid"][idk]
    )
    weightleft = 1 - weightright

    Trans_array = np.zeros(
        (
            numerical_params["n_points_capital_grid"],
            numerical_params["n_points_income_grid"],
            numerical_params["n_points_capital_grid"],
            numerical_params["n_points_income_grid"],
        ),
    )  # Assets now x Income now x Assets next x Income next
    # Fill this array with probabilities
    for zz in range(
        numerical_params["n_points_income_grid"],
    ):  # all current income states
        for kk in range(
            numerical_params["n_points_capital_grid"],
        ):  # all current asset states
            Trans_array[kk, zz, idk[kk, zz], :] = weightleft[kk, zz] * np.reshape(
                economic_params["transition_mat"][zz, :],
                [1, 1, 1, numerical_params["n_points_income_grid"]],
            )  # probability to move left to optimal policy choice
            Trans_array[kk, zz, idk[kk, zz] + 1, :] = weightright[kk, zz] * np.reshape(
                economic_params["transition_mat"][zz, :],
                [1, 1, 1, numerical_params["n_points_income_grid"]],
            )  # probability to move right to optimal policy choice

    Gamma = np.reshape(
        Trans_array,
        [
            numerical_params["n_points_capital_grid"]
            * numerical_params["n_points_income_grid"],
            numerical_params["n_points_capital_grid"]
            * numerical_params["n_points_income_grid"],
        ],
        order="F",
    )  # Turn 4-d array into a transition matrix stacking 2 dimensions
    evals, evecs = np.linalg.eig(Gamma.T)
    # Choose the eigenvector with the largest magnitude eigenvalues
    mark = np.argmax(evals)
    x = evecs[:, mark].real  # Select the eigenvector with the largest eigenvalue
    StDist = x / np.sum(x)  # Normalize Eigenvector to sum(x) = 1

    return Gamma, StDist


def endogenous_grid_method(
    consumption_grid, Rate, Rateprime, economic_params, numerical_params,
):
    """This function uses the Endogenous Grid Method (EGM) to solve for the consumption
    and savings decisions of households in a dynamic economic model with incomplete
    markets.

    Args:
    economic_params (dict): A dictionary containing economic parameters such as the discount factor,
     return rate, borrowing limit, and a transition matrix for employment states.
    numerical_params (dict): A dictionary containing numerical parameters such as the number
     of grid points for capital and employment.

    Returns:
    consumption_grid (numpy.ndarray): A one-dimensional array of length n_points_capital_grid
     times n_points_income_grid containing the consumption decisions for each combination of capital and income.
    Kprime (numpy.ndarray): A two-dimensional array with shape (n_points_capital_grid, n_points_income_grid)
     containing the savings decisions for each combination of capital and income.

    """
    # Reshape C to a matrix with shape (nk, nz)
    consumption_grid = consumption_grid.reshape(
        (
            numerical_params["n_points_income_grid"],
            numerical_params["n_points_capital_grid"],
        ),
    ).T

    # Calculate marginal utility from c'
    mu = get_marginal_utility(consumption_grid)

    # Calculate expected marginal utility
    expected_mu = mu @ economic_params["transition_mat"].T

    Cstar = get_inverse_marginal_utility(
        economic_params["beta"] * Rateprime * expected_mu,
    )

    Kstar = (
        Cstar + economic_params["capital_mesh"] - economic_params["income_mesh"]
    ) / Rateprime

    # Initialize (Guess) Kprime as the mesh for capital
    Kprime = economic_params["capital_mesh"].copy()

    # TODO: vectorize
    for z in range(numerical_params["n_points_income_grid"]):
        # Generate savings function
        Savings = interp1d(
            Kstar[:, z],
            economic_params["capital_grid"],
            kind="linear",
            fill_value="extrapolate",
        )

        Kprime[:, z] = Savings(economic_params["capital_grid"])

        # Check Borrowing Constraint
        BC = economic_params["capital_grid"] < Kstar[0, z]

        # Households with the BC flag choose borrowing constraint
        Kprime[BC, z] = economic_params["borrowing_limit"]

    # Update consumption
    consumption_grid = (
        economic_params["capital_mesh"] * Rate + economic_params["income_mesh"] - Kprime
    )
    consumption_grid = consumption_grid.flatten(order="F")

    return consumption_grid, Kprime


# Vectorized versions are faster than numba or jax. BUT with jax you can generalize it to any Utility F.
def get_marginal_utility(c, gamma=economic_params["gamma"]):
    """This function calculates the marginal utility of consumption using the Constant
    Relative Risk Aversion (CRRA) utility function.

    Args:
        c (numpy.ndarray or float): Consumption level(s) for which the marginal
         utility is calculated.
        gamma (float, optional): The risk aversion parameter for the CRRA utility function.
         Defaults to economic_params["gamma"].

    Returns:
        numpy.ndarray or float: The marginal utility of consumption for the given
         consumption level(s) and risk aversion parameter.

    """
    if gamma == 1:
        return 1 / c
    else:
        return 1 / (c**gamma)


def get_inverse_marginal_utility(mu, gamma=economic_params["gamma"]):
    """This function calculates the inverse marginal utility of consumption using the
    Constant Relative Risk Aversion (CRRA) utility function.

    Args:
        mu (numpy.ndarray or float): Marginal utility level(s) for which the inverse marginal
         utility is calculated.
        gamma (float, optional): The risk aversion parameter for the CRRA utility function.
         Defaults to economic_params["gamma"].

    Returns:
        numpy.ndarray or float: The inverse marginal utility of consumption for the given marginal
         utility level(s) and risk aversion parameter.

    """
    if gamma == 1:
        return 1 / mu
    else:
        return (1 / mu) ** (1 / gamma)
