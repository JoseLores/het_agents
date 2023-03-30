"""Functions for obtaining model solving steady state results."""

import timeit

import estimagic as em
import numpy as np

from het_agents.solver_functions import *


def capital_demand_supply(economic_params, numerical_params):
    """Compute the capital demand and capital supply curves for a range of interest
    rates.

    Args:
        economic_params (dict): A dictionary containing economic parameters.
        numerical_params (dict): A dictionary containing numerical parameters.

    Returns:
        demand_curve (numpy.ndarray): Capital demand curve for the given range of interest rates.
        supply_curve (numpy.ndarray): Capital supply curve for the given range of interest rates.

    """
    rgrid = np.arange(
        -0.01,
        0.055,
        0.0025,
    )  # TODO: should this be an interval (+-r_star)?
    demand_curve = np.array(
        [capital_demand(r, economic_params, numerical_params) for r in rgrid],
    )
    capital_excess_demanded = np.array(
        [excess_demand(k, economic_params, numerical_params) for k in demand_curve],
    )
    supply_curve = demand_curve + capital_excess_demanded

    return demand_curve, supply_curve


def benchmark_algorithms_iterations(algorithms, economic_params, numerical_params):
    """Benchmark a list of optimization solvers using Estimagic for a given criterion
    function.

    Args:
        algorithms (list): A list of strings representing the names of optimization solvers to be benchmarked.

    Returns:
        dict: A dictionary containing the results of the benchmark for each solver.

    """
    results = {}

    for algo in algorithms:
        results[algo] = em.minimize(
            criterion=excess_demand,
            params=4,
            algorithm=algo,
            lower_bounds=float(
                capital_demand(0.052, economic_params, numerical_params),
            ),
            upper_bounds=float(
                capital_demand(-0.01, economic_params, numerical_params),
            ),
            criterion_kwargs={
                "economic_params": economic_params,
                "numerical_params": numerical_params,
            },
        )

    return results


def benchmark_algorithms_time(
    algorithms,
    economic_params,
    numerical_params,
    num_of_runs=50,
):
    """Run the benchmark for a list of optimization solvers for the steady-state
    solution using the solve_steady_state function and record the average runtime.

    Args:
        algorithms (list): A list of strings representing the names of optimization solvers to be benchmarked.
        num_of_runs (int): The number of times each solver will be run in the benchmark.
        economic_params (dict): A dictionary containing economic parameters.
        numerical_params (dict): A dictionary containing numerical parameters.

    Returns:
        dict: A dictionary containing the average runtime of the benchmark for each solver.

    """
    runtimes = {}
    for algorithm in algorithms:
        runtime = (
            timeit.timeit(
                lambda: solve_steady_state(
                    economic_params,
                    numerical_params,
                    algorithm,
                ),
                globals=globals(),
                number=num_of_runs,
            )
            / num_of_runs
        )

        runtimes[algorithm] = runtime

    return runtimes
