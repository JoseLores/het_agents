import numpy as np
import pytest
from het_agents.analysis.benchmark import (
    benchmark_algorithms_iterations,
    benchmark_algorithms_time,
    capital_demand_supply,
)
from het_agents.config import TEST_DIR
from het_agents.utilities import read_pkl


@pytest.fixture()
def params():
    numerical_params, economic_params = read_pkl(
        TEST_DIR / "analysis" / "econ_data_fixture.pkl",
    )
    return economic_params, numerical_params


def test_capital_demand_supply(params):
    economic_params, numerical_params = params

    demand_curve, supply_curve = capital_demand_supply(
        economic_params,
        numerical_params,
    )
    assert isinstance(demand_curve, np.ndarray), "Demand curve should be a numpy array"
    assert isinstance(supply_curve, np.ndarray), "Supply curve should be a numpy array"
    assert (
        demand_curve.shape == supply_curve.shape
    ), "Demand and supply curves should have the same shape"


def test_benchmark_algorithms_iterations(params):
    economic_params, numerical_params = params

    algorithms = ["scipy_lbfgsb", "scipy_truncated_newton"]

    results = benchmark_algorithms_iterations(
        algorithms,
        economic_params,
        numerical_params,
    )
    assert isinstance(results, dict), "Results should be a dictionary"
    assert len(results) == len(
        algorithms,
    ), "Results should have the same length as the list of algorithms"
    for algo in algorithms:
        assert algo in results, f"{algo} should be in the results"


def test_benchmark_algorithms_time(params):
    economic_params, numerical_params = params

    algorithms = ["scipy_lbfgsb", "scipy_truncated_newton"]

    num_of_runs = 2

    runtimes = benchmark_algorithms_time(
        algorithms,
        economic_params,
        numerical_params,
        num_of_runs=num_of_runs,
    )
    assert isinstance(runtimes, dict), "Runtimes should be a dictionary"
    assert len(runtimes) == len(
        algorithms,
    ), "Runtimes should have the same length as the list of algorithms"
    for algo in algorithms:
        assert algo in runtimes, f"{algo} should be in the runtimes"
        assert isinstance(
            runtimes[algo],
            float,
        ), f"Runtime for {algo} should be a float"
