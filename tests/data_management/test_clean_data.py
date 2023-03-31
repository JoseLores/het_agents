import numpy as np
import pytest
from het_agents.data_management.produce_grids import *


@pytest.fixture()
def numerical_params():
    return {
        "n_points_capital_grid": 10,
        "n_points_income_grid": 2,
        "min_value_capital_grid": 0,
        "max_value_capital_grid": 10,
        "critical_value": 1e-5,
        "time_irfs": 10,
    }


@pytest.fixture()
def economic_params(numerical_params):
    return {
        "beta": 0.95,
        "alpha": 0.36,
        "gamma": 4,
        "delta": 0.1,
        "rho_Z": 0.75,
        "productivity": np.array([1, 1]),
        "unemp_benefit": 0,
        "borrowing_limit": numerical_params["min_value_capital_grid"],
        "transition_mat": np.array([[3 / 5, 2 / 5], [4 / 90, 86 / 90]]),
    }


def test_get_ergodic_dist(economic_params):
    ergodic_dist = get_ergodic_dist(economic_params["transition_mat"])
    assert np.allclose(ergodic_dist, np.array([0.1, 0.9]))


def test_get_aggregate_labor(economic_params):
    ergodic_dist = get_ergodic_dist(economic_params["transition_mat"])
    (
        aggregate_labor_sup,
        employed_share,
        unemployed_share,
    ) = get_aggregate_labor(ergodic_dist, economic_params["productivity"])
    assert np.isclose(aggregate_labor_sup, 0.9)
    assert np.isclose(employed_share, 0.9)
    assert np.isclose(unemployed_share, 0.1)


def test_get_tax_rate(economic_params):
    unemployed_share = 0.1
    ergodic_dist = np.array([0.1, 0.9])
    tax_rate = get_tax_rate(
        economic_params["unemp_benefit"],
        unemployed_share,
        ergodic_dist,
        economic_params["productivity"],
    )
    assert np.isclose(tax_rate, 0)


def test_get_state_grid(numerical_params, economic_params):
    n_points_z = numerical_params["n_points_income_grid"]
    unemp_benefit = economic_params["unemp_benefit"]
    tax_rate = 0
    productivity = economic_params["productivity"]

    state_grid = get_state_grid(n_points_z, unemp_benefit, tax_rate, productivity)

    assert isinstance(
        state_grid,
        np.ndarray,
    ), "The function should return a numpy array"
    assert state_grid.shape == (
        n_points_z,
    ), "The shape of the income grid is incorrect"
    assert np.isclose(
        state_grid[0],
        unemp_benefit,
    ), "The first element should be close to unemp_benefit * productivity"
    assert np.allclose(
        state_grid[1:],
        productivity[1:] * (1 - tax_rate),
    ), "The remaining elements should be close to productivity * (1 - tax_rate)"


def test_get_k_grid(numerical_params):
    capital_grid = get_k_grid(
        numerical_params["n_points_capital_grid"],
        numerical_params["max_value_capital_grid"],
        numerical_params["min_value_capital_grid"],
    )

    assert isinstance(
        capital_grid,
        np.ndarray,
    ), "The function should return a numpy array"
    assert capital_grid.shape == (
        numerical_params["n_points_capital_grid"],
    ), "The shape of the capital grid is incorrect"
    assert np.isclose(
        capital_grid[0],
        numerical_params["min_value_capital_grid"],
    ), "The first element should be close to min_value_capital_grid"
    assert (
        np.isclose(capital_grid[-1], numerical_params["max_value_capital_grid"])
        or capital_grid[-1] < numerical_params["max_value_capital_grid"]
    ), "The last element should be almost equal to or less than max_value_capital_grid"


def test_get_meshes(numerical_params, economic_params):
    economic_params["tax_rate"] = 0

    capital_grid = get_k_grid(
        numerical_params["n_points_capital_grid"],
        numerical_params["max_value_capital_grid"],
        numerical_params["min_value_capital_grid"],
    )

    state_grid = get_state_grid(
        numerical_params["n_points_income_grid"],
        economic_params["unemp_benefit"],
        economic_params["tax_rate"],
        economic_params["productivity"],
    )

    capital_mesh, state_mesh = get_meshes(capital_grid, state_grid)

    assert isinstance(
        capital_mesh,
        np.ndarray,
    ), "The capital_mesh should be a numpy array"
    assert capital_mesh.shape == (
        numerical_params["n_points_capital_grid"],
        numerical_params["n_points_income_grid"],
    ), "The shape of the capital_mesh is incorrect"
    assert isinstance(
        state_mesh,
        np.ndarray,
    ), "The state_mesh should be a numpy array"
    assert state_mesh.shape == (
        numerical_params["n_points_capital_grid"],
        numerical_params["n_points_income_grid"],
    ), "The shape of the state_mesh is incorrect"


def test_produce_grids(numerical_params, economic_params):
    updated_economic_params = produce_grids(economic_params, numerical_params)

    # Check if the returned dictionary has the expected keys
    expected_keys = [
        "beta",
        "alpha",
        "gamma",
        "delta",
        "rho_Z",
        "productivity",
        "unemp_benefit",
        "borrowing_limit",
        "transition_mat",
        "aggregate_labor_sup",
        "employed_share",
        "unemployed_share",
        "tax_rate",
        "state_grid",
        "capital_grid",
        "capital_mesh",
        "state_mesh",
    ]
    for key in expected_keys:
        assert (
            key in updated_economic_params
        ), f"The key '{key}' is missing from the returned dictionary"
