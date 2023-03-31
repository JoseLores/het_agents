import numpy as np
import pytest
from het_agents.analysis.solver_functions import *
from het_agents.config import TEST_DIR
from het_agents.utilities import read_pkl


@pytest.fixture()
def params():
    numerical_params, economic_params = read_pkl(
        TEST_DIR / "analysis" / "econ_data_fixture.pkl",
    )
    return economic_params, numerical_params


def test_solve_steady_state(params):
    economic_params, numerical_params = params

    steady_state = solve_steady_state(economic_params, numerical_params)

    # Check that the steady state dictionary has the correct keys
    expected_keys = [
        "aggregate_capital",
        "saving_policy",
        "marginal_capital",
        "capital_distribution",
        "transition_mat",
        "consumption_policy",
    ]
    assert set(steady_state.keys()) == set(
        expected_keys,
    ), "Some keys are missing from results"

    # Check the shape of the arrays in the steady state
    assert steady_state["saving_policy"].shape == (
        numerical_params["n_points_capital_grid"],
        numerical_params["n_points_income_grid"],
    )
    assert steady_state["marginal_capital"].shape == (
        numerical_params["n_points_capital_grid"],
    )
    assert steady_state["capital_distribution"].shape == (
        numerical_params["n_points_capital_grid"]
        * numerical_params["n_points_income_grid"],
    )
    assert steady_state["transition_mat"].shape == (
        numerical_params["n_points_capital_grid"]
        * numerical_params["n_points_income_grid"],
        numerical_params["n_points_capital_grid"]
        * numerical_params["n_points_income_grid"],
    )
    assert steady_state["consumption_policy"].shape == (
        numerical_params["n_points_capital_grid"]
        * numerical_params["n_points_income_grid"],
    )

    assert steady_state["aggregate_capital"] > 0


def test_aiyagari(params):
    economic_params, numerical_params = params

    Rate = 0.02
    Wage = 1.0
    capital_stock, kprime, marginal_k, StDist, Gamma, consumption_grid = aiyagari(
        Rate, Wage, economic_params, numerical_params,
    )

    # Check the shape of the output arrays
    assert kprime.shape == (
        numerical_params["n_points_capital_grid"],
        numerical_params["n_points_income_grid"],
    )
    assert marginal_k.shape == (numerical_params["n_points_capital_grid"],)
    assert StDist.shape == (
        numerical_params["n_points_capital_grid"]
        * numerical_params["n_points_income_grid"],
    )
    assert Gamma.shape == (
        numerical_params["n_points_capital_grid"]
        * numerical_params["n_points_income_grid"],
        numerical_params["n_points_capital_grid"]
        * numerical_params["n_points_income_grid"],
    )
    assert consumption_grid.shape == (
        numerical_params["n_points_capital_grid"]
        * numerical_params["n_points_income_grid"],
    )
    assert np.isclose(capital_stock, 0.04382384816150433)

    # Check that the steady-state marginal distribution realizations sum to 1
    assert np.isclose(np.sum(marginal_k), 1, rtol=1e-5)


def test_endogenous_grid_method(params):
    economic_params, numerical_params = params

    Wage = 1.3958498978115297
    Rate = 0.01
    Rateprime = 0.01
    economic_params["income_mesh"] = economic_params["state_mesh"] * Wage

    consumption_grid = np.random.rand(
        numerical_params["n_points_income_grid"]
        * numerical_params["n_points_capital_grid"],
    )

    consumption_result, Kprime_result = endogenous_grid_method(
        consumption_grid,
        Rate,
        Rateprime,
        economic_params,
        numerical_params,
    )

    assert consumption_result.shape == (
        numerical_params["n_points_income_grid"]
        * numerical_params["n_points_capital_grid"],
    )

    assert Kprime_result.shape == (
        numerical_params["n_points_capital_grid"],
        numerical_params["n_points_income_grid"],
    )

    assert np.all(consumption_result >= 0)
    assert np.all(Kprime_result >= economic_params["borrowing_limit"])


def test_young(params):
    economic_params, numerical_params = params

    Rate = 0.02
    Wage = 1.0
    capital_stock, kprime, marginal_k, StDist, Gamma, consumption_grid = aiyagari(
        Rate, Wage, economic_params, numerical_params,
    )

    # Test the young function separately
    Gamma_test, StDist_test = young(kprime, economic_params, numerical_params)

    # Check the shape of the output arrays
    assert Gamma_test.shape == (
        numerical_params["n_points_capital_grid"]
        * numerical_params["n_points_income_grid"],
        numerical_params["n_points_capital_grid"]
        * numerical_params["n_points_income_grid"],
    )
    assert StDist_test.shape == (
        numerical_params["n_points_capital_grid"]
        * numerical_params["n_points_income_grid"],
    )

    assert np.isclose(np.sum(StDist_test), 1, rtol=1e-5)

    for row in Gamma_test:
        assert np.isclose(np.sum(row), 1, rtol=1e-5)

    assert np.allclose(Gamma_test, Gamma, atol=1e-8, rtol=1e-5)
    assert np.allclose(StDist_test, StDist, atol=1e-8, rtol=1e-5)


def test_rate(params):
    economic_params, numerical_params = params

    capital = 10.0
    expected_rate = (
        economic_params["alpha"]
        * economic_params["aggregate_labor_sup"] ** (1 - economic_params["alpha"])
        * capital ** (economic_params["alpha"] - 1)
        - economic_params["delta"]
    )
    assert np.isclose(rate(capital, economic_params, numerical_params), expected_rate)


def test_excess_demand(params):
    economic_params, numerical_params = params

    capital = 6.25
    result = excess_demand(capital, economic_params, numerical_params)

    assert isinstance(result, float | np.float64)
    assert np.isclose(result, -6.25)


def test_capital_demand(params):
    economic_params, numerical_params = params

    return_rate = 0.0285
    result = capital_demand(return_rate, economic_params, numerical_params)

    assert isinstance(result, float | np.float64)
    assert np.isclose(result, 5.001053729433481)


def test_wage(params):
    economic_params, numerical_params = params

    capital = 10.0
    expected_wage = (
        (1 - economic_params["alpha"])
        * economic_params["aggregate_labor_sup"] ** (-economic_params["alpha"])
        * capital ** (economic_params["alpha"])
    )
    assert np.isclose(wage(capital, economic_params, numerical_params), expected_wage)


def test_get_marginal_utility(params):
    economic_params, numerical_params = params

    consumption = np.array([1.0, 2.0, 3.0])
    gamma = economic_params["gamma"]
    expected_mu = 1 / (consumption**gamma)

    assert np.allclose(
        get_marginal_utility(consumption, gamma),
        expected_mu,
    )


def test_get_inverse_marginal_utility(params):
    economic_params, numerical_params = params

    marginal_utility = np.array([0.5, 0.2, 0.1])
    gamma = economic_params["gamma"]
    expected_c = (1 / marginal_utility) ** (1 / gamma)

    assert np.allclose(
        get_inverse_marginal_utility(marginal_utility, gamma),
        expected_c,
    )
