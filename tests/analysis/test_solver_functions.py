import numpy as np
import pytest
from het_agents.analysis.solver_functions import *
from het_agents.config import TEST_DIR
from het_agents.utilities import read_pkl
from scipy.interpolate import interp1d


@pytest.fixture()
def params():
    numerical_params, economic_params = read_pkl(
        TEST_DIR / "analysis" / "econ_data_fixture.pkl",
    )
    return economic_params, numerical_params


# TODO: OPEN DATA IN NOTEBOOK AND ASSERT IT PRODUCES THE SAME RESULTS
def test_excess_demand(params):
    economic_params, numerical_params = params

    capital = 4.0
    result = excess_demand(capital, economic_params, numerical_params)

    assert isinstance(result, float | np.float64)


def test_capital_demand(params):
    economic_params, numerical_params = params

    return_rate = 0.04
    result = capital_demand(return_rate, economic_params, numerical_params)

    assert isinstance(result, float | np.float64)


def test_endogenous_grid_method(params):
    economic_params, numerical_params = params

    consumption_grid = np.random.rand(
        numerical_params["n_points_income_grid"]
        * numerical_params["n_points_capital_grid"],
    )
    Rate = 0.04
    Rateprime = 0.05

    consumption_result, Kprime_result = endogenous_grid_method(
        consumption_grid,
        Rate,
        Rateprime,
        economic_params,
        numerical_params,
    )

    # Test the shape of the consumption grid
    assert consumption_result.shape == (
        numerical_params["n_points_income_grid"]
        * numerical_params["n_points_capital_grid"],
    )

    # Test the shape of the Kprime grid
    assert Kprime_result.shape == (
        numerical_params["n_points_capital_grid"],
        numerical_params["n_points_income_grid"],
    )

    # Test that the consumption grid contains only positive values
    assert np.all(consumption_result >= 0)

    # Test that the Kprime grid satisfies the borrowing constraint
    assert np.all(Kprime_result >= economic_params["borrowing_limit"])

    # Test that the consumption function is consistent with the savings function
    Kstar = (
        consumption_result.reshape(
            (
                numerical_params["n_points_capital_grid"],
                numerical_params["n_points_income_grid"],
            ),
            order="F",
        )
        + economic_params["capital_mesh"]
        - economic_params["income_mesh"]
    ) / Rateprime

    for z in range(numerical_params["n_points_income_grid"]):
        Savings = interp1d(
            Kstar[:, z],
            economic_params["capital_grid"],
            kind="linear",
            fill_value="extrapolate",
        )
        Kprime_savings = Savings(economic_params["capital_grid"])
        assert np.allclose(Kprime_result[:, z], Kprime_savings, atol=1e-8, rtol=1e-5)
