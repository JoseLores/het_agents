import numpy as np
import pytest


@pytest.fixture()
def numerical_params():
    return {
        "n_points_income_grid": 2,
        "n_points_capital_grid": 100,
        "max_value_capital_grid": 10,
        "min_value_capital_grid": 0,
    }


@pytest.fixture()
def economic_params():
    return {
        "transition_mat": np.array([[0.9, 0.1], [0.2, 0.8]]),
        "productivity": 2,
        "unemp_benefit": 0.5,
    }


def test_get_ergodic_dist():
    transition_mat = np.array([[0.9, 0.1], [0.2, 0.8]])
    ergodic_dist = produce_grids.get_ergodic_dist(transition_mat)
    assert np.allclose(ergodic_dist, np.array([0.35714286, 0.64285714]))


def test_get_aggregate_labor():
    ergodic_dist = np.array([0.35714286, 0.64285714])
    productivity = 2
    (
        aggregate_labor_sup,
        employed_share,
        unemployed_share,
    ) = produce_grids.get_aggregate_labor(ergodic_dist, productivity)
    assert np.isclose(aggregate_labor_sup, 1.28571429)
    assert np.allclose(employed_share, np.array([0.35714, 0.64286]))
    assert np.allclose(unemployed_share, np.array([0.64286, 0.35714]))


def test_get_tax_rate():
    unemp_benefit = 0.5
    employed_share = np.array([0.35714, 0.64286])
    unemployed_share = np.array([0.64286, 0.35714])
    tax_rate = produce_grids.get_tax_rate(
        unemp_benefit,
        employed_share,
        unemployed_share,
    )
    assert np.allclose(tax_rate, np.array([0.70313, 0.19538]))


def test_get_income_grid():
    n_points_z = 10
    unemp_benefit = 0.5
    tax_rate = np.array([0.70313, 0.19538])
    productivity = 2
    income_grid = produce_grids.get_income_grid(
        n_points_z,
        unemp_benefit,
        tax_rate,
        productivity,
    )
    assert np.allclose(
        income_grid,
        np.array(
            [
                0.5,
                0.707345,
                0.91469,
                1.122035,
                1.32938,
                1.536725,
                1.74407,
                1.951415,
                2.15876,
                2.366105,
            ],
        ),
    )


def test_get_k_grid():
    n_points_k = 5
    max_k = 10
    min_k = 1
    capital_grid = produce_grids.get_k_grid(n_points_k, max_k, min_k)
    assert np.allclose(
        capital_grid,
        np.array([1.0, 2.82842712, 4.44948974, 6.12811446, 7.84956392]),
    )


# def test_clean_data_drop_columns(data, data_info):


# def test_clean_data_dropna(data, data_info):


# def test_clean_data_categorical_columns(data, data_info):
#     for cat_col in data_info["categorical_columns"]:


# def test_clean_data_column_rename(data, data_info):


# def test_convert_outcome_to_numerical(data, data_info):
#     assert outcome_numerical_name in data_clean.columns
