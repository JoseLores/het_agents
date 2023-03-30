"""Tasks running the core analyses."""

import pytask

from het_agents.analysis.model import (
    benchmark_algorithms_iterations,
    benchmark_algorithms_time,
    capital_demand_supply,
)
from het_agents.config import ALGORITHMS, BLD, MODELS
from het_agents.solver_functions import rate, solve_steady_state
from het_agents.utilities import read_pkl, to_pkl

for model in MODELS:

    @pytask.mark.depends_on(
        {
            "script": ["model.py"],
            "data": BLD / "python" / "data" / f"econ_data_{model}_mkt.pkl",
        },
    )
    @pytask.mark.produces(BLD / "python" / "models" / f"results_{model}_mkt.pkl")
    def task_get_model_results_python(depends_on, produces):
        """Get quantitative data of the model (Python version)."""
        data_for_plots = {}

        numerical_params, economic_params = read_pkl(depends_on["data"])

        steady_state_results = solve_steady_state(economic_params, numerical_params)

        steady_state_results["interest_rate"] = rate(
            steady_state_results["aggregate_capital"],
            economic_params,
            numerical_params,
        )

        (
            data_for_plots["demand_curve"],
            data_for_plots["supply_curve"],
        ) = capital_demand_supply(economic_params, numerical_params)

        data_for_plots[
            "benchmarking_iterations_results"
        ] = benchmark_algorithms_iterations(
            ALGORITHMS,
            economic_params,
            numerical_params,
        )

        data_for_plots["benchmarking_time_results"] = benchmark_algorithms_time(
            ALGORITHMS,
            economic_params,
            numerical_params,
        )

        to_pkl(steady_state_results, data_for_plots, produces)
