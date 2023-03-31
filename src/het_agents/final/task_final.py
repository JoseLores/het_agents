"""Tasks running for the plotting part."""

import pytask

from het_agents.config import BLD, MODELS
from het_agents.final.plot import (
    plot_criterion_iterations,
    plot_lorenz_curve,
    plot_run_times,
    plot_supply_demand_curves,
    plot_wealth_distribution,
)
from het_agents.utilities import read_pkl

for model in MODELS:
    depends_on = {
        "script": ["plot.py"],
        "model": BLD / "python" / "models" / f"results_{model}_mkt.pkl",
        "data": BLD / "python" / "data" / f"econ_data_{model}_mkt.pkl",
    }
    produces = {
        f"{model}_supply_demand_plot": BLD
        / "python"
        / "figures"
        / f"{model}_supply_demand_plot.png",
    }

    @pytask.mark.task(
        id=model,
        kwargs={"model": model, "depends_on": depends_on, "produces": produces},
    )
    def task_plot_economy_python(depends_on, model, produces):
        """Plot the capital markets allocation in the steady state (Python version)."""
        _, economic_params = read_pkl(depends_on["data"])

        _, data_for_plots = read_pkl(depends_on["model"])

        fig_curves = plot_supply_demand_curves(
            data_for_plots["supply_curve"],
            data_for_plots["demand_curve"],
        )
        fig_curves.write_image(produces[f"{model}_supply_demand_plot"])


@pytask.mark.depends_on(
    {
        "script": ["plot.py"],
        "model": BLD / "python" / "models" / "results_standard_mkt.pkl",
        "data": BLD / "python" / "data" / "econ_data_standard_mkt.pkl",
    },
)
@pytask.mark.produces(
    {
        "iterations_plot": BLD / "python" / "figures" / "iterations_plot.png",
        "run_time_plot": BLD / "python" / "figures" / "run_time_plot.png",
        "wealth_distrib_plot": BLD / "python" / "figures" / "wealth_distrib_plot.png",
        "lorenz_plot": BLD / "python" / "figures" / "lorenz_plot.png",
    },
)
def task_plot_benchmarking_python(depends_on, produces):
    """Plot the benchmarking exercise results (Python version)."""
    _, economic_params = read_pkl(depends_on["data"])

    steady_state_results, data_for_plots = read_pkl(depends_on["model"])

    fig_iter = plot_criterion_iterations(
        data_for_plots["benchmarking_iterations_results"],
    )
    fig_iter.write_image(produces["iterations_plot"])

    fig_time = plot_run_times(data_for_plots["benchmarking_time_results"])
    fig_time.write_image(produces["run_time_plot"])

    fig_wealth = plot_wealth_distribution(
        economic_params["capital_grid"],
        steady_state_results["marginal_capital"],
    )
    fig_wealth.write_image(produces["wealth_distrib_plot"])

    fig_lorenz = plot_lorenz_curve(
        economic_params["capital_grid"],
        steady_state_results["marginal_capital"],
    )
    fig_lorenz.write_image(produces["lorenz_plot"])
