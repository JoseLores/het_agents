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
        "data": BLD / "python" / "models" / f"results_{model}_mkt.pkl",
    }
    produces = {
        f"{model}_supply_demand_plot": BLD
        / "python"
        / "figures"
        / f"{model}_supply_demand_plot.png",
        f"{model}_wealth_distrib_plot": BLD
        / "python"
        / "figures"
        / f"{model}_wealth_distrib_plot.png",
        f"{model}_lorenz_plot": BLD / "python" / "figures" / f"{model}_lorenz_plot.png",
    }

    @pytask.mark.task(
        id=model,
        kwargs={"model": model, "depends_on": depends_on, "produces": produces},
    )
    def task_plot_economy_python(depends_on, model, produces):
        """Plot the capital markets allocation in the steady state (Python version)."""
        data_for_plots, steady_state_results = read_pkl(depends_on["data"])

        fig_curves = plot_supply_demand_curves(
            data_for_plots["supply_curve"],
            data_for_plots["demand_curve"],
            steady_state_results["interest_rate"],
        )
        fig_curves.save(produces[f"{model}_supply_demand_plot"])

        plot_wealth_distribution(
            steady_state_results["capital_distribution"],
        )
        fig.save(produces[f"{model}_supply_demand_plot"])

        fig_lorenz = plot_lorenz_curve(steady_state_results["capital_distribution"])
        fig_lorenz.save(produces[f"{model}_lorenz_plot"])


@pytask.mark.depends_on(
    {
        "script": ["plot.py"],
        "data": BLD / "python" / "models" / "results_standard_mkt.pkl",
    },
)
@pytask.mark.produces(
    {
        "iterations_plot": BLD / "python" / "figures" / "iterations_plot.png",
        "run_time_plot": BLD / "python" / "figures" / "run_time_plot.png",
    },
)
def task_plot_benchmarking_python(depends_on, produces):
    """Plot the benchmarking exercise results (Python version)."""
    data_for_plots, _ = read_pkl(depends_on["data"])

    fig_iter = plot_criterion_iterations(
        data_for_plots["benchmarking_iterations_results"],
    )
    fig_iter.save(produces["iterations_plot"])

    fig_time = plot_run_times(data_for_plots["benchmarking_time_results"])
    fig_time.save(produces["run_time_plot"])
