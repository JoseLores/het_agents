"""Functions plotting results."""

import plotly.graph_objects as go
import estimagic as em
import numpy as np


def plot_supply_demand_curves(supply_curve, demand_curve):
    """Plots the supply and demand curves for funds.

    Args:
        supply_curve (numpy.ndarray): The supply of funds curve.
        demand_curve (numpy.ndarray): The demand for funds curve.
    Outputs:
        fig (plotly.graph_objects.Figure): the figure to be saved.

    """
    # Create a range of interest rates for the x-axis from -0.01 to 0.052
    interest_rate_range = np.linspace(-0.01, 0.04, len(supply_curve))

    # Create a plotly figure
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=supply_curve, y=interest_rate_range, mode="lines", name="Supply of Funds",
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=demand_curve, y=interest_rate_range, mode="lines", name="Demand for Funds",
        ),
    )

    # Set labels for the plot
    fig.update_layout(
        xaxis_title="Funds",
        yaxis_title="Interest Rate",
        title="Supply and Demand Curves for Funds",
        font={"size": 14},
        plot_bgcolor="white",
        xaxis={"linecolor": "black"},
        yaxis={"linecolor": "black"},
    )

    return fig


def plot_criterion_iterations(results, max_evaulations=30):
    """This function generates a plot of the criterion value across iterations for
    different optimization algorithms using the estimagic criterion plot function. The
    plot is limited to a specified number of evaluations.

    Args:
        results (dict): A dictionary containing the results of the optimization process for different algorithms.
            The keys of the dictionary are the names of the optimization algorithms, and the values are the
            optimization result objects.
        max_evaluations (int, optional): The maximum number of evaluations to display on the criterion plot. Defaults
            to 30.

    Returns:
        fig (plotly.graph_objs.Figure): A plotly Figure object containing the criterion plot with the specified
            maximum number of evaluations.

    """
    fig = em.criterion_plot(results, max_evaulations, monotone=True)

    return fig


def plot_run_times(runtimes):
    """This function generates a bar chart of the average runtimes for different
    optimization algorithms using the Plotly library.

    Args:
        runtimes (dict): A dictionary containing the average runtime of the benchmark for each solver.
            The keys of the dictionary are the names of the optimization algorithms, and the values are
            the average runtimes.

    Returns:
        fig (plotly.graph_objs.Figure): A plotly Figure object containing the bar chart of the average runtimes
            for different optimization algorithms.

    """
    algorithms = list(runtimes.keys())
    avg_runtimes = list(runtimes.values())

    trace = go.Bar(x=algorithms, y=avg_runtimes, text=avg_runtimes, textposition="auto")

    layout = go.Layout(
        title="Average Runtimes for Different Optimization Algorithms",
        xaxis={"title": "Optimization Algorithm"},
        yaxis={"title": "Average Runtime (seconds)"},
        plot_bgcolor="rgba(255, 255, 255, 1)",
    )

    fig = go.Figure(data=[trace], layout=layout)

    return fig


def plot_wealth_distribution(StDist, n_bins=5):
    """This function generates a bar chart of the wealth distribution using the Plotly
    library.

    Args:
        StDist (numpy.ndarray): A one-dimensional array representing the stationary distribution of income and wealth.
        n_bins (int): The number of bins to divide the population.

    Returns:
        fig (plotly.graph_objs.Figure): A plotly Figure object containing the bar chart of the wealth distribution.

    """
    StDist_percent = StDist * 100
    bin_size = len(StDist) // n_bins
    wealth_distribution = [
        np.sum(StDist_percent[i * bin_size : (i + 1) * bin_size]) for i in range(n_bins)
    ]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=list(range(1, n_bins + 1)), y=wealth_distribution))

    fig.update_layout(
        title="Wealth Distribution",
        xaxis={"title": f"Wealth Percentile ({100/n_binds}% bins)"},
        yaxis={"title": "Percentage of Total Wealth"},
        plot_bgcolor="rgba(255, 255, 255, 1)",
    )

    return fig


def plot_lorenz_curve(StDist, n_bins=5):
    """This function generates a Lorenz curve using the Plotly library.

    Args:
        StDist (numpy.ndarray): A one-dimensional array representing the stationary distribution of income and wealth.
        n_bins (int): The number of bins to divide the population.

    Returns:
        fig (plotly.graph_objs.Figure): A plotly Figure object containing the Lorenz curve.

    """
    bin_size = len(StDist) // n_bins
    cumulative_wealth = np.array(
        [np.sum(StDist[: (i + 1) * bin_size]) for i in range(n_bins)],
    )
    cumulative_population = np.linspace(0, 1, n_bins)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=cumulative_population,
            y=cumulative_wealth,
            name="Lorenz Curve",
            mode="lines",
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=cumulative_population,
            y=cumulative_population,
            name="Line of Equality",
            mode="lines",
            line={"dash": "dash"},
        ),
    )

    fig.update_layout(
        title="Lorenz Curve",
        xaxis={"title": "Cumulative Population"},
        yaxis={"title": "Cumulative Wealth"},
        plot_bgcolor="rgba(255, 255, 255, 1)",
    )

    return fig
