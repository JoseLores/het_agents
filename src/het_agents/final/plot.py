"""Functions plotting results."""

import estimagic as em
import numpy as np
import plotly.graph_objects as go


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
            x=supply_curve,
            y=interest_rate_range,
            mode="lines",
            name="Supply of Funds",
        ),
    )
    fig.add_trace(
        go.Scatter(
            x=demand_curve,
            y=interest_rate_range,
            mode="lines",
            name="Demand for Funds",
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


def plot_criterion_iterations(results):
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
    fig = em.criterion_plot(results, max_evaluations=25, monotone=True)

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


def plot_wealth_distribution(capital_grid, marginal_capital, n_bins=30):
    """This function generates a bar chart of the wealth distribution using the Plotly
    library.

    Args:
        capital_grid (numpy.ndarray): A one-dimensional array representing the capital grid.
        marginal_capital (numpy.ndarray): A one-dimensional array representing the marginal capital distribution.
        n_bins (int): The number of bins to divide the population.

    Returns:
        fig (plotly.graph_objs.Figure): A plotly Figure object containing the bar chart of the wealth distribution.

    """
    total_wealth = capital_grid * marginal_capital
    bin_counts, bin_edges = np.histogram(
        capital_grid, bins=n_bins, weights=total_wealth,
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    fig = go.Figure()
    fig.add_trace(go.Bar(x=bin_centers, y=bin_counts))

    fig.update_layout(
        title="Distribution of Asset Holdings",
        xaxis={"title": "Capital Grid"},
        yaxis={"title": "Marginal Capital"},
        plot_bgcolor="rgba(255, 255, 255, 1)",
    )

    return fig


# TODO: lorenz curve goes above 45-degree line-> fix it
def plot_lorenz_curve(capital_grid, marginal_capital):
    """This function generates a Lorenz curve using the Plotly library.

    Args:
        capital_grid (numpy.ndarray): A one-dimensional array representing the capital grid.
        marginal_capital (numpy.ndarray): A one-dimensional array representing the marginal capital distribution.

    Returns:
        fig (plotly.graph_objs.Figure): A plotly Figure object containing the Lorenz curve.

    """
    total_wealth = capital_grid * marginal_capital
    cumulative_population = np.linspace(0, 1, len(capital_grid))
    cumulative_wealth = np.cumsum(total_wealth) / np.sum(total_wealth)

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
