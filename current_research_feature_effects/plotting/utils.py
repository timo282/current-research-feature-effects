"""
This module contains utility functions for plotting.
"""

from typing import Literal
import matplotlib.pyplot as plt


def set_style():
    """
    Set the style of the plots.
    """
    plt.style.use("fivethirtyeight")

    plt.rcParams.update(
        {
            "axes.titlesize": 12,  # Smaller title size
            "axes.labelsize": 10,  # Smaller axis labels size
            "xtick.labelsize": 8,  # Smaller x-axis tick labels size
            "ytick.labelsize": 8,  # Smaller y-axis tick labels size
            "legend.fontsize": 10,  # Smaller legend font size
            "font.size": 10,  # This sets the overall default font size
            "grid.linewidth": 0.5,  # Thin grid lines
            "figure.facecolor": "white",  # White background color
            "figure.dpi": 300,  # Higher resolution
            "axes.facecolor": "white",  # White background color
            "axes.edgecolor": "white",  # White background edge color
            "lines.linewidth": 1.5,  # Thin edge linewidth
        }
    )


def get_boxplot_style():
    """
    Set the style of the boxplot.

    Returns
    -------
    Dict
        Dictionary containing the style of the boxplot.
    """
    style = dict(
        boxprops=dict(edgecolor="black"),  # Box properties
        whiskerprops=dict(color="black"),  # Whisker properties
        capprops=dict(color="black"),  # Cap properties
        medianprops=dict(color="black", linewidth=1.5),  # Median properties
        flierprops=dict(marker="o", markeredgecolor="black", markersize=5, linestyle="none"),
        palette="Set2",
    )

    return style


def get_feature_effect_plot_style():
    """
    Set the style of the feature effect plot.

    Returns
    -------
    Dict
        Dictionary containing the style of the feature effect plot.
    """
    style = dict(linewidth=2, marker="+", markeredgewidth=1, markersize=5)

    return style


def set_fontsize(size: Literal["standard", "large", "xlarge"]):
    """
    Set the font size of the plots.

    Parameters
    ----------
    size : {'standard', 'large', 'xlarge'}
        The font size to set.
    """
    if size == "xlarge":
        plt.rcParams.update(
            {
                "font.size": 16,
                "axes.labelsize": 18,
                "axes.titlesize": 20,
                "figure.titlesize": 22,
                "xtick.labelsize": 16,
                "ytick.labelsize": 16,
                "legend.fontsize": 16,
            }
        )
    elif size == "large":
        plt.rcParams.update(
            {
                "font.size": 14,
                "axes.labelsize": 16,
                "axes.titlesize": 18,
                "figure.titlesize": 20,
                "xtick.labelsize": 14,
                "ytick.labelsize": 14,
                "legend.fontsize": 14,
            }
        )
    else:
        plt.rcParams.update(
            {
                "font.size": 11,
                "axes.labelsize": 12,
                "axes.titlesize": 12,
                "figure.titlesize": 16,
                "xtick.labelsize": 10,
                "ytick.labelsize": 10,
                "legend.fontsize": 12,
            }
        )
