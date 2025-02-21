"""
This module contains functions to plot feature effects, model evaluation results, and feature effect errors.
"""

from typing_extensions import Dict, Literal, List, Tuple, Optional
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cm
import seaborn as sns

from current_research_feature_effects.feature_effects import FeatureEffect, compute_theoretical_effects
from current_research_feature_effects.data_generating.data_generation import Groundtruth
from current_research_feature_effects.plotting.utils import (
    set_style,
    get_boxplot_style,
    get_feature_effect_plot_style,
    set_fontsize,
)


def plot_feature_effect(effect: FeatureEffect | List[FeatureEffect], labels: List[str] = None) -> plt.Figure:
    """
    Plot feature effects for a given feature effect object.

    Parameters
    ----------
    effect : FeatureEffect | List[FeatureEffect]
        A FeatureEffect object or a list of FeatureEffect objects to be plotted.
    labels : List[str], optional
        A list of labels for the different FeatureEffect objects (default is None).

    Returns
    -------
    plt.Figure
        A Figure object containing the generated plots.
    """
    if isinstance(effect, list):
        n_features = len(effect[0].features)
        fig, axs = plt.subplots(1, n_features, figsize=(5 * n_features, 5))
        for i, feature in enumerate(effect[0].features):
            for e in effect:
                axs[i].plot(
                    e.features[feature]["grid"], e.features[feature]["effect"], **get_feature_effect_plot_style()
                )
            axs[i].set_title(f"{effect[0].effect_type.upper()} of ${feature}$")

        if labels is not None:
            axs[-1].legend(labels)

        return fig

    n_features = len(effect.features)
    fig, axs = plt.subplots(1, n_features, figsize=(5 * n_features, 5))
    for i, feature in enumerate(effect.features):
        axs[i].plot(
            effect.features[feature]["grid"], effect.features[feature]["effect"], **get_feature_effect_plot_style()
        )
        axs[i].set_title(f"{effect.effect_type.upper()} of ${feature}$")

    return fig


def plot_feature_effect_vs_groundtruth(
    effect: FeatureEffect,
    groundtruth: FeatureEffect,
    feature_names: List[str],
    large_font: bool = False,
    save_figs: None | Path = None,
) -> plt.Figure:
    """
    Plot feature effects against groundtruth for a given feature effect object.

    Parameters
    ----------
    effect : FeatureEffect
        The FeatureEffect object to be plotted.
    groundtruth : FeatureEffect
        The FeatureEffect object representing the groundtruth.
    feature_names : List[str]
        A list of feature names to plot.
    large_font : bool, optional
        If True, use larger font sizes (default is False).
    save_figs : None | Path
        Path to save the figure (default is None).

    Returns
    -------
    plt.Figure
        A Figure object containing the generated plots.
    """
    sns.set(style="ticks")
    palette = sns.color_palette("Set2")

    if large_font:
        set_fontsize("xlarge")
    else:
        set_fontsize("standard")

    fig, axes = plt.subplots(1, len(feature_names), figsize=(5 * len(feature_names), 5), dpi=300, sharey=True)
    for i, f in enumerate(feature_names):
        sns.lineplot(
            x=effect.features[f]["grid"],
            y=effect.features[f]["effect"],
            label=f"{effect.effect_type.upper()} estimate",
            color=palette[1],
            ax=axes[i],
            marker="+",
            markeredgecolor=palette[1],
        )

        sns.lineplot(
            x=groundtruth.features[f]["grid"],
            y=groundtruth.features[f]["effect"],
            label=f"{groundtruth.effect_type.upper()} theoretical",
            color=palette[0],
            ax=axes[i],
        )

        axes[i].set_xlabel(f"${f}$")
        axes[i].set_ylabel(f"{effect.effect_type.upper()}")
        axes[i].legend()
        sns.despine(ax=axes[i])

    plt.tight_layout()

    if save_figs is not None:
        fig.savefig(save_figs / f"{effect.effect_type}_groundtruth_comparison.png", bbox_inches="tight")

    return fig


def boxplot_model_results(
    metric: Literal["mse", "mae", "r2"],
    df: pd.DataFrame,
    ylim: Optional[Tuple[float, float]] = None,
    large_font: bool = False,
) -> plt.Figure:
    """
    Create a boxplot of the model evaluation results.

    Parameters
    ----------
    metric : Literal[&quot;mse&quot;, &quot;mae&quot;, &quot;r2&quot;]
        The metric to plot.
    df : pd.DataFrame
        The dataframe containing the model evaluation results.
    ylim : Optional[Tuple[float, float]], optional
        The y-axis limits, by default None.
    large_font : bool, optional
        If True, use larger font sizes, by default False.

    Returns
    -------
    plt.Figure
        The figure object containing the generated plot.
    """
    set_style()
    if large_font:
        set_fontsize("large")
    else:
        set_fontsize("standard")
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), dpi=300, sharey=True)
    fig.suptitle("Model evaluation", fontsize=16, fontweight="bold")
    ax[0].set_title(f"{metric} on train set")
    sns.boxplot(
        x="n_train",
        y=f"{metric}_train",
        hue="model",
        data=df,
        ax=ax[0],
        **get_boxplot_style(),
    )
    ax[0].legend().set_visible(False)
    ax[0].set_ylabel(metric)
    if ylim is not None:
        ax[0].set_ylim(ylim)
    sns.boxplot(
        x="n_train",
        y=f"{metric}_test",
        hue="model",
        data=df,
        ax=ax[1],
        **get_boxplot_style(),
    )
    ax[1].set_title(f"{metric} on test set")
    ax[1].legend(title="Learner", bbox_to_anchor=(1.05, 1), loc="upper left")
    fig.tight_layout()

    return fig


def plot_feature_effect_error_table(
    df: pd.DataFrame,
    models: List[str],
    type: Literal["pdp", "ale"],
    save_figs: None | Path = None,
    large_font: bool = False,
    show_title: bool = True,
):
    """
    Plot a table of feature effect errors for different models.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the aggregated feature effect errors.
    models : List[str]
        List of model names.
    type : Literal[&quot;pdp&quot;, &quot;ale&quot;]
        Type of feature effect, either 'pdp' or 'ale'.
    save_figs : None | Path
        Path to save the figure (default is None).
    large_font : bool
        If True, use larger font sizes (default is False).
    show_title : bool
        If True, show the title of the plot (default is True).
    """
    for model in models:
        if large_font:
            set_fontsize("xlarge")
        else:
            set_fontsize("standard")

        g = sns.FacetGrid(
            df.loc[df["model"] == model], row="n_train", col="feature", height=3, sharey="row", aspect=0.67
        )
        g.map_dataframe(
            sns.barplot, x="split", y="value", hue="metric", hue_order=["MSE", "Bias^2", "Variance"], palette="Set2"
        )
        g.set_titles(col_template="${col_name}$", row_template="n_train={row_name}")
        if show_title:
            g.fig.suptitle(f"Feature Effect Errors {type.upper()} {model}", y=1.02)
            legend_pos = (0.5, 1.0)
        else:
            g.fig.suptitle("", y=1.02)
            legend_pos = (0.5, 1.05)
        g.add_legend(loc="upper center", bbox_to_anchor=legend_pos, ncol=3)
        g.set_ylabels("")
        plt.tight_layout()
        if save_figs is not None:
            g.savefig(save_figs / f"feature_effect_errors_{type}_{model}.png", bbox_inches="tight", dpi=300)

    plt.show()


def plot_variance_table(
    df: pd.DataFrame,
    models: List[str],
    type: Literal["pdp", "ale"],
    save_figs: None | Path = None,
    large_font: bool = False,
    show_title: bool = True,
):
    """
    Plot a table of feature effect variances for different models.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the aggregated variances.
    models : List[str]
        List of model names.
    type : Literal[&quot;pdp&quot;, &quot;ale&quot;]
        Type of feature effect, either 'pdp' or 'ale'.
    save_figs : None | Path
        Path to save the figure (default is None).
    large_font : bool
        If True, use larger font sizes (default is False).
    show_title : bool
        If True, show the title of the plot (default is True).
    """
    for model in models:
        if large_font:
            set_fontsize("xlarge")
        else:
            set_fontsize("standard")

        g = sns.FacetGrid(
            df.loc[df["model"] == model], row="n_train", col="feature", height=3, sharey="row", aspect=0.67
        )
        g.map_dataframe(
            sns.barplot,
            x="split",
            y="value",
            hue="metric",
            hue_order=["Variance", "Model Variance", "MC Variance"],
            palette=sns.color_palette("Set2")[2:5],
        )
        for ax in g.axes.flat:
            for bar in ax.patches:
                if bar.get_height() < 0:
                    bar.set_hatch("////")
                    bar.set_edgecolor((0, 0, 0, 0.3))
        g.set_titles(col_template="${col_name}$", row_template="n_train={row_name}")
        if show_title:
            g.fig.suptitle(f"Variance Decomposition {type.upper()} {model}", y=1.02)
            legend_pos = (0.5, 1.0)
        else:
            g.fig.suptitle("", y=1.02)
            legend_pos = (0.5, 1.05)
        g.add_legend(loc="upper center", bbox_to_anchor=legend_pos, ncol=3)
        g.set_ylabels("")
        plt.tight_layout()
        if save_figs is not None:
            g.savefig(save_figs / f"variance_decomposition_{type}_{model}.png", bbox_inches="tight", dpi=300)

    plt.show()


def plot_fe_bias_variance(error_dict, sharey=True, large_font=False) -> plt.Figure:
    """
    Plot the bias and variance of the feature effects over the grid points.

    Parameters
    ----------
    error_dict : Dict
        A dictionary containing the feature effect errors.
    sharey : bool, optional
        If True, share the y-axis across all subplots (default is True).
    large_font : bool, optional
        If True, use larger font sizes (default is False).

    Returns
    -------
    plt.Figure
        A Figure object containing the generated plots.
    """
    sns.set_theme(style="ticks")
    palette = sns.color_palette("Set2")

    if large_font:
        set_fontsize("xlarge")
    else:
        set_fontsize("standard")

    n_features = len(list(list(error_dict.values())[0].values())[0].features)
    fig, axes = plt.subplots(len(error_dict), n_features, figsize=(5 * n_features, 5 * len(error_dict)), sharey=sharey)

    colors = {"Bias^2": palette[1], "Variance": palette[2]}  # Orange  # Purple

    legend_lines = []
    legend_labels = []

    for i, (split, metrics) in enumerate(error_dict.items()):
        for metric in ["Bias^2", "Variance"]:
            for j, feature in enumerate(metrics[metric].features):
                line = axes[i, j].plot(
                    metrics[metric].features[feature]["grid"],
                    metrics[metric].features[feature]["effect"],
                    linewidth=2,
                    color=colors[metric],
                    marker="+",
                )[0]

                if i == 0 and j == 0:
                    legend_lines.append(line)
                    legend_labels.append(metric)

                axes[i, j].set_title(f"{split}: {metrics[metric].effect_type.upper()} of ${feature}$", pad=10)
                axes[i, j].set_xlabel(f"${feature}$")

                sns.despine(ax=axes[i, j])

    fig.legend(
        legend_lines,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.0),
        ncol=2,
        frameon=False,
        fontsize=18 if large_font else 12,
    )

    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    fig.set_dpi(200)
    sns.reset_orig()

    return fig


def plot_mcvariance_over_features(
    mc_variance_data: Dict[float, FeatureEffect],
    feature_names: List[str],
    title: Optional[str] = None,
    sharey: bool = False,
    groundtruth: Optional[Groundtruth] = None,
    large_font: bool = False,
    save_figs: None | Path = None,
) -> plt.Figure:
    """
    Plot Monte Carlo variance over features for different sample sizes, optionally with groundtruth.

    This function creates a series of line plots, each representing the Monte Carlo
    variance of a feature effect estimate across different sample sizes. The x-axis
    represents the feature values, while the y-axis represents the Monte Carlo variance.
    Different sample sizes are represented by different colors. If groundtruth is provided,
    it plots the theoretical effect without showing its scale.

    Parameters
    ----------
    mc_variance_data : Dict
        A dictionary containing the Monte Carlo variance data for different sample sizes
        and features.
    feature_names : List[str]
        A list of feature names to plot.
    title : str, optional
        The title of the plot (default is None).
    sharey : bool, optional
        Whether to share y-axes across subplots (default is False).
    groundtruth : Groundtruth, optional
        The Groundtruth object to plot theoretical effects (default is None).
    large_font : bool, optional
        If True, use larger font sizes (default is False).
    save_figs : None | Path
        Path to save the figure (default is None).

    Returns
    -------
    plt.Figure
        A Figure object containing the generated plots.
    """
    set_style()
    n_feat = len(feature_names)

    if large_font:
        set_fontsize("large")
    else:
        set_fontsize("standard")

    fig = plt.figure(figsize=(4 * n_feat + 0.5, 4))

    gs = fig.add_gridspec(1, n_feat + 1, width_ratios=[1] * n_feat + [0.1])
    axes = [fig.add_subplot(gs[0, i]) for i in range(n_feat)]
    cax = fig.add_subplot(gs[0, -1])  # colorbar axis

    if title is not None:
        fig.suptitle(title, y=1.02, fontsize=14, fontweight="bold")

    n_samples = list(mc_variance_data.keys())
    norm = colors.LogNorm(vmin=min(n_samples), vmax=max(n_samples))
    cmap = cm.viridis

    if groundtruth is not None:
        groundtruth_effect = compute_theoretical_effects(
            groundtruth,
            mc_variance_data[n_samples[0]].effect_type,
            feature_names,
            [mc_variance_data[n_samples[0]].features[feature]["grid"] for feature in feature_names],
            center_curves=True,
        )

    for feature, ax in zip(feature_names, axes):
        for n in n_samples:
            color = cmap(norm(n))
            ax.plot(
                mc_variance_data[n].features[feature]["grid"],
                mc_variance_data[n].features[feature]["effect"],
                color=color,
            )
            ax.fill_between(
                mc_variance_data[n].features[feature]["grid"],
                mc_variance_data[n].features[feature]["effect"],
                alpha=0.1,
                color=color,
            )
        ax.set_title(f"MC Variance ${feature}$")
        ax.set_xlabel(f"${feature}$")
        ax.set_ylabel("MC Variance")

        if groundtruth is not None:
            ax2 = ax.twinx()
            ax2.plot(
                groundtruth_effect.features[feature]["grid"],
                groundtruth_effect.features[feature]["effect"],
                color="grey",
                linestyle="--",
                label="groundtruth",
            )
            ax2.set_yticklabels([])
            ax2.set_ylabel("")
            ax2.grid(False)
            ax2.legend(loc="upper center")

    if sharey:
        y_min = min(ax.get_ylim()[0] for ax in axes)
        y_max = max(ax.get_ylim()[1] for ax in axes)
        for ax in axes:
            ax.set_ylim(y_min, y_max)

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    plt.colorbar(sm, cax=cax, label="$n_{mc}$")

    plt.tight_layout()

    if save_figs is not None:
        effect_type = list(mc_variance_data.values())[0].effect_type
        fig.savefig(save_figs / f"mc_variance_{effect_type}.png", bbox_inches="tight")

    return fig


def plot_mcvariance_mean(
    mc_variance_data: Dict[float, FeatureEffect],
    feature_names: List[str],
    title: Optional[str] = None,
    sharey: bool = False,
    xscale: Literal["linear", "log"] = "linear",
    yscale: Literal["linear", "log"] = "log",
    large_font: bool = False,
    save_figs: None | Path = None,
) -> plt.Figure:
    """
    Plot the mean Monte Carlo variance over features for different sample sizes.

    This function creates a series of line plots, each representing the mean Monte Carlo
    variance of a feature effect estimate across different sample sizes. The x-axis
    represents the sample sizes, while the y-axis represents the mean Monte Carlo variance.

    Parameters
    ----------
    mc_variance_data : Dict
        A dictionary containing the Monte Carlo variance data for different sample sizes
        and features.
    feature_names : List[str]
        A list of feature names to plot.
    title : str, optional
        The title of the plot (default is None).
    sharey : bool, optional
        If True, share the y-axis across all subplots (default is False).
    xscale : {'linear', 'log'}, optional
        The scale of the x-axis (default is 'linear').
    yscale : {'linear', 'log'}, optional
        The scale of the y-axis (default is 'log').
    large_font : bool, optional
        If True, use larger font sizes (default is False).
    save_figs : None | Path
        Path to save the figure (default is None).

    Returns
    -------
    plt.Figure
        A Figure object containing the generated plots.
    """
    set_style()
    palette = sns.color_palette("Set2")

    if large_font:
        set_fontsize("large")
    else:
        set_fontsize("standard")

    n_feat = len(feature_names)
    fig, axes = plt.subplots(
        1,
        n_feat,
        figsize=(5 * n_feat, 5),
        sharey=sharey,
        dpi=300,
    )
    fig.suptitle(title, y=1.02, fontsize=14, fontweight="bold")

    for feature, ax in zip(feature_names, axes):
        means = []
        for k in mc_variance_data.keys():
            means.append(np.mean(mc_variance_data[k].features[feature]["effect"]))
        ax.plot(mc_variance_data.keys(), means, marker="+", color=palette[4])
        ax.set_yscale(yscale)
        ax.set_xscale(xscale)
        ax.set_title(f"Mean MC Variance ${feature}$")
        ax.set_xlabel("$n_{mc}$")
        ax.set_ylabel("Mean MC Variance")

    plt.tight_layout()

    if save_figs is not None:
        effect_type = list(mc_variance_data.values())[0].effect_type
        fig.savefig(save_figs / f"mean_mc_variance_{effect_type}.png", bbox_inches="tight")

    return fig
