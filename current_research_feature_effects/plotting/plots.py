# from configparser import ConfigParser
# import math
from typing_extensions import Dict, Literal, List, Tuple, Optional
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# from matplotlib import transforms
import matplotlib.colors as colors
import matplotlib.cm as cm
import seaborn as sns

# from sklearn.base import BaseEstimator
# from scipy.stats import pearsonr, spearmanr

from current_research_feature_effects.feature_effects import FeatureEffect
from current_research_feature_effects.data_generating.data_generation import Groundtruth
from current_research_feature_effects.plotting.utils import (
    set_style,
    get_boxplot_style,
    get_feature_effect_plot_style,
)

# from current_research_feature_effects.feature_effects import compute_pdps, compute_ales


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


def plot_groundtruth(groundtruth: Groundtruth, grid_points: Dict[str, np.ndarray], effect_type: Literal["pdp", "ale"]):
    """
    Plot the theoretical groundtruth feature effects.

    Parameters
    ----------
    groundtruth : Groundtruth
        The Groundtruth object to be plotted.
    grid_points : Dict[str, np.ndarray]
        A dictionary containing the grid points for each feature.
    effect_type : {'pdp', 'ale'}
        The type of effect to plot, either 'pdp' or 'ale'.

    Returns
    -------
    plt.Figure
        A Figure object containing the generated plots.
    """
    fe = []
    for feature_name in groundtruth.feature_names:
        if effect_type == "pdp":
            effect_func = groundtruth.get_theoretical_partial_dependence(feature_name)
        elif effect_type == "ale":
            effect_func = groundtruth.get_theoretical_accumulated_local_effects(feature_name)

        fe.append(
            {
                "feature": feature_name,
                "grid_values": grid_points[feature_name],
                "effect": effect_func(grid_points[feature_name]),
            }
        )

    feature_effect = FeatureEffect(effect_type=effect_type, feature_effects=fe)
    fig = plot_feature_effect(feature_effect)

    return fig


def boxplot_model_results(
    metric: Literal["mse", "mae", "r2"],
    df: pd.DataFrame,
    ylim: Optional[Tuple[float, float]] = None,
    large_font: bool = False,
) -> plt.Figure:
    set_style()
    if large_font:
        _set_fontsize("large")
    else:
        _set_fontsize("standard")
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


# def plot_effect_comparison(
#     model: BaseEstimator,
#     groundtruth: BaseEstimator,
#     X_train: np.ndarray,
#     effect: Literal["PDP", "ALE"],
#     features: List[Literal["x_1", "x_2", "x_3", "x_4", "x_5"]],
#     groundtruth_feature_effect: Literal["theoretical", "empirical"],
#     config: ConfigParser,
# ) -> plt.Figure:
#     set_style()
#     if effect == "PDP":
#         effect_func = compute_pdps
#         title = "Partial dependence"
#     elif effect == "ALE":
#         effect_func = compute_ales
#         title = "Accumulated local effects"
#     effects = effect_func(model, X_train, features, config)
#     if groundtruth_feature_effect == "theoretical":
#         grid = [effects[i]["grid_values"] for i in range(len(features))]
#         pdp_groundtruth_functions = [
#             groundtruth.get_theoretical_partial_dependence(x, feature_distribution="uniform") for x in features
#         ]
#         effects_gt = [
#             {
#                 "feature": features[i],
#                 "grid_values": grid[i],
#                 "effect": [pdp_groundtruth_functions[i](p) for p in grid[i]],
#             }
#             for i in range(len(features))
#         ]
#     elif groundtruth_feature_effect == "empirical":
#         effects_gt = effect_func(groundtruth, X_train, features, config)
#     fig, axes = plt.subplots(1, len(features), figsize=(6 * len(features), 6), dpi=300, sharey=True)
#     fig.suptitle(f"{title} comparison", fontsize=16, fontweight="bold")
#     for i in range(len(features)):  # pylint: disable=consider-using-enumerate
#         if effects[i]["feature"] != features[i]:
#             raise ValueError(f"Feature {features[i]} does not match {effects[i]['feature']}")
#         axes[i].plot(
#             effects[i]["grid_values"],
#             effects[i]["effect"],
#             label=model.__class__.__name__,
#             **get_feature_effect_plot_style(),
#         )
#         axes[i].plot(
#             effects_gt[i]["grid_values"],
#             effects_gt[i]["effect"],
#             label="Groundtruth",
#             **get_feature_effect_plot_style(),
#         )
#         axes[i].set_xlabel(f"${effects[i]['feature']}$")
#         axes[i].set_ylabel(title)
#         deciles = np.percentile(X_train[:, 0], np.arange(10, 101, 10))
#         trans = transforms.blended_transform_factory(axes[i].transData, axes[i].transAxes)
#         axes[i].vlines(deciles, 0, 0.045, transform=trans, color="k", linewidth=1)
#         axes[i].legend()

#     return fig


def _set_fontsize(size: Literal["standard", "large", "xlarge"]) -> None:
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
            _set_fontsize("xlarge")
        else:
            _set_fontsize("standard")

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


def plot_mcvariance_over_features(
    mc_variance_data: Dict[float, FeatureEffect],
    feature_names: List[str],
    title: Optional[str] = None,
    sharey: bool = False,
) -> plt.Figure:
    """
    Plot Monte Carlo variance over features for different sample sizes.

    This function creates a series of line plots, each representing the Monte Carlo
    variance of a feature effect estimate across different sample sizes. The x-axis
    represents the feature values, while the y-axis represents the Monte Carlo variance.
    Different sample sizes are represented by different colors.

    Parameters
    ----------
    mc_variance_data : Dict
        A dictionary containing the Monte Carlo variance data for different sample sizes
        and features.
    feature_names : List[str]
        A list of feature names to plot.
    title : str, optional
        The title of the plot (default is None).

    Returns
    -------
    plt.Figure
        A Figure object containing the generated plots.
    """
    set_style()
    n_feat = len(feature_names)
    fig, axes = plt.subplots(1, n_feat, figsize=(4 * n_feat, 4), sharey=sharey)
    fig.suptitle(title, y=1.02, fontsize=14, fontweight="bold")

    for feature, ax in zip(feature_names, axes):
        n_samples = list(mc_variance_data.keys())
        norm = colors.LogNorm(vmin=min(n_samples), vmax=max(n_samples))
        cmap = cm.viridis
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
        ax.set_xlabel(feature)
        ax.set_ylabel("MC Variance")

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    plt.colorbar(sm, ax=axes[-1], label="$n_{mc}$")

    return fig


def plot_mcvariance_mean(
    mc_variance_data: Dict[float, FeatureEffect],
    feature_names: List[str],
    title: Optional[str] = None,
    sharey: bool = False,
    xscale: Literal["linear", "log"] = "linear",
    yscale: Literal["linear", "log"] = "log",
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

    Returns
    -------
    plt.Figure
        A Figure object containing the generated plots.
    """
    set_style()
    n_feat = len(feature_names)
    fig, axes = plt.subplots(
        1,
        n_feat,
        figsize=(4 * n_feat, 4),
        sharey=sharey,
    )
    fig.suptitle(title, y=1.02, fontsize=14, fontweight="bold")

    for feature, ax in zip(feature_names, axes):
        means = []
        for k in mc_variance_data.keys():
            means.append(np.mean(mc_variance_data[k].features[feature]["effect"]))
        ax.plot(mc_variance_data.keys(), means, marker="+")
        ax.set_yscale(yscale)
        ax.set_xscale(xscale)
        ax.set_title(f"Mean MC Variance ${feature}$")
        ax.set_xlabel("$n_{mc}$")
        ax.set_ylabel("Mean MC Variance")

    return fig
