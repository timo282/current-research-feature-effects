from configparser import ConfigParser
import math
from typing_extensions import Literal, List, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import transforms
import seaborn as sns
from sklearn.base import BaseEstimator
from scipy.stats import pearsonr, spearmanr

from feature_effect_empirical_analysis.plotting.utils import (
    set_style,
    get_boxplot_style,
    get_feature_effect_plot_style,
)
from feature_effect_empirical_analysis.feature_effects import compute_pdps, compute_ales


def boxplot_model_results(metric: Literal["mse", "mae", "r2"], df: pd.DataFrame) -> plt.Figure:
    set_style()
    fig, ax = plt.subplots(1, 2, figsize=(12, 6), dpi=300, sharey=True)
    fig.suptitle("Model evaluation", fontsize=16, fontweight="bold")
    ax[0].set_title(f"{metric} on train set")
    sns.boxplot(
        x="snr",
        y=f"{metric}_train",
        hue="model",
        data=df,
        ax=ax[0],
        **get_boxplot_style(),
    )
    ax[0].legend().set_visible(False)
    sns.boxplot(
        x="snr",
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


def boxplot_feature_effect_results(
    features: List[Literal["x_1", "x_2", "x_3", "x_4", "x_5"]],
    df: pd.DataFrame,
    effect_type: Literal["PDP", "ALE"],
) -> plt.Figure:
    set_style()
    fig = plt.figure(
        figsize=(min(6 * len(features), 18), math.ceil(len(features) / 3) * 5),
        dpi=300,
    )
    fig.suptitle(
        f"Feature effect evaluation {effect_type} with {df['metric'].iloc[0]}",
        fontsize=16,
        fontweight="bold",
    )
    for i, feature in enumerate(features):
        if i == 0:
            plt.subplot(
                math.ceil(len(features) / 3),
                min(len(features), 3),
                i + 1,
            )
        else:
            ax = plt.gca()
            plt.subplot(
                math.ceil(len(features) / 3),
                min(len(features), 3),
                i + 1,
                sharey=ax,
            )
        plt.title(f"{effect_type} of {feature}")
        sns.boxplot(
            x="snr",
            y=feature,
            hue="model",
            data=df,
            **get_boxplot_style(),
        )
        plt.legend().set_visible(False)
    plt.legend(title="Learner", bbox_to_anchor=(1.05, 1), loc="upper left").set_visible(True)
    fig.tight_layout()

    return fig


def plot_effect_comparison(
    model: BaseEstimator,
    groundtruth: BaseEstimator,
    X_train: np.ndarray,
    effect: Literal["PDP", "ALE"],
    features: List[Literal["x_1", "x_2", "x_3", "x_4", "x_5"]],
    groundtruth_feature_effect: Literal["theoretical", "empirical"],
    config: ConfigParser,
) -> plt.Figure:
    set_style()
    if effect == "PDP":
        effect_func = compute_pdps
        title = "Partial dependence"
    elif effect == "ALE":
        effect_func = compute_ales
        title = "Accumulated local effects"
    effects = effect_func(model, X_train, features, config)
    if groundtruth_feature_effect == "theoretical":
        grid = [effects[i]["grid_values"] for i in range(len(features))]
        pdp_groundtruth_functions = [
            groundtruth.get_theoretical_partial_dependence(x, feature_distribution="uniform") for x in features
        ]
        effects_gt = [
            {
                "feature": features[i],
                "grid_values": grid[i],
                "effect": [pdp_groundtruth_functions[i](p) for p in grid[i]],
            }
            for i in range(len(features))
        ]
    elif groundtruth_feature_effect == "empirical":
        effects_gt = effect_func(groundtruth, X_train, features, config)
    fig, axes = plt.subplots(1, len(features), figsize=(6 * len(features), 6), dpi=300, sharey=True)
    fig.suptitle(f"{title} comparison", fontsize=16, fontweight="bold")
    for i in range(len(features)):  # pylint: disable=consider-using-enumerate
        if effects[i]["feature"] != features[i]:
            raise ValueError(f"Feature {features[i]} does not match {effects[i]['feature']}")
        axes[i].plot(
            effects[i]["grid_values"],
            effects[i]["effect"],
            label=model.__class__.__name__,
            **get_feature_effect_plot_style(),
        )
        axes[i].plot(
            effects_gt[i]["grid_values"],
            effects_gt[i]["effect"],
            label="Groundtruth",
            **get_feature_effect_plot_style(),
        )
        axes[i].set_xlabel(f"${effects[i]['feature']}$")
        axes[i].set_ylabel(title)
        deciles = np.percentile(X_train[:, 0], np.arange(10, 101, 10))
        trans = transforms.blended_transform_factory(axes[i].transData, axes[i].transAxes)
        axes[i].vlines(deciles, 0, 0.045, transform=trans, color="k", linewidth=1)
        axes[i].legend()

    return fig


def plot_correlation_analysis(
    df_melted: pd.DataFrame,
    models: List[str],
    feature_effect: str,
    model_error_metric: str = "mse_test",
    correlation_metric: Literal["Pearson", "Spearman"] = "Pearson",
    overall_correlation: bool = False,
    return_correlation_table: bool = False,
    noise_name: str = "snr_x",
) -> sns.axisgrid.FacetGrid | Tuple[sns.axisgrid.FacetGrid, pd.DataFrame]:
    """
    Plot correlation analysis between model error and a specified feature effect
    error across different signal-to-noise ratios and models.

    This function creates a series of scatter plots using Seaborn's FacetGrid,
    each representing a correlation analysis between model error metrics and
    feature effect errors. It can compute either Pearson or Spearman correlation
    depending on the specified correlation metric. Annotations indicating the
    correlation coefficients are added to each plot.

    Parameters
    ----------
    df_melted : pd.DataFrame
        A melted DataFrame (via plotting.utils.create_joined_melted_df) with each
        row representing a specific observation and containing columns for model id,
        feature error, and other variables.
    models : List[str]
        A list of model names to be used for hue in plots, should match values in
        'model_x' column of `df_melted`.
    feature_effect : str
        The name of the feature effect to plot.
    model_error_metric : str, optional
        The name of the model error metric to plot (default is "mse_test"),
        used to set xlabel in the plots.
    correlation_metric : {'Pearson', 'Spearman'}, optional
        The type of correlation to compute, either 'Pearson' or 'Spearman' (default is 'Pearson').
    overall_correlation : bool, optional
        If True, compute and annotate overall correlation per subplot. If False, only
        annotate correlations per model (default is False).
    return_correlation_table : bool, optional
        If True, return a DataFrame containing the correlation results (default is False).
    noise_name : str, optional
        Name of the noise column in the merged DataFrame, by default "snr_x".

    Returns
    -------
    seaborn.axisgrid.FacetGrid
        A FacetGrid object containing the generated plots.
    """
    set_style()

    def corr(x, y):
        if correlation_metric == "Pearson":
            return pearsonr(x, y)[0]
        elif correlation_metric == "Spearman":
            return spearmanr(x, y)[0]

    snrs = df_melted[noise_name].unique()

    g = sns.FacetGrid(
        df_melted,
        col=noise_name,
        row="feature",
        hue="model_x",
        palette="Set2",
        col_order=sorted(snrs),
        hue_order=models,
        aspect=1.5,
        height=4,
    )
    g.map(sns.scatterplot, model_error_metric, "effect_error")
    g.figure.suptitle(
        f"Correlation Analysis ({correlation_metric}): {feature_effect} Error vs. Model Error",
        fontsize=20,
        fontweight="bold",
        y=1.025,
    )
    g.figure.set_dpi(300)

    correlation_results = []

    for ax, ((feature, snr), sub_df) in zip(g.axes.flatten(), df_melted.groupby(["feature", noise_name])):
        if overall_correlation:
            overall_corr = (
                corr(sub_df[model_error_metric], sub_df["effect_error"])
                if len(sub_df["effect_error"]) > 1
                else float("nan")
            )
            ax.text(
                0.5,
                0.9,
                f"Overall Correlation: {overall_corr:.2f}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=9,
            )
            correlation_results.append(
                {"snr": snr, "feature": feature, "model": "Overall", "correlation": overall_corr}
            )
        for model in models:
            model_data = sub_df[sub_df["model_x"] == model]
            if not model_data.empty:
                model_error = model_data[model_error_metric]
                effect_error = model_data["effect_error"]
                model_corr = corr(model_error, effect_error) if len(model_error) > 1 else float("nan")
                ax.text(
                    model_error.iloc[-1],
                    effect_error.iloc[-1],
                    f"{model_corr:.2f}",
                    color=sns.color_palette("Set2", n_colors=len(models)).as_hex()[models.index(model)],
                    fontsize=9,
                )
                correlation_results.append(
                    {"snr": snr, "feature": feature, "model": model, "correlation": model_corr}
                )

    g.set_titles(col_template=noise_name+": {col_name}", row_template="${row_name}$", fontweight=16)
    g.set_axis_labels(f"Model Error ({model_error_metric})", f"{feature_effect} Error")
    g.add_legend(title="Estimator")

    if return_correlation_table:
        return g, pd.DataFrame(correlation_results)

    return g
