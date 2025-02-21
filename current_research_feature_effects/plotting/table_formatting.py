"""
This module contains functions to format dataframes for better readibility and
to simplify plotting.
"""

from pathlib import Path
from configparser import ConfigParser
from typing import Literal
import pandas as pd
import numpy as np


def format_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Format the dataframe to a more readable format for the table.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the results of the model training.

    Returns
    -------
    pd.DataFrame
        Formatted dataframe.
    """
    df_out = get_grouped_df(df)
    sorted_cols = [
        (feature, metric)
        for feature in sorted(set(col[0] for col in df_out.columns))
        for metric in ["MSE", "Bias^2", "Variance"]
    ]
    sorted_idx = [
        (model, n, split)
        for model in ["LinReg", "GAM_OF", "GAM_OT", "SVM_OF", "SVM_OT", "XGBoost_OF", "XGBoost_OT"]
        for n in sorted(df_out.index.get_level_values("n_train").unique())
        for split in ["train", "val", "cv"]
    ]
    df_out = df_out[sorted_cols].reindex(pd.MultiIndex.from_tuples(sorted_idx))

    return df_out


def highlight_min_feature_metric(data: pd.DataFrame) -> pd.DataFrame:
    """
    Highlight the minimum value in each feature-metric pair.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe containing the results of the feature effect analysis.

    Returns
    -------
    pd.DataFrame
        Dataframe with the minimum values highlighted.
    """
    mask = pd.DataFrame(False, index=data.index, columns=data.columns)
    groups = data.index.droplevel(2).unique()

    for model_n_train in groups:
        group_mask = (data.index.get_level_values(0) == model_n_train[0]) & (
            data.index.get_level_values(1) == model_n_train[1]
        )

        for feature in data.columns.get_level_values(0).unique():
            for metric in data.columns.get_level_values(1).unique():
                col = (feature, metric)
                is_min = data[col][group_mask] == data[col][group_mask].min()
                mask.loc[group_mask, col] = is_min

    return pd.DataFrame(np.where(mask, "font-weight: bold", ""), index=data.index, columns=data.columns)


def get_concatenated_results(
    dataset: str,
    effect: Literal["pdp", "ale"],
    config: ConfigParser,
    main_config: ConfigParser,
    main_experiment_path: Path,
    experiment_path: Path,
):
    """
    Get the concatenated results of the main and ablation experiment
    for the analysis of variance decomposition.

    Parameters
    ----------
    dataset : str
        Name of the dataset.
    effect : str
        Name of the effect.
    config : configparser.ConfigParser
        Configuration of the ablation experiment.
    main_config : configparser.ConfigParser
        Configuration of the main experiment.
    main_experiment_path : pathlib.Path
        Path to the main experiment results.
    experiment_path : pathlib.Path
        Path to the ablation experiment results.

    Returns
    -------
    pd.DataFrame
        Concatenated results of the main and ablation experiment.
    """
    df_main = pd.read_sql_table(
        f"{effect}_results",
        f"sqlite:///{main_experiment_path}/{dataset}/{main_config.get('storage', 'effects_results')}",
    )
    df_main_filtered = df_main.loc[(df_main["model"] == "XGBoost_OT") | (df_main["model"] == "XGBoost_OF")]
    df_ablation = pd.read_sql_table(
        f"{effect}_results", f"sqlite:///{experiment_path}/{dataset}/{config.get('storage', 'effects_results')}"
    )
    df_concat = pd.concat([df_main_filtered, df_ablation], ignore_index=True)

    return df_concat


def get_grouped_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Group the dataframe by the number of training samples, model, and split.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the results of the model training.

    Returns
    -------
    pd.DataFrame
        Grouped dataframe
    """
    pivoted = df.pivot_table(
        index=["n_train", "model", "split"], columns=["feature", "metric"], values="value"
    ).reset_index()

    pivoted.columns.name = None

    return pivoted.groupby(by=["n_train", "model", "split"]).mean()


def filter_variance_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter the dataframe to only contain the variance metrics
    and calculate the model variance.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the results of the model training.

    Returns
    -------
    pd.DataFrame
        Dataframe containing only the variance metrics.
    """
    features = df.columns.get_level_values(0).unique()
    result_dict = {}

    for feature in features:
        mc_var = df[(feature, "MC Variance")]
        var = df[(feature, "Variance")]

        # Calculate Model Variance
        model_var = np.where(mc_var.notna(), var - mc_var, np.nan)

        result_dict[(feature, "Variance")] = var
        result_dict[(feature, "Model Variance")] = model_var
        result_dict[(feature, "MC Variance")] = mc_var

    return pd.DataFrame(result_dict, index=df.index)


def flatten_variance_df(hierarchical_df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten the hierarchical dataframe to a flat dataframe.

    Parameters
    ----------
    hierarchical_df : pd.DataFrame
        Hierarchical dataframe containing the results of the model training.

    Returns
    -------
    pd.DataFrame
        Flattened dataframe.
    """
    records = []
    for idx, row in hierarchical_df.iterrows():
        n_train, model, split = idx
        for col in hierarchical_df.columns:
            feature, metric = col
            value = row[col]
            record = {
                "model": model,
                "split": split,
                "n_train": n_train,
                "feature": feature,
                "metric": metric,
                "value": value,
            }
            records.append(record)
    flat_df = pd.DataFrame(records)
    flat_df = flat_df.reset_index(drop=True)
    split_order = ["train", "val", "cv"]
    flat_df["split"] = pd.Categorical(flat_df["split"], categories=split_order, ordered=True)
    flat_df = flat_df.sort_values(by=["n_train", "model", "split", "feature", "metric"])

    return flat_df
