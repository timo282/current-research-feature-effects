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
    pivoted = df.pivot_table(
        index=["model", "n_train", "split"], columns=["feature", "metric"], values="value"
    ).reset_index()

    pivoted.columns.name = None
    df_out = pivoted.groupby(by=["model", "n_train", "split"]).mean()
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
