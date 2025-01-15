import matplotlib.pyplot as plt
import pandas as pd


def set_style():
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
    style = dict(linewidth=2, marker="+", markeredgewidth=1, markersize=5)

    return style


def create_joined_melted_df(
    df_model_res: pd.DataFrame,
    df_pdp_res: pd.DataFrame,
    noise_name: str = "snr_x",
    value_vars: list[str] = ["x_1", "x_2", "x_3", "x_4", "x_5"],
) -> pd.DataFrame:
    """
    Merge two DataFrames on 'model_id' and reshape the resulting DataFrame to a long format.

    This function performs an inner join on two DataFrames based on the 'model_id' column.
    After merging, it reshapes the DataFrame to long format, transforming feature columns
    ('x_1' to 'x_5') into a single 'feature' column and their values into an 'effect_error'
    column.

    Parameters
    ----------
    df_model_res : pd.DataFrame
        DataFrame containing model results, which must include the 'model_id' column
        along with other model performance metrics such as mse_test, r2_train, etc.
    df_pdp_res : pd.DataFrame
        DataFrame containing PDP results, also must include the 'model_id' column and
        the values for each feature ('x_1' to 'x_5').
    noise_name : str, optional
        Name of the noise column in the merged DataFrame, by default "snr_x"
    value_vars : list[str], optional
        List of feature column names to be reshaped, by default ["x_1", "x_2", "x_3", "x_4", "x_5"]

    Returns
    -------
    pd.DataFrame
        A melted DataFrame containing columns from both input DataFrames along with
        'feature' and 'effect_error'.
    """
    df_merged = df_model_res.merge(df_pdp_res, on="model_id", how="inner")
    id_vars = [
        "model_id",
        "model_x",
        "simulation_x",
        "n_train_x",
        noise_name,
        "mse_train",
        "mse_test",
        "mae_train",
        "mae_test",
        "r2_train",
        "r2_test",
    ]
    df_melted = pd.melt(
        df_merged, id_vars=id_vars, value_vars=value_vars, var_name="feature", value_name="effect_error"
    )

    return df_melted
