from configparser import ConfigParser
from pathlib import Path
import os
from typing_extensions import List, Dict, Tuple
from multiprocessing import Pool, cpu_count
from joblib import dump
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_squared_error
import optuna

from current_research_feature_effects.data_generating.data_generation import generate_data, Groundtruth
from current_research_feature_effects.model_training import optimize
from current_research_feature_effects.model_eval import eval_model
from current_research_feature_effects.utils import (
    parse_sim_params,
    create_and_set_sim_dir,
)
from current_research_feature_effects.feature_effects import (
    compute_pdps,
    compute_ales,
    compare_effects,
    get_modified_grids,
)

sim_config = ConfigParser()
sim_config.read("config.ini")


def simulate(
    models: Dict[str, Dict],
    groundtruth: Groundtruth,
    n_sim: int,
    n_train_vals: List[Tuple[int, int]],
    snrs: List[float],
    config: ConfigParser,
):
    np.random.seed(42)

    # create dir for dataset
    os.mkdir(str(groundtruth))

    # create databases for results
    engine_model_results = create_engine(f"sqlite:///{str(groundtruth)}{config.get('storage', 'model_results')}")
    engine_effects_results = create_engine(f"sqlite:///{str(groundtruth)}{config.get('storage', 'effects_results')}")

    # calulate feature effects of groundtruth
    feature_names = groundtruth.feature_names
    grid_points = config.getint("feature_effects", "grid_points")
    quantiles = np.linspace(0, 1, grid_points, endpoint=True)
    base_grids = [groundtruth.get_theoretical_quantiles(feature, quantiles) for feature in feature_names]
    center_curves = config["feature_effects"].getboolean("centered")
    remove_first_last = config["feature_effects"].getboolean("remove_first_last")

    # sample data for MC approximation of groundtruth feature effects
    X_mc, y_mc, _, _ = generate_data(
        groundtruth=groundtruth,
        n_train=config.getint("simulation_metadata", "n_mc"),
        n_test=1,
        snr=0,
        seed=config.getint("simulation_metadata", "mc_data_seed"),
    )

    for sim_no in range(n_sim):
        for n_train, n_val in n_train_vals:
            for snr in snrs:
                # generate data
                X_train, y_train, X_val, y_val, X_test, y_test = generate_data(
                    groundtruth=groundtruth,
                    n_train=n_train,
                    n_val=n_val,
                    n_test=config.getint("simulation_metadata", "n_test"),
                    snr=snr,
                    seed=sim_no,
                )

                train_grid, val_grid, mc_grid = get_modified_grids(
                    base_grids=base_grids, Xs=[X_train, X_val, X_mc], feature_names=feature_names
                )

                pdp_groundtruth = compute_pdps(
                    groundtruth,
                    X_mc,
                    feature_names,
                    grid_values=mc_grid,
                    center_curves=center_curves,
                    remove_first_last=remove_first_last,
                )

                ale_groundtruth = compute_ales(
                    groundtruth,
                    X_mc,
                    feature_names,
                    grid_values=mc_grid,
                    center_curves=center_curves,
                    remove_first_last=remove_first_last,
                )

                for model_str in models:
                    model_name = f"{model_str}_{sim_no+1}_{n_train}_{int(snr)}"
                    model: BaseEstimator = models[model_str]["model"]

                    if models[model_str]["model_params"] == "to_tune":
                        tuning_studies_dir = config.get("storage", "tuning_studies_folder")
                        os.makedirs(Path(os.getcwd()) / tuning_studies_dir, exist_ok=True)
                        study_name = f"{model_str}_{n_train}_{int(snr)}"
                        storage_name = f"sqlite:///{str(groundtruth)}/{tuning_studies_dir}/{model_str}.db"
                        try:
                            model_params = optuna.load_study(
                                study_name=study_name,
                                storage=storage_name,
                            ).best_params
                        except KeyError as e:
                            if str(e) == "Record does not exist.":
                                X_tuning_train, y_tuning_train, X_tuning_val, y_tuning_val = generate_data(
                                    groundtruth=groundtruth,
                                    n_train=n_train,
                                    n_test=config.get("simulation_metadata", "n_tuning_val"),
                                    snr=snr,
                                    seed=config.get("simulation_metadata", "tuning_data_seed"),
                                )
                                model_params = optimize(
                                    model=model,
                                    X_train=X_tuning_train,
                                    y_train=y_tuning_train,
                                    X_val=X_tuning_val,
                                    y_val=y_tuning_val,
                                    n_trials=config.get("simulation_metadata", "n_tuning_trials"),
                                    metric=config.get("simulation_metadata", "tuning_metric"),
                                    direction=config.get("simulation_metadata", "tuning_direction"),
                                    study_name=study_name,
                                    storage_name=storage_name,
                                ).best_params
                            else:
                                raise
                    else:
                        model_params = models[model_str]["model_params"]

                    model.set_params(**model_params)

                    model.fit(X_train, y_train)

                    # save model
                    os.makedirs(Path(str(groundtruth)) / config.get("storage", "models"), exist_ok=True)
                    dump(
                        model,
                        Path(os.getcwd()) / str(groundtruth) / config.get("storage", "models") / f"{model_name}.joblib",
                    )

                    # evaluate model
                    model_results = eval_model(model, X_train, y_train, X_test, y_test)
                    df_model_result = pd.DataFrame(
                        {
                            "model_id": [model_name],
                            "model": [model_str],
                            "simulation": [sim_no + 1],
                            "n_train": [n_train],
                            "snr": [snr],
                            "mse_train": [model_results[0]],
                            "mse_test": [model_results[1]],
                            "mae_train": [model_results[2]],
                            "mae_test": [model_results[3]],
                            "r2_train": [model_results[4]],
                            "r2_test": [model_results[5]],
                        }
                    )

                    # save model results
                    os.makedirs(Path(str(groundtruth)) / "results", exist_ok=True)
                    df_model_result.to_sql(
                        "model_results",
                        con=engine_model_results,
                        if_exists="append",
                    )

                    # # calculate ales and compare to groundtruth
                    # ale = compute_ales(model, X_train, feature_names, config)
                    # ale_comparison = compare_effects(
                    #     ale_groundtruth,
                    #     ale,
                    #     mean_squared_error,
                    #     center_curves=config["errors"].getboolean("centered"),
                    # )
                    # df_ale_result = pd.concat(
                    #     (
                    #         pd.DataFrame(
                    #             {
                    #                 "model_id": [model_name],
                    #                 "model": [model_str],
                    #                 "simulation": [sim_no + 1],
                    #                 "n_train": [n_train],
                    #                 "snr": [snr],
                    #             }
                    #         ),
                    #         ale_comparison,
                    #     ),
                    #     axis=1,
                    # )

                    # # save ale results
                    # df_ale_result.to_sql(
                    #     "ale_results",
                    #     con=engine_effects_results,
                    #     if_exists="append",
                    # )

                    # # calculate and compare pdps to groundtruth
                    # pdp = compute_pdps(model, X_train, feature_names, config)
                    # pdp_comparison = compare_effects(
                    #     pdp_groundtruth,
                    #     pdp,
                    #     mean_squared_error,
                    #     center_curves=config["errors"].getboolean("centered"),
                    # )
                    # df_pdp_result = pd.concat(
                    #     (
                    #         pd.DataFrame(
                    #             {
                    #                 "model_id": [model_name],
                    #                 "model": [model_str],
                    #                 "simulation": [sim_no + 1],
                    #                 "n_train": [n_train],
                    #                 "snr": [snr],
                    #             }
                    #         ),
                    #         pdp_comparison,
                    #     ),
                    #     axis=1,
                    # )

                    # # save pdp results
                    # df_pdp_result.to_sql(
                    #     "pdp_results",
                    #     con=engine_effects_results,
                    #     if_exists="append",
                    # )


if __name__ == "__main__":
    sim_params = parse_sim_params(sim_config)
    create_and_set_sim_dir(sim_config)
    groundtruths = sim_params["groundtruths"]
    num_processes = min(len(groundtruths), cpu_count())

    # Create a pool of processes and map groundtruths to the processing function
    with Pool(processes=num_processes) as pool:
        pool.starmap(
            simulate,
            [
                (
                    sim_params["models_config"],
                    gt,
                    sim_params["n_sim"],
                    sim_params["n_train_val"],
                    sim_params["snr"],
                    sim_config,
                )
                for gt in groundtruths
            ],
        )
