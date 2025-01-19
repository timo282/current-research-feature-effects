import argparse
from configparser import ConfigParser
from pathlib import Path
import warnings
import os
from typing_extensions import List, Dict, Tuple
from multiprocessing import Pool, cpu_count
from joblib import dump
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import KFold

from current_research_feature_effects.data_generating.data_generation import generate_data, Groundtruth
from current_research_feature_effects.model_training import initialize_model
from current_research_feature_effects.model_eval import eval_model
from current_research_feature_effects.utils import (
    parse_sim_params,
    create_and_set_sim_dir,
)
from current_research_feature_effects.feature_effects import (
    compute_pdps,
    compute_ales,
    compute_cv_feature_effect,
    compute_feature_effect_metrics,
)


def simulate(
    models: Dict[str, Dict],
    groundtruth: Groundtruth,
    n_sim: int,
    n_train_vals: List[Tuple[int, int]],
    snr: float,
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
    grid_values = [groundtruth.get_theoretical_quantiles(feature, quantiles) for feature in feature_names]
    center_curves = config["feature_effects"].getboolean("centered")
    remove_first_last = config["feature_effects"].getboolean("remove_first_last")

    # sample data for MC approximation of groundtruth feature effects
    X_mc, _, _, _ = generate_data(
        groundtruth=groundtruth,
        n_train=config.getint("simulation_metadata", "n_mc"),
        n_test=1,
        snr=0,
        seed=config.getint("simulation_metadata", "mc_data_seed"),
    )

    k_cv = config.getint("simulation_metadata", "k_cv")

    pdps_train = {}
    pdps_val = {}
    pdps_cv = {}

    ales_train = {}
    ales_val = {}
    ales_cv = {}

    for n_train, n_val in n_train_vals:
        for sim_no in range(n_sim):
            # generate data
            X_train, y_train, X_val, y_val, X_test, y_test = generate_data(
                groundtruth=groundtruth,
                n_train=n_train,
                n_val=n_val,
                n_test=config.getint("simulation_metadata", "n_test"),
                snr=snr,
                seed=sim_no,
            )

            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            X_all, y_all = np.concatenate([X_train, X_val], axis=0), np.concatenate([y_train, y_val], axis=0)

            pdp_groundtruth = compute_pdps(
                groundtruth,
                X_mc,
                feature_names,
                grid_values=grid_values,
                center_curves=center_curves,
                remove_first_last=remove_first_last,
            )

            ale_groundtruth = compute_ales(
                groundtruth,
                X_mc,
                feature_names,
                grid_values=grid_values,
                center_curves=center_curves,
                remove_first_last=remove_first_last,
            )

            for model_str in models:

                model_name = f"{model_str}_{sim_no+1}_{n_train}_{int(snr)}"
                model_dict = models[model_str]["model"]
                model = initialize_model(model_dict, model_str, groundtruth, n_train, snr, config)

                try:
                    model.fit(X_train, y_train)

                    # save model
                    os.makedirs(Path(str(groundtruth)) / config.get("storage", "models"), exist_ok=True)
                    dump(
                        model,
                        Path(os.getcwd()) / str(groundtruth) / config.get("storage", "models") / f"{model_name}.joblib",
                    )

                    # evaluate model
                    model_results = eval_model(model, X_train, y_train, X_test, y_test)
                except Exception as e:
                    model_results = (np.nan,) * 6
                    warnings.warn(f"Training of model {model_name} failed with error:\n{e}")

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

                # calculate pdps
                pdp_train = compute_pdps(model, X_train, feature_names, grid_values, center_curves, remove_first_last)
                pdp_val = compute_pdps(model, X_val, feature_names, grid_values, center_curves, remove_first_last)
                pdp_cv = compute_cv_feature_effect(
                    model,
                    X_all,
                    y_all,
                    cv,
                    feature_names,
                    [grid_values] * k_cv,
                    compute_pdps,
                    center_curves,
                    remove_first_last,
                )

                pdps_train[model_str] = (
                    [pdp_train] if model_str not in pdps_train else pdps_train[model_str] + [pdp_train]
                )
                pdps_val[model_str] = [pdp_val] if model_str not in pdps_val else pdps_val[model_str] + [pdp_val]
                pdps_cv[model_str] = [pdp_cv] if model_str not in pdps_cv else pdps_cv[model_str] + [pdp_cv]

                # calculate ales
                ale_train = compute_ales(model, X_train, feature_names, grid_values, center_curves, remove_first_last)
                ale_val = compute_ales(model, X_val, feature_names, grid_values, center_curves, remove_first_last)
                ale_cv = compute_cv_feature_effect(
                    model,
                    X_all,
                    y_all,
                    cv,
                    feature_names,
                    grid_values,
                    compute_ales,
                    center_curves,
                    remove_first_last,
                )

                ales_train[model_str] = (
                    [ale_train] if model_str not in ales_train else ales_train[model_str] + [ale_train]
                )
                ales_val[model_str] = [ale_val] if model_str not in ales_val else ales_val[model_str] + [ale_val]
                ales_cv[model_str] = [ale_cv] if model_str not in ales_cv else ales_cv[model_str] + [ale_cv]

        # compute metrics
        pdp_metrics_train = {
            model_str: compute_feature_effect_metrics(pdps, pdp_groundtruth) for model_str, pdps in pdps_train.items()
        }
        pdp_metrics_val = {
            model_str: compute_feature_effect_metrics(pdps, pdp_groundtruth) for model_str, pdps in pdps_val.items()
        }
        pdp_metrics_cv = {
            model_str: compute_feature_effect_metrics(pdps, pdp_groundtruth) for model_str, pdps in pdps_cv.items()
        }

        ales_metrics_train = {
            model_str: compute_feature_effect_metrics(ales, ale_groundtruth) for model_str, ales in ales_train.items()
        }
        ales_metrics_val = {
            model_str: compute_feature_effect_metrics(ales, ale_groundtruth) for model_str, ales in ales_val.items()
        }
        ales_metrics_cv = {
            model_str: compute_feature_effect_metrics(ales, ale_groundtruth) for model_str, ales in ales_cv.items()
        }

        # # compute ale feature effect metrics
        # ale_comparison = compare_effects(
        #     ale_groundtruth,
        #     ale_train,
        #     mean_squared_error,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to config.ini file")
    args = parser.parse_args()
    sim_config = ConfigParser()
    sim_config.read(Path(args.config))

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
