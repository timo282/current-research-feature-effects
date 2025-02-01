import argparse
from configparser import ConfigParser
from pathlib import Path
import warnings
import logging
import os
from copy import deepcopy
from multiprocessing import Pool, cpu_count
import numpy as np
from sqlalchemy import create_engine
from sklearn.model_selection import KFold

from current_research_feature_effects.data_generating.data_generation import generate_data
from current_research_feature_effects.model_training import initialize_model
from current_research_feature_effects.utils import (
    create_parameter_space,
    create_and_set_sim_dir,
    setup_logger,
    configure_worker_logger,
    save_fe_aggregated_results,
    save_fe_results,
    SimulationParameter,
)
from current_research_feature_effects.feature_effects import (
    compute_pdps,
    compute_ales,
    compute_cv_feature_effect,
    compute_estimator_mc_variances,
)


def simulate(
    params: SimulationParameter,
):
    np.random.seed(42)
    logging.info(f"Starting simulation with parameters: {params.groundtruth}, {params.model_name}, {params.n_train}")

    # create directories
    os.makedirs(str(params.groundtruth), exist_ok=True)
    os.makedirs(Path(str(params.groundtruth)) / "results", exist_ok=True)

    # create databases for results
    engine_effects_results = create_engine(
        f"sqlite:///{str(params.groundtruth)}{params.config.get('storage', 'effects_results')}"
    )

    feature_names = params.groundtruth.feature_names
    grid_points = params.config.getint("feature_effects", "grid_points")
    quantiles = np.linspace(0.0001, 0.9999, grid_points, endpoint=True)
    grid_values = [params.groundtruth.get_theoretical_quantiles(feature, quantiles) for feature in feature_names]
    center_curves = params.config["feature_effects"].getboolean("centered")
    remove_first_last = params.config["feature_effects"].getboolean("remove_first_last")
    k_cv = params.config.getint("simulation_metadata", "k_cv")

    pdps = {}
    ales = {}

    K = params.config.getint("simulation_metadata", "n_datasets")
    M = params.config.getint("simulation_params", "n_sim")
    for sim_no in range(M):
        logging.info(
            f"Starting simulation {sim_no+1}/{M} " + f"for {params.groundtruth} {params.model_name} {params.n_train}."
        )
        # generate data -> same as in main study due to same seed
        X_train, y_train, X_val, y_val, _, _ = generate_data(
            groundtruth=params.groundtruth,
            n_train=params.n_train,
            n_val=params.n_val,
            n_test=params.config.getint("simulation_metadata", "n_test"),
            snr=params.snr,
            seed=sim_no,
        )

        cv = KFold(n_splits=k_cv, shuffle=True, random_state=42)
        X_all, y_all = np.concatenate([X_train, X_val], axis=0), np.concatenate([y_train, y_val], axis=0)

        # initialize model -> same hyperparameters as in main study (also when tuned due to seed)
        model = initialize_model(
            params.model_config,
            params.model_name,
            params.groundtruth,
            params.n_train,
            params.snr,
            params.config,
        )

        # try to train and evaluate model -> random state ensures same model is trained as in main study
        try:
            model.fit(X_train, y_train)
        except Exception as e:
            warnings.warn(f"Training of model {params.model_name} {sim_no+1} {params.n_train} failed with error:\n{e}")

        # fit cv models -> random state ensures same models are trained as in main study
        models = []
        cv_splits = list(cv.split(X=X_all, y=y_all))
        for train_index, _ in cv_splits:
            X_train, y_train = X_all[train_index], y_all[train_index]
            model_fold = deepcopy(model)
            model_fold.fit(X_train, y_train)
            models.append(model_fold)

        # compute feature effects on K different MC datasets
        for k in range(K):
            # sample new dataset
            X_train, y_train, X_val, y_val = generate_data(
                groundtruth=params.groundtruth,
                n_train=params.n_train,
                n_test=params.n_val,
                snr=params.snr,
                seed=params.config.getint("simulation_metadata", "dataset_base_seed") + sim_no * K + k,
            )
            # calculate pdps
            pdp_train = compute_pdps(model, X_train, feature_names, grid_values, center_curves, remove_first_last)
            pdp_val = compute_pdps(model, X_val, feature_names, grid_values, center_curves, remove_first_last)

            pdp_cv = (
                sum(
                    compute_pdps(
                        model_fold, X_all[test_index], feature_names, grid_values, center_curves, remove_first_last
                    )
                    for (_, test_index), model_fold in zip(cv_splits, models)
                )
                / k_cv
            )

            # calculate ales
            ale_train = compute_ales(model, X_train, feature_names, grid_values, center_curves, remove_first_last)
            ale_val = compute_ales(model, X_val, feature_names, grid_values, center_curves, remove_first_last)
            ale_cv = (
                sum(
                    compute_ales(
                        model_fold, X_all[test_index], feature_names, grid_values, center_curves, remove_first_last
                    )
                    for (_, test_index), model_fold in zip(cv_splits, models)
                )
                / k_cv
            )

            ale_cv = compute_cv_feature_effect(
                model,
                X_all,
                y_all,
                cv,
                feature_names,
                [grid_values] * k_cv,
                compute_ales,
                center_curves,
                remove_first_last,
            )

            for split, pdp, ale in zip(
                ["train", "val", "cv"], [pdp_train, pdp_val, pdp_cv], [ale_train, ale_val, ale_cv]
            ):
                pdps[split] = [pdp] if split not in pdps else pdps[split] + [pdp]
                ales[split] = [ale] if split not in ales else ales[split] + [ale]

    # compute MC variances
    pdp_metrics = {
        split: {"MC Variance": compute_estimator_mc_variances(pdps_split, M=M, K=K)}
        for split, pdps_split in pdps.items()
    }
    ale_metrics = {
        split: {"MC Variance": compute_estimator_mc_variances(ales_split, M=M, K=K)}
        for split, ales_split in ales.items()
    }

    # save metrics
    save_fe_results(pdp_metrics, params, "pdp")
    save_fe_results(ale_metrics, params, "ale")

    # aggregate metrics
    pdp_metrics_agg = {
        split: {metric: pdp_metrics[split][metric].mean() for metric in pdp_metrics[split].keys()}
        for split in pdp_metrics.keys()
    }
    ale_metrics_agg = {
        split: {metric: ale_metrics[split][metric].mean() for metric in ale_metrics[split].keys()}
        for split in ale_metrics.keys()
    }

    # save aggregated metrics
    save_fe_aggregated_results(pdp_metrics_agg, engine_effects_results, params, "pdp")
    save_fe_aggregated_results(ale_metrics_agg, engine_effects_results, params, "ale")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to config.ini file")
    args = parser.parse_args()
    sim_config = ConfigParser()
    sim_config.read(Path(args.config))

    log_queue, log_listener = setup_logger(Path(sim_config.get("storage", "log_dir")))

    param_space = create_parameter_space(sim_config)
    logging.info(f"Created parameter space with {len(param_space)} simulation parameters.")

    create_and_set_sim_dir(sim_config, config_path=Path(args.config))

    num_processes = min(len(param_space), cpu_count())

    with Pool(processes=num_processes, initializer=configure_worker_logger, initargs=(log_queue,)) as pool:
        pool.map(
            simulate,
            param_space,
        )

    log_listener.stop()
