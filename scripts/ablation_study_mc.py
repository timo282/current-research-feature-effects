import argparse
import configparser
from typing import Dict
import yaml
import os
import shutil
from joblib import dump
from pathlib import Path
from collections import defaultdict
from multiprocessing import Pool, cpu_count
import numpy as np

from current_research_feature_effects.mappings import map_dataset_to_groundtruth
from current_research_feature_effects.data_generating.data_generation import generate_data, Groundtruth
from current_research_feature_effects.feature_effects import compute_theoretical_effects, compute_ales, compute_pdps


def perform_mc_ablation_study(groundtruth: Groundtruth, study_params: Dict):
    feature_names = groundtruth.feature_names
    grid_values = [
        groundtruth.get_theoretical_quantiles(feature, study_params["quantiles"])
        for feature in groundtruth.feature_names
    ]

    # compute theoretical effects for pdp and ale
    pdp_groundtruth_theoretical = compute_theoretical_effects(
        groundtruth,
        "pdp",
        feature_names,
        grid_values=grid_values,
        center_curves=study_params["center_curves"],
        remove_first_last=study_params["remove_first_last"],
    )
    ale_groundtruth_theoretical = compute_theoretical_effects(
        groundtruth,
        "ale",
        feature_names,
        grid_values=grid_values,
        center_curves=study_params["center_curves"],
        remove_first_last=study_params["remove_first_last"],
    )

    results = {"pdp": {}, "ale": {}}

    for n_samples in study_params["n_samples"]:
        pdp_variances = defaultdict(float)
        ale_variances = defaultdict(float)

        for i in range(study_params["k"]):
            # generate data for MC integration
            X_mc, *_ = generate_data(
                groundtruth=groundtruth, n_train=int(n_samples), n_test=1, snr=0, seed=study_params["base_seed"] + i
            )

            # estimate effects for pdp and ale
            pdp = compute_pdps(
                groundtruth,
                X_mc,
                feature_names,
                grid_values=grid_values,
                center_curves=study_params["center_curves"],
                remove_first_last=study_params["remove_first_last"],
            )
            ale = compute_ales(
                groundtruth,
                X_mc,
                feature_names,
                grid_values=grid_values,
                center_curves=study_params["center_curves"],
                remove_first_last=study_params["remove_first_last"],
            )

            # compute point-wise squared errors
            pdp_vars = {
                feature: (pdp[i]["effect"] - pdp_groundtruth_theoretical[i]["effect"]) ** 2
                for i, feature in enumerate(feature_names)
            }
            ale_vars = {
                feature: (ale[i]["effect"] - ale_groundtruth_theoretical[i]["effect"]) ** 2
                for i, feature in enumerate(feature_names)
            }

            # accumulate point-wise squared errors
            for feature in feature_names:
                pdp_variances[feature] += pdp_vars[feature]
                ale_variances[feature] += ale_vars[feature]

        # average point-wise squared errors to get the variance of the effect estimates
        results["pdp"][n_samples] = {k: v / study_params["k"] for k, v in pdp_variances.items()}
        results["ale"][n_samples] = {k: v / study_params["k"] for k, v in ale_variances.items()}

    # save results, groundtruth and grid values
    os.makedirs(EXPERIMENT_PATH / str(groundtruth), exist_ok=True)
    dump(results, EXPERIMENT_PATH / str(groundtruth) / "ablation_results.joblib")
    dump(groundtruth, EXPERIMENT_PATH / str(groundtruth) / "groundtruth.joblib")
    if study_params["remove_first_last"]:
        dump([grid[1:-1] for grid in grid_values], EXPERIMENT_PATH / str(groundtruth) / "grid_values.joblib")
    else:
        dump(grid_values, EXPERIMENT_PATH / str(groundtruth) / "grid_values.joblib")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to config.ini file")
    args = parser.parse_args()
    config = configparser.ConfigParser()
    config.read(Path(args.config))

    EXPERIMENT_PATH = Path(config["storage"]["experiments_dir"]) / config["storage"]["experiment_name"]
    os.makedirs(EXPERIMENT_PATH, exist_ok=True)
    shutil.copy2(Path(args.config), EXPERIMENT_PATH / Path(args.config).name)

    with open(Path(config["study_params"]["datasets_yaml"]), "r") as file:
        datasets_config = yaml.safe_load(file)

    groundtruths = [
        map_dataset_to_groundtruth(
            conf["groundtruth"],
            [(v["marginal"]["type"], tuple(v["marginal"]["params"])) for v in conf["features"].values()],
            np.array(conf["correlation_matrix"]),
            feature_names=list(conf["features"].keys()),
            name=name,
        )
        for name, conf in datasets_config.items()
    ]
    groundtruths = [groundtruths[int(i)] for i in config.get("study_params", "use_datasets").split(",")]

    study_params = {
        "n_samples": np.logspace(
            int(config.get("study_params", "n_samples").split(",")[0]),
            int(config.get("study_params", "n_samples").split(",")[1]),
            num=int(config.get("study_params", "n_samples").split(",")[2]),
        ),
        "k": config.getint("study_params", "k"),
        "n_grid_points": config.getint("study_params", "n_grid_points"),
        "base_seed": config.getint("study_params", "base_seed"),
        "quantiles": np.linspace(
            float(config.get("study_params", "quantiles").split(",")[0]),
            float(config.get("study_params", "quantiles").split(",")[1]),
            config.getint("study_params", "n_grid_points"),
            endpoint=True,
        ),
        "center_curves": config.getboolean("study_params", "center_curves"),
        "remove_first_last": config.getboolean("study_params", "remove_first_last"),
    }

    num_processes = min(len(groundtruths), cpu_count())

    # perform ablation studies in parallel
    with Pool(processes=num_processes) as pool:
        pool.starmap(
            perform_mc_ablation_study,
            [(gt, study_params) for gt in groundtruths],
        )
