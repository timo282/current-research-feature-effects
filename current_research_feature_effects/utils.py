from configparser import ConfigParser
from typing import Dict
from pathlib import Path
import os
import shutil
import yaml
import numpy as np

from current_research_feature_effects.mappings import map_dataset_to_groundtruth, map_modelname_to_estimator


def parse_sim_params(sim_config: ConfigParser) -> Dict:
    """Parse simulation parameters from configuration file.

    Parameters
    ----------
    sim_config : ConfigParser
        Config containing simulation parameters.

    Returns
    -------
    Dict
        Dictionary of simulation parameters.
    """
    param_dict = {}
    models_config_path = Path(sim_config["simulation_params"]["models_yaml"])
    datasets_config_path = Path(sim_config["simulation_params"]["datasets_yaml"])

    param_dict["n_sim"] = sim_config.getint("simulation_params", "n_sim")
    param_dict["n_train_val"] = [
        (int(n_train), int(n_val))
        for n_train, n_val in zip(
            sim_config.get("simulation_params", "n_train").split(","),
            sim_config.get("simulation_params", "n_val").split(","),
        )
    ]
    param_dict["snr"] = [float(x) for x in sim_config.get("simulation_params", "snr").split(",")]

    with open(models_config_path, "r") as file:
        models_config: Dict = yaml.safe_load(file)

    for model in models_config.keys():
        models_config[model]["model"] = map_modelname_to_estimator(models_config[model]["model"])
    param_dict["models_config"] = models_config

    with open(datasets_config_path, "r") as file:
        datasets_config: Dict = yaml.safe_load(file)

    param_dict["groundtruths"] = [
        map_dataset_to_groundtruth(
            config["groundtruth"],
            [(v["marginal"]["type"], tuple(v["marginal"]["params"])) for v in config["features"].values()],
            np.array(config["correlation_matrix"], dtype=float),
            feature_names=list(config["features"].keys()),
            name=name,
        )
        for name, config in datasets_config.items()
    ]

    return param_dict


def create_and_set_sim_dir(sim_config: ConfigParser) -> None:
    """Create and set simulation directory.

    Parameters
    ----------
    sim_config : ConfigParser
        Config containing simulation name and base directory.
    """
    simulation_name = sim_config.get("storage", "simulation_name")

    base_dir = Path(sim_config.get("storage", "simulations_dir")) / simulation_name
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
        shutil.copy2("config.ini", base_dir / f"config_{simulation_name}.ini")
        os.chdir(base_dir)
    else:
        raise ValueError(f"Simulation {base_dir} already exists.")
