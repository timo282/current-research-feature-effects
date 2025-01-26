from configparser import ConfigParser
import logging
from typing import Dict, List, Literal
from pathlib import Path
from dataclasses import dataclass
from itertools import product
from joblib import dump
import os
import shutil
import yaml
import numpy as np
import pandas as pd
from sqlalchemy import Engine

from current_research_feature_effects.mappings import map_dataset_to_groundtruth, map_modelname_to_estimator
from current_research_feature_effects.data_generating.data_generation import Groundtruth


@dataclass
class SimulationParameter:
    """Parameter combination for simulation"""

    groundtruth: Groundtruth
    model_name: str
    model_config: Dict
    n_train: int
    n_val: int
    snr: float
    config: ConfigParser


def _parse_sim_params(sim_config: ConfigParser) -> Dict:
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
    param_dict["snr"] = sim_config.getfloat("simulation_params", "snr")

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


def create_parameter_space(config: ConfigParser) -> List[SimulationParameter]:
    """
    Create parameter space for simulation.

    Parameters
    ----------
    config : ConfigParser
        Configuration file containing simulation parameters.

    Returns
    -------
    List[SimulationParameter]
        List of simulation parameter
    """
    sim_params = _parse_sim_params(config)

    return [
        SimulationParameter(
            groundtruth=gt,
            model_name=model_name,
            model_config=model_config,
            n_train=n_train,
            n_val=n_val,
            snr=sim_params["snr"],
            config=config,
        )
        for gt, (n_train, n_val), (model_name, model_config) in product(
            sim_params["groundtruths"], sim_params["n_train_val"], sim_params["models_config"].items()
        )
    ]


def create_and_set_sim_dir(sim_config: ConfigParser, config_path: Path) -> None:
    """Create and set simulation directory.

    Parameters
    ----------
    sim_config : ConfigParser
        Config containing simulation name and base directory.
    config_path : Path
        Path to the configuration file.
    """
    simulation_name = sim_config.get("storage", "simulation_name")

    base_dir = Path(sim_config.get("storage", "simulations_dir")) / simulation_name
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
        shutil.copy2(config_path, base_dir / config_path.name)
        os.chdir(base_dir)
        logging.info(f"Simulation directory {base_dir} created.")
    else:
        raise ValueError(f"Simulation {base_dir} already exists.")


def save_model_results(model_metrics: Dict[str, float], conn: Engine, params: SimulationParameter, sim_no: int):
    """
    Save model results to database.

    Parameters
    ----------
    model_metrics : Dict[str, float]
        Dictionary of model metrics.
    conn : Engine
        SQLAlchemy engine for results database.
    params : SimulationParameter
        Simulation parameters.
    sim_no : int
        Simulation number.
    """
    df_model_result = pd.DataFrame(
        {
            "model": [params.model_name],
            "simulation": [sim_no + 1],
            "n_train": [params.n_train],
            "snr": [params.snr],
        }
        | model_metrics
    )

    df_model_result.to_sql(
        "model_results",
        con=conn,
        if_exists="append",
    )

    logging.info(f"Saved model results for {params.model_name} {params.n_train} simulation {sim_no + 1}.")


def save_fe_aggregated_results(
    res_agg: Dict[str, Dict[str, Dict]],
    conn: Engine,
    params: SimulationParameter,
    type: Literal["pdp", "ale"],
):
    """
    Save aggregated feature effect results to database.

    Parameters
    ----------
    res_agg : Dict[str, Dict[str, Dict]]
        Aggregated feature effect results.
    conn : Engine
        SQLAlchemy engine for results database.
    params : SimulationParameter
        Simulation parameters.
    type : Literal[&quot;pdp&quot;, &quot;ale&quot;]
        Type of feature effect.
    """
    rows = []
    for split, split_results in res_agg.items():
        for metric, feature_values in split_results.items():
            for feature, value in feature_values.items():
                rows.append(
                    {
                        "model": params.model_name,
                        "n_train": params.n_train,
                        "split": split,
                        "metric": metric,
                        "feature": feature,
                        "value": value,
                    }
                )

    df_result = pd.DataFrame(rows)

    df_result.to_sql(
        f"{type}_results",
        con=conn,
        if_exists="append",
    )

    logging.info(f"Saved aggregated {type} results for {params.model_name} {params.n_train}.")


def save_fe_results(fe_metrics: Dict, params: SimulationParameter, type: Literal["pdp", "ale"]):
    """
    Save feature effect results to joblib file.

    Parameters
    ----------
    fe_metrics : Dict
        Feature effect metrics.
    params : SimulationParameter
        Simulation parameters.
    type : Literal[&quot;pdp&quot;, &quot;ale&quot;]
        Type of feature effect.
    """
    os.mkdir(Path(str(params.groundtruth)) / "results" / params.model_name, exist_ok=True)
    dump(
        fe_metrics,
        Path(str(params.groundtruth)) / "results" / params.model_name / f"{type}_metrics_{params.n_train}.joblib",
    )

    logging.info(f"Saved {type} results for {params.model_name} {params.n_train}.")
