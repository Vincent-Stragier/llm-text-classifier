"""The utility functions of the project."""
import os
import yaml


def ensure_path_exists(path: os.PathLike):
    """Ensure the path exists"""
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def load_all_configs(
    models_configs_path: os.PathLike,
) -> dict:
    """Load all the models configs from a specified directory.

    Args:
        models_configs_path (os.PathLike): the path to the models configs directory.

    Returns:
        dict: a dictionary with all the models configs.
        The key of the models config is its filename.
    """
    # load all files as models config, file name is the key of the models config
    models_configs_files = os.listdir(models_configs_path)

    models_configs_dict = {}

    for models_config in models_configs_files:
        models_config_file_path = os.path.join(
            models_configs_path, models_config)
        if not os.path.isfile(models_config_file_path):
            continue

        # We have a file
        with open(models_config_file_path, mode='r', encoding='utf-8') as models_config_file:
            models_configs_dict[models_config] = yaml.safe_load(
                models_config_file)

    return models_configs_dict
