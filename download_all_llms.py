"""This script locally download all the LLMS weights from huggingface"""
import os

import huggingface_hub

from constants import MODELS_CONFIGS_PATH
from utils import load_all_configs


def download_all_llms(
    models_configs: dict,
    configs_path: os.PathLike,
):
    """Download all the LLMS weights from huggingface.

    Args:
        models_configs (dict): the models configs.
        configs_path (os.PathLike): the path to the models configs directory.
    """
    configs_path = os.path.abspath(configs_path)

    def absolute_join(*args):
        return os.path.abspath(os.path.join(*args))

    for _, model_config in models_configs.items():
        # Download the model
        local_path = absolute_join(
            configs_path, model_config.get("model", {}).get("local_path")
        )
        local_path = os.path.dirname(local_path)

        huggingface_hub.hf_hub_download(
            repo_id=model_config.get("model", {}).get("hub"),
            filename=os.path.basename(
                model_config.get("model", {}).get("file")
            ),
            cache_dir=local_path,
        )


def main():
    """The main function of the script."""
    models_configs = load_all_configs(MODELS_CONFIGS_PATH)
    download_all_llms(models_configs, MODELS_CONFIGS_PATH)


if __name__ == "__main__":
    main()
