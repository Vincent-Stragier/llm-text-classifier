# Load models configs
# Load all prompts
# Run prompts through the correct models
# Save the predictions
# Compare to the ground truth
# Calculate the accuracy
# Save the results
import json
import pathlib

from huggingface_hub import hf_hub_download


from classifiers import Classifier
from constants import (
    DEFAULT_MODELS_CONFIGS_PATH,
    DEFAULT_PROMPTS_PATH,
)
from utils import load_all_configs


# Create a function to load all the prompts
def load_all_prompts(root_path: pathlib.Path):
    """Load all the prompts from a specified directory.

    Args:
        root_path (pathlib.Path): the path to the prompts directory.

    Returns:
        dict: a dictionary with all the prompts.
        The key of the prompts is its filename.
    """
    prompts_dict = {}

    # We could ignore some files
    glob = pathlib.Path(root_path).glob("**/*")

    ground_truth = None

    for file in glob:
        if not file.is_file():
            continue

        with file.open(mode='r', encoding='utf-8') as file_processor:
            file_content = file_processor.read()

        if file.suffix == '.json':
            ground_truth = file_content

        prompts_dict.update(
            {
                str(file): file_content
            }
        )

    return prompts_dict, ground_truth


def main():
    models_configs = load_all_configs(DEFAULT_MODELS_CONFIGS_PATH)
    prompts, ground_truth = load_all_prompts(DEFAULT_PROMPTS_PATH)

    def append_to_dict(dictionary, key, value):
        if key not in dictionary:
            dictionary[key] = [value]
        else:
            dictionary[key].append(value)

    # We need to know which prompts have not been analysed yet
    # (to do so, we also need to list all the results)
    # Then we can remove the processed prompts to reduce
    # the execution time and allow to restart the process.

    prompts_by_model = {}
    for path, prompt in prompts.items():
        model_friendly_name = path.replace('/', '\\').split('\\')[1]

        append_to_dict(prompts_by_model, model_friendly_name, {path: prompt})

    for _, model_config in models_configs.items():
        model_friendly_name = model_config.get(
            'model', {}).get('friendly_name')

        model_repo_id = model_config.get(
            'model', {}).get('hub')

        model_cache_dir = pathlib.Path(
            model_config.get('model', {}).get('local_path')
        ).parent

        model_file = pathlib.Path(
            model_config.get('model', {}).get('local_path')
        ).name

        # We should load the model path here
        # llm_model_path = hf_hub_download(
        #     model_repo_id,
        #     model_file,
        #     cache_dir=model_cache_dir,
        #     revision="main"
        # )

        for path_and_prompt in prompts_by_model.get(model_friendly_name, []):
            prompt_path, prompt = list(path_and_prompt.items())[0]

            # Put the prompt through the model


if __name__ == '__main__':
    main()
