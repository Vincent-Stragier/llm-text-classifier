# Load models configs
# Load all prompts
# Run prompts through the correct models
# Save the predictions
# Compare to the ground truth
# Calculate the accuracy
# Save the results
import pathlib

from utils import load_all_configs
#

DEFAULT_MODELS_CONFIGS_PATH = './models_configs'
DEFAULT_PROMPTS_PATH = './prompts'


# Create a function to load all the prompts
def load_all_prompts(root_path: pathlib.Path):
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

    prompts_by_model = {}
    for path, prompt in prompts.items():
        model_friendly_name = path.replace('/', '\\').split('\\')[1]

        append_to_dict(prompts_by_model, model_friendly_name, {path: prompt})

    for _, model_config in models_configs.items():
        model_friendly_name = model_config.get(
            'model', {}).get('friendly_name')

        # We should load the model path here

        for path_and_prompt in prompts_by_model.get(model_friendly_name, []):
            prompt_path, prompt = list(path_and_prompt.items())[0]

            # Put the prompt through the model


if __name__ == '__main__':
    main()
