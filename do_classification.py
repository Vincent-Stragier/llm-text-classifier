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

    for file in glob:
        if not file.is_file():
            continue

        with file.open(mode='r', encoding='utf-8') as file_processor:
            file_content = file_processor.read()

        prompts_dict.update(
            {
                str(file): file_content
            }
        )

    return prompts_dict


def main():
    models_configs = load_all_configs(DEFAULT_MODELS_CONFIGS_PATH)
    print(models_configs)

    prompts = load_all_prompts(DEFAULT_PROMPTS_PATH)

    print()

    print(prompts)


if __name__ == '__main__':
    main()
