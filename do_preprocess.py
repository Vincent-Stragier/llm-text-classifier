"""
Do classifications with Mixtral Instruct, Mistral Instruct, and all the Llama 2 chat models.
"""
import json
import os

from copy import deepcopy
from typing import Iterable, List

import numpy as np
import jinja2

from utils import load_all_configs


def load_dataset(path) -> list:
    """Load a dataset.

    Args:
        path (str | Path): the path for the dataset1

    Returns:
        list: the list of the dataset elements
    """
    with open(path, mode='rb') as json_file:
        return json.load(json_file)


def split_dataset(dataset: Iterable, split: Iterable = (0.33, 0.33, 0.33), random_state: int = 53):
    """Split a dataset into three parts.

    Args:
        dataset (Iterable): the dataset to split
        split (Iterable): the split ratios
        random_state (int): the random state

    Returns:
        tuple: the three splits
    """
    assert len(split) == 3

    np.random.seed(random_state)
    np.random.shuffle(dataset)  # what about equipotent classes?

    # normalize the split
    normalized_split = np.array(split) / np.sum(split)

    assert len(normalized_split) == 3
    assert np.sum(normalized_split) == 1

    splitted_dataset = deepcopy(dataset)
    for index, tool in enumerate(dataset):
        # Compute the split
        subdataset_length = len(tool['dataset'])
        adapted_split = np.array(
            normalized_split * subdataset_length, dtype=int).cumsum()

        # Shuffle the dataset
        subdataset = deepcopy(tool['dataset'])
        np.random.shuffle(subdataset)

        # Split the dataset
        assert len(adapted_split) >= 3
        train, test, validation = np.split(subdataset, adapted_split)[:3]
        splitted_dataset[index]['dataset'] = {
            'train': train, 'test': test, 'validation': validation}

    return splitted_dataset


# def generate_prompts(tools: str):
#     pass

# DoE
# models, prompts, datasets
# Features, classification performances, speed (min, max, median, average)
# extract probabilities, etc.
# https://llama-cpp-python.readthedocs.io/en/latest/api-reference/#llama_cpp.llama_cpp.llama_sample_apply_guidance


# Models
# Don't do that, use a models_configs/model_name_config_name.yaml file
models = (
    # "https://huggingface.co/TheBloke/Llama-2-7b-Chat-GGUF/blob/main/llama-2-7b-chat.Q5_K_M.gguf",

)
# Do dataset splits (evaluation, training, and testing)
# Do classifications
# Save the results
# Plot the results


def load_template(
    template_path: os.PathLike,
) -> dict:
    """Load all the templates from a specified directory.

    Args:
        template_path (os.PathLike): the path to the templates directory.

    Returns:
        dict: a dictionary with all the templates.
        The key of the template is its filename.
    """
    # load all files as template, file name is the key of the template
    template_files = os.listdir(template_path)

    template_dict = {}

    for template in template_files:
        template_file_path = os.path.join(template_path, template)
        if not os.path.isfile(template_file_path):
            continue

        # We have a file
        with open(template_file_path, mode='r', encoding='utf-8') as template_file:
            template_dict[template] = jinja2.Template(template_file.read())

    return template_dict


def render_template(
    template: jinja2.Template,
    content: dict,
):
    """Render a template with a given content.

    Args:
        template (jinja2.Template): the template to render
        content (dict): the content to render the template with

    Returns:
        str: the rendered template
    """
    return template.render(
        **content
    )


def make_prompts(template, model, datasets, random_seed: int = 5876) -> List[str]:
    for dataset_name, dataset in datasets.items():
        tools = sorted(list({tool.get('tool_name', None) for tool in dataset}))

        for tool in dataset:
            test_set = tool.get('dataset', {}).get('test', None)
            train_set = deepcopy(tool.get('dataset', {}).get('train', None))

            if test_set is None:
                raise ValueError(
                    f"Test set is missing for tool {tool['tool_name']}")

            if train_set is None:
                raise ValueError(
                    f"Train set is missing for tool {tool['tool_name']}")

            # randomly select an element from the train set using a specific seed
            np.random.seed(random_seed)
            selected_element = np.random.choice(train_set)

            for element in test_set:
                # user_request
                print(element.get('user_request', None),
                      element.get('command', None))

        """Make a prompt."""
        # I want to build example from the dataset via train set
        # And use every element of the dev set
        # list all the class, list all the

    return []


def main():
    """The main function of the script."""
    # Dataset generated via Belunga using a zero-shot approach
    dataset_a = load_dataset("datasets/dataset_a.json")
    # Dataset generated via Belunga using a one-shot approach
    dataset_b = load_dataset("datasets/dataset_b.json")

    datasets = {
        "dataset_zero_shot": split_dataset(dataset_a),
        "dataset_one_shot": split_dataset(dataset_b),
    }

    # Build prompts
    # Load templates
    templates = load_template("./system_prompt_templates")
    models_configs = load_all_configs("./models_configs")

    prompts = make_prompts(templates, models_configs, datasets)

    exit()
    for config_file, model_config in models_configs.items():
        # Extract model info
        model_info = model_config.get('model', {})
        friendly_name = model_info.get('friendly_name', config_file)
        prompt_config = model_config.get('prompt', {})
        system_prompt_template = prompt_config.get(
            'system_template', 'default')
        system_prompt_template = jinja2.Template(system_prompt_template)

        test = render_template(system_prompt_template, {})

    # Render templates
    # Tricky bit, we need template for each system prompt, for each models, for each datasets


if __name__ == '__main__':
    main()
