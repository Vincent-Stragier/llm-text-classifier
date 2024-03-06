"""
Do classifications with Mixtral Instruct, Mistral Instruct, and all the Llama 2 chat models.
"""
import json
import os

from copy import deepcopy
from typing import Iterable

import numpy as np
import jinja2


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


def generate_prompts(tools: str):
    pass

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
    template: Iterable,
    content: dict,
):
    rendered_template = template.render(
        name='Jane Doe',
        kwargs={'name': "John Doe"},
    )


def main():
    dataset_a = load_dataset("datasets/dataset_a.json")
    dataset_b = load_dataset("datasets/dataset_b.json")

    split_a = split_dataset(dataset_a)
    split_b = split_dataset(dataset_b)

    print(split_a)
    print(split_b)

    # Build prompts
    # Load templates
    templates = load_template("system_prompt_templates")
    print(templates)
    # Render templates
    # Tricky bit, we need template for each system prompt, for each models, for each datasets


if __name__ == '__main__':
    main()
