"""
Do classifications with Mixtral Instruct, Mistral Instruct,
and all the Llama 2 chat models.
"""

import json
import os
import pathlib

from copy import deepcopy
from typing import Iterable

import numpy as np
import jinja2

from constants import DEFAULT_PROMPTS_PATH, PATH_DATASET_A, PATH_DATASET_B
from utils import load_all_configs, load_dataset


def split_dataset(
    dataset: Iterable,
    split: Iterable = (0.33, 0.33, 0.33),
    random_state: int = 53,
):
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
        subdataset_length = len(tool["dataset"])
        adapted_split = np.array(
            normalized_split * subdataset_length, dtype=int
        ).cumsum()

        # Shuffle the dataset
        subdataset = deepcopy(tool["dataset"])
        np.random.shuffle(subdataset)

        # Split the dataset
        assert len(adapted_split) >= 3
        train, test, validation = np.split(subdataset, adapted_split)[
            :3
        ]  # noqa
        splitted_dataset[index]["dataset"] = {
            "train": train,
            "test": test,
            "validation": validation,
        }

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
        with open(
            template_file_path, mode="r", encoding="utf-8"
        ) as template_file:
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
    return template.render(**content)


def generate_example_text(example: dict) -> str:
    """Generate the example from test set.

    Args:
        example (dict): the element extracted from the test set

    Returns:
        str: a formatted string
    """
    return (
        f"```\n{example.get('user_request', None)}[/INST]\n"
        f"{example.get('command', None)}\n```"
    )


def make_prompts(
    prompts_templates,
    models_dict,
    datasets,
    root=DEFAULT_PROMPTS_PATH,
    random_seed: int = 5876,
):
    # Models

    expected_class = {}

    for config_file, model_config in models_dict.items():
        # Extract model info
        model_info = model_config.get("model", {})
        friendly_name = model_info.get("friendly_name", config_file)
        prompt_config = model_config.get("prompt", {})
        system_prompt_template = prompt_config.get(
            "system_template", "default"
        )
        model_template = jinja2.Template(system_prompt_template)

        for dataset_name, dataset in datasets.items():
            # All the tools in the dataset
            tools = sorted(
                list({tool.get("tool_name", None) for tool in dataset})
            )

            classes_list = "\n-"
            classes_list = f"-{classes_list.join(tools)}"

            for tool in dataset:
                tool_name = tool.get("tool_name", None)
                test_set = tool.get("dataset", {}).get("test", None)
                train_set = deepcopy(
                    tool.get("dataset", {}).get("train", None)
                )

                if test_set is None:
                    raise ValueError(
                        f"Test set is missing for tool {tool['tool_name']}"
                    )

                if train_set is None:
                    raise ValueError(
                        f"Train set is missing for tool {tool['tool_name']}"
                    )

                # randomly select an element from the train set
                # using a specific seed
                np.random.seed(random_seed)
                selected_element = np.random.choice(train_set)

                # print(model)
                examples_from_train_list = generate_example_text(
                    selected_element
                )

                element = test_set[0]

                for (
                    template_name,
                    prompt_template,
                ) in prompts_templates.items():
                    # Create the directory for the tools prompts
                    save_path = pathlib.Path(
                        f"{root}/{friendly_name}/{dataset_name}/"
                        f"{template_name}/{tool_name}"
                    )
                    save_path.mkdir(parents=True, exist_ok=True)

                    template_params = {
                        "classes_list": classes_list,
                        "examples_from_train_list": examples_from_train_list,
                    }

                    system_prompt = render_template(
                        prompt_template,
                        template_params,
                    )

                    for index, element in enumerate(test_set):
                        element_prompt = render_template(
                            model_template,
                            {
                                "system_prompt": system_prompt,
                                "prompt": element.get("user_request", None),
                            },
                        )

                        # Save element_prompt
                        element_prompt_path = save_path / f"prompt_{index}.txt"
                        expected_class[str(element_prompt_path)] = element.get(
                            "command", None
                        )

                        with element_prompt_path.open(
                            mode="w", encoding="utf-8"
                        ) as prompt_file:
                            prompt_file.write(element_prompt)

                        # We can save the expected class here

        root = pathlib.Path(root)
        ground_truth = root / "ground_truth.json"
        with ground_truth.open(
            mode="w", encoding="utf-8"
        ) as ground_truth_file:
            ground_truth_file.write(json.dumps(expected_class, indent=4))


def main():
    """The main function of the script."""
    # Dataset generated via StableBeluga2 using a zero-shot approach
    dataset_a = load_dataset(PATH_DATASET_A)
    # Dataset generated via StableBeluga2 using a one-shot approach
    dataset_b = load_dataset(PATH_DATASET_B)

    datasets = {
        "dataset_zero_shot": split_dataset(dataset_a),
        "dataset_one_shot": split_dataset(dataset_b),
    }

    # Build prompts
    # Load templates
    templates = load_template("./system_prompt_templates")
    models_configs = load_all_configs("./models_configs")

    make_prompts(templates, models_configs, datasets)


if __name__ == "__main__":
    main()
