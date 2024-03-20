import json
import pathlib

from tqdm import tqdm
from huggingface_hub import hf_hub_download


from classifiers import Classifier
from constants import (
    DEFAULT_MODELS_CONFIGS_PATH,
    DEFAULT_PROMPTS_PATH,
    PATH_DATASET_A,
    PATH_DATASET_B,
)
from utils import (
    load_all_configs,
    make_result_path,
    load_dataset,
)


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

        with file.open(mode="r", encoding="utf-8") as file_processor:
            file_content = file_processor.read()

        if file.suffix == ".json":
            ground_truth = file_content
            continue

        prompts_dict.update({str(file): file_content})

    return prompts_dict, ground_truth


def get_classes_by_dataset():
    """Load the classes from the datasets."""
    # Dataset generated via StableBeluga2 using a zero-shot approach
    dataset_a = load_dataset(PATH_DATASET_A)
    # Dataset generated via StableBeluga2 using a one-shot approach
    dataset_b = load_dataset(PATH_DATASET_B)

    def get_classes_from_dataset(dataset):
        """Get the classes from a dataset."""
        return sorted(list({tool.get("tool_name", None) for tool in dataset}))

    return {
        "dataset_zero_shot": get_classes_from_dataset(dataset_a),
        "dataset_one_shot": get_classes_from_dataset(dataset_b),
    }


def main():
    models_configs = load_all_configs(DEFAULT_MODELS_CONFIGS_PATH)
    prompts, ground_truth = load_all_prompts(DEFAULT_PROMPTS_PATH)

    ground_truth = json.loads(ground_truth)
    classes_by_dataset = get_classes_by_dataset()

    def append_to_dict(dictionary, key, value):
        if key not in dictionary:
            dictionary[key] = [value]

        else:
            dictionary[key].append(value)

    prompts_by_model = {}
    for path, prompt in prompts.items():
        model_friendly_name = path.replace("/", "\\").split("\\")[1]

        # Check if results are already available
        # If so, we can skip the prompt

        result_path = make_result_path(path)

        if not result_path.exists():
            truth = ground_truth.get(path, None)
            append_to_dict(
                prompts_by_model,
                model_friendly_name,
                {path: (prompt, result_path, truth)},
            )

    for _, model_config in tqdm(models_configs.items()):
        model_friendly_name = model_config.get("model", {}).get(
            "friendly_name"
        )

        model_repo_id = model_config.get("model", {}).get("hub")

        model_cache_dir = pathlib.Path(
            model_config.get("model", {}).get("local_path")
        ).parent

        model_file = pathlib.Path(
            model_config.get("model", {}).get("local_path")
        ).name

        # We should load the model path here
        llm_model_path = hf_hub_download(
            model_repo_id,
            model_file,
            cache_dir=model_cache_dir,
            revision="main",
        )

        classifier = Classifier(
            llm_model_path,
            # We will define the classes later
            classes=[],
            n_gpu_layers=-1,
        )

        for path_and_prompt in tqdm(
            prompts_by_model.get(model_friendly_name, [])
        ):
            prompt_path, prompt_and_result_path = list(
                path_and_prompt.items()
            )[0]
            prompt = prompt_and_result_path[0]
            result_path = prompt_and_result_path[1]
            truth = prompt_and_result_path[2]

            dataset_name = prompt_path.split("\\")[2]

            result = classifier.classify(
                prompt, classes_by_dataset.get(dataset_name)
            )

            result_classes_and_truth = {
                "result": result,
                "truth": truth,
                "classes": classes_by_dataset.get(dataset_name),
            }

            # Save the result
            result_path = make_result_path(prompt_path)
            # Ensure the directory exists
            result_path.parent.mkdir(parents=True, exist_ok=True)

            with result_path.open(
                mode="w", encoding="utf-8"
            ) as file_processor:
                json.dump(result_classes_and_truth, file_processor, indent=4)

        del classifier


if __name__ == "__main__":
    main()
