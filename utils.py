"""The utility functions of the project."""
import os
import yaml

import numpy


def softmax(logits):
    """Compute the softmax of the logits.

    Args:
        logits (numpy.ndarray): The logits.

    Returns:
        numpy.ndarray: The softmax of the logits.
    """
    e_x = numpy.exp(logits - numpy.max(logits))
    return e_x / e_x.sum(axis=0)


def log_softmax(logits):
    """Compute the log softmax of the logits.

    Log softmax function for numerical stability.

    Args:
        logits (numpy.ndarray): The logits.

    Returns:
        numpy.ndarray: The log softmax of the logits.
    """
    return numpy.log(softmax(logits))


def compute_cumulative_probabilities(classes_tokens_and_logits: dict):
    """Compute the cumulative probabilities of the classes.

    Args:
        classes_tokens_and_logits (dict):
        A dictionary with the classes tokens and logits.
        The key is the class name and the value is a list
        of tokens and their logits.

    Returns:
        dict: A dictionary with the classes probabilities.
        The key is the class name and the value is the class probability.
    """
    classes_probabilities = {}
    classes_logits = {}

    for class_name, tokens in classes_tokens_and_logits.items():
        _logits = [token_info['logit'] for token_info in tokens]
        classes_logits[class_name] = sum(_logits)

    _logits = numpy.array(list(classes_logits.values()))
    _probabilities = softmax(_logits)

    for index, class_name in enumerate(classes_logits.keys()):
        classes_probabilities[class_name] = _probabilities[index]

    return classes_probabilities
    # normalized_log_probs = {}

    # for class_name, tokens in classes_tokens_and_logits.items():
    #     logits = [token_info['logit'] for token_info in tokens]
    #     log_probs = log_softmax(logits)
    #     # Sum log probabilities and normalize by the number of tokens
    #     normalized_log_prob = numpy.sum(log_probs) / len(tokens)
    #     normalized_log_probs[class_name] = normalized_log_prob

    # # If you need them back in probability space
    # normalized_probs = {k: numpy.exp(v)
    #                     for k, v in normalized_log_probs.items()}

    # print(normalized_log_probs)
    # print(normalized_probs)


def ensure_path_exists(path: os.PathLike):
    """Ensure the path exists."""
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
