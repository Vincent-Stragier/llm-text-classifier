# LLM based text classifier

This project is a compilation of scripts used to study the classification performance of LLMs. It's a naïve approach to the problem, based on the assumption that logits can be blindly used for the classification of texts. The main goal is to understand the performance of LLMs in a classification task, and to compare it with traditional methods.

Some improvement can already be implemented, like better prompt. Another improvement would be to let the model start generating (using greedy decoding) and then force the generation of the "label" token, as we do here. Another improvement would be to train the model on the classification task, instead of using a pre-trained model.

## Remarks

- It seems that small models are bad for the classification task. They prefer to generate spaces, newlines, sentences before generating the class tokens.
  - It could be a solution to implement the previously proposed idea. Fine-tuning the model on the classification task could also be a solution.
- When providing a `unknown_class` or `another_class`, the model tends to generate the `unknown_class` or `another_class` tokens. This is obviously a problem, however, using bigger models and better prompts could solve this issue.
- Fine-tuning the model on the classification task should be explored.

## Installation

Here is the installation process for CPU only, Linux based systems:

```bash
# Clone the repository
git clone https://github.com/Vincent-Stragier/llm-text-classifier.git
cd llm-text-classifier

# Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install the requirements
pip install -r requirements.txt
```

## Usage

### Synthetic datasets

First, you must download the synthetic datasets, `python download_datasets.py`. This dataset is aimed toward the classification of visually impaired and blind individuals requests.

### Prompts generation

Then you must generate the prompts, they are based on various templates, `python do_preprocess.py`. The generated prompts are stored in the `prompts` directory.

#### System prompt templates

The system prompt templates are stored in the `system_prompt_templates` directory. They are used to generate the system prompts. You can use `{{ classes_list }}` and `{{ examples_from_train_list }}` to insert the classes and examples from the training set.

#### Model specific prompt templates

Each model has its own prompt template. They are stored in the models' configuration files, which are stored in the `models_config` directory.

For instance:

```yaml
---
model:
  hub: TheBloke/Llama-2-70B-chat-GGUF
  file: https://huggingface.co/TheBloke/Llama-2-70B-Chat-GGUF/raw/main/llama-2-70b-chat.Q4_K_M.gguf
  local_path: ../models/llama-2-70b-chat.Q4_K_M.gguf
  friendly_name: llama_2_70b_chat
  description: Llama 2.70B Chat model

prompt:
  system_template: "[INST]<<SYS>>{{system_prompt}}<</SYS>>{{prompt}}[/INST]"
```

Where `{{system_prompt}}` is the system prompt and `{{prompt}}` is the user prompt.

### Classification

Finally, you can classify the examples, `python do_classification.py`. The classification results are stored in the `result` directory.

<!-- ### Metrics

Now you can compute the metrics, `python do_metrics.py`. The metrics are stored in the `metrics` directory. -->

## Standalone classification

You could also use the classification alone, without generating the prompts. Here is an example:

```python
from huggingface_hub import hf_hub_download
from classifier import Classifier

# Load the model from the Hugging Face Hub
llama_model_path = hf_hub_download(
    "TheBloke/Llama-2-7B-chat-GGUF",
    "llama-2-7b-chat.Q5_K_M.gguf",
    cache_dir="./models",
    revision="main",
)

my_classifier = Classifier(
    # You are not forced to use the Llama model
    # And you are not forced provide the classes when you instanciate the classifier
    llama_model_path, ["positive", "negative", "neutral", "another_class"]
)

probabilities = my_classifier.classify(
    b"[INST]You must classify the following sentence as "
    b"'positive', 'negative', 'neutral' or 'another_class',"
    b"only respond in lowercase with one of the previously"
    b" mentioned class name:\n"
    b"'You are a loser!'[\\INST]\n",
    # You can also provide the classes when you call the classify method
    classes=["positive", "negative", "neutral", "another_class"]
)
print("One shot classification")
print(probabilities)

probabilities = my_classifier.classify(
    b"[INST]'You are a loser!'[\\INST]\n"
)
print("Zero shot classification")
print(probabilities)
```
