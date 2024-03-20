import ctypes
import os

# from traceback import format_exc
from typing import Iterable

import numpy

from utils import compute_cumulative_probabilities

import llama_cpp


class Classifier:
    """An LLM based classifier."""

    def __init__(
        self,
        model_path: str,
        classes: Iterable[str],
        n_ctx: int = 512,
        n_new_tokens: int = 32,
        # Should be changed to the number of classes
        # n_parallel: int = 1,
        n_threads: int = os.cpu_count(),
        n_threads_batch: int = 1,
        n_gpu_layers: int = -1,
        numa=False,
    ):
        """Initialize the classifier.

        Args:
            model_path (str): The path to the model file.
            classes (Iterable[str]): The classes to classify.
            n_ctx (int): The context size.
            n_new_tokens (int): The number of new tokens.
            n_threads (int): The number of threads.
            n_threads_batch (int): The number of threads per batch.
            n_gpu_layers (int): The number of GPU layers.
            numa (bool): Whether to use NUMA.
        """
        llama_cpp.llama_backend_init(numa=numa)  # noqa

        # Initialize the model with the default parameters
        params = llama_cpp.llama_model_default_params()

        # Number of classes + 1 for unknown class
        self.classes = classes
        self.n_parallel = len(classes) + 1

        params.n_ctx = n_ctx
        # params.n_parallel = self.n_parallel
        params.n_batch = max(n_ctx, self.n_parallel)
        params.n_threads = n_threads
        params.n_threads_batch = n_threads_batch
        params.n_gpu_layers = n_gpu_layers

        if isinstance(model_path, str):
            model_path = model_path.encode("utf-8")

        self.model = llama_cpp.llama_load_model_from_file(
            model_path, params
        )

        self.n_new_tokens = n_new_tokens

        self.tokens = (llama_cpp.llama_token * n_ctx)()

        # Pre-allocate the KV cache (-1 means not initialized)
        self.n_kv_req = -1

        self.ctx = -1
        self.n_ctx = -1

        self.batch = None

        self.tokens_len = -1

    def _tokenize_initial_prompt(self, prompt: bytes, n_new_tokens: int = 32):
        self.tokens_len = llama_cpp.llama_tokenize(
            self.model,
            prompt,
            len(prompt),
            self.tokens,
            len(self.tokens),
            True,
            True,
        )

        # Pre-allocate the KV cache
        if self.n_kv_req == -1:
            if self.n_new_tokens != n_new_tokens:
                self.n_new_tokens = n_new_tokens

            self.n_kv_req = (
                self.tokens_len
                + (self.n_new_tokens - self.tokens_len) * self.n_parallel
            )

        return self.tokens[: self.tokens_len]

    def _init_context(
        self,
        random_seed: int = 3254,
        n_threads: int = os.cpu_count(),
        n_threads_batch: int = 1,
    ):
        ctx_params = llama_cpp.llama_context_default_params()
        ctx_params.seed = random_seed
        ctx_params.n_ctx = self.n_kv_req
        ctx_params.n_batch = max(self.n_new_tokens, self.n_parallel)
        ctx_params.n_threads = n_threads
        ctx_params.n_threads_batch = n_threads_batch

        self.ctx = llama_cpp.llama_new_context_with_model(
            self.model, ctx_params
        )

        self.n_ctx = llama_cpp.llama_n_ctx(self.ctx)

    def _init_batch(self):
        if self.tokens_len == -1:
            raise ValueError(
                "Tokens length not initialized. "
                "Call _tokenize_initial_prompt first."
            )

        self.batch = llama_cpp.llama_batch_init(
            max(self.tokens_len, self.n_parallel), 0, 1
        )

        self.batch.n_tokens = self.tokens_len
        for index in range(self.tokens_len):
            self.batch.token[index] = self.tokens[index]
            self.batch.pos[index] = index
            self.batch.seq_id[index][0] = 0
            self.batch.n_seq_id[index] = 1
            self.batch.logits[index] = False

        self.batch.logits[self.batch.n_tokens - 1] = True

        if llama_cpp.llama_decode(self.ctx, self.batch) != 0:
            raise ValueError("Failed to initialize the batch. Error decoding.")

        # Initialize the KV cache
        for index in range(self.n_parallel):
            llama_cpp.llama_kv_cache_seq_cp(
                self.ctx, 0, index, 0, self.batch.n_tokens
            )

    def _tokenize_string(
        self, string: bytes | str, add_bos: bool = False, special: bool = True
    ) -> list[int]:
        # Adapte context size to the string length
        # A safe value is the string length + 1
        if isinstance(string, str):
            string = string.encode("utf-8")

        context_size = len(string) + 1

        tokens = (llama_cpp.llama_token * context_size)()

        tokens_len = llama_cpp.llama_tokenize(
            # model
            self.model,
            # text
            string,
            # text_len
            len(string),
            # tokens
            tokens,
            # n_max_tokens
            len(tokens),
            # add_bos
            add_bos,
            # special
            special,
        )

        return list(tokens[:tokens_len])

    def _add_token_to_batch(
        self,
        token_id: int,
        pos: int,
        seq_id: int,
        n_seq_id: int = 1,
        logits: bool = True,
    ):
        self.batch.token[self.batch.n_tokens] = token_id
        # n_cur = pos
        self.batch.pos[self.batch.n_tokens] = pos
        # seq_id = index
        self.batch.seq_id[self.batch.n_tokens][0] = seq_id
        self.batch.n_seq_id[self.batch.n_tokens] = n_seq_id
        self.batch.logits[self.batch.n_tokens] = logits

        self.batch.n_tokens += 1

    def _decode_token(self, token_id: int) -> str:
        buffer = (ctypes.c_char * 32)()
        out_len = llama_cpp.llama_token_to_piece(
            self.model, token_id, buffer, len(buffer)
        )

        return bytes(buffer[:out_len]).decode("utf-8")

    def _append_to_dict(self, dictionary, key, value):
        if key not in dictionary:
            dictionary[key] = [value]

        else:
            dictionary[key].append(value)

    def _do_classification(
        self,
        add_most_likely_token: bool = False,
    ):
        classes_tokens_and_logit = {
            class_name: [] for class_name in self.classes
        }

        most_likely_tokens_index = self.n_parallel - 1
        if add_most_likely_token:
            classes_tokens_and_logit.update({"most_likely": []})

        i_batch = [self.batch.n_tokens - 1] * self.n_parallel

        tokenized_classes = [
            self._tokenize_string(current_class)
            for current_class in self.classes
        ]

        n_cur_initial = self.batch.n_tokens
        n_cur = n_cur_initial
        n_decode = 0

        # print(f'Initially {i_batch=}')

        while n_cur <= self.n_new_tokens:
            self.batch.n_tokens = 0

            # Go through all the classes, except the unknown class
            for index, current_class in enumerate(self.classes):
                token_id_index = n_cur - n_cur_initial

                if (
                    token_id_index >= len(tokenized_classes[index])
                    or i_batch[index] == -1
                ):
                    i_batch[index] = -1
                    continue

                current_class_token = tokenized_classes[index][token_id_index]

                # print(f'Before logits {i_batch=}, {i_batch[index]=}')
                logits = llama_cpp.llama_get_logits_ith(
                    self.ctx, i_batch[index]
                )

                self._append_to_dict(
                    classes_tokens_and_logit,
                    current_class,
                    {
                        "logit": logits[current_class_token],  # noqa
                        "token": current_class_token,
                        "token_str": self._decode_token(current_class_token),
                    },
                )

                if (
                    current_class_token == llama_cpp.llama_token_eos(self.ctx)
                    or n_cur >= self.n_new_tokens
                ):
                    i_batch[index] = -1
                    continue

                self._add_token_to_batch(current_class_token, n_cur, index)
                i_batch[index] = self.batch.n_tokens
                n_decode += 1

            # Generate most likely token for unknown class
            # if we have not reached the end
            if (
                i_batch[most_likely_tokens_index] != -1
                and add_most_likely_token
            ):
                logits = llama_cpp.llama_get_logits_ith(
                    self.ctx, i_batch[most_likely_tokens_index]
                )

                n_vocabulary = llama_cpp.llama_n_vocab(self.model)
                converted_logits = logits[:n_vocabulary]  # noqa
                converted_logits = numpy.array(converted_logits)

                max_logit = converted_logits.max()
                most_likely_token = converted_logits.argmax()

                self._append_to_dict(
                    classes_tokens_and_logit,
                    "unknown",
                    {
                        "logit": max_logit,
                        "token": most_likely_token,
                        "token_str": self._decode_token(most_likely_token),
                    },
                )

                if (
                    most_likely_token == llama_cpp.llama_token_eos(self.ctx)
                    or n_cur >= self.n_new_tokens
                ):
                    i_batch[most_likely_tokens_index] = -1

                else:
                    self._add_token_to_batch(
                        most_likely_token, n_cur, most_likely_tokens_index
                    )
                    # Update the index
                    i_batch[most_likely_tokens_index] = self.batch.n_tokens

            # If we have reached the end of the tokens, break
            if self.batch.n_tokens == 0:
                break

            n_cur += 1

            if llama_cpp.llama_decode(self.ctx, self.batch) != 0:
                raise ValueError("Error decoding", flush=True)

        return classes_tokens_and_logit

    def classify(
        self, prompt: bytes | str, classes: Iterable = None, max_new_tokens: int = 512
    ) -> dict[str, float]:
        """Classify the prompt.

        Args:
            prompt (bytes): The prompt to classify.
            classes (Iterable, optional): The classes to classify.
            max_new_tokens (int, optional): The maximum number of new tokens.
            Defaults to 512.

        Returns:
            dict[str, float]: The probabilities of the classes.

        """
        if isinstance(prompt, str):
            # The encoding is a bit tricky, let's assume utf-8.
            prompt = prompt.encode("utf-8")

        if classes is not None:
            self.classes = classes
            self.n_parallel = len(classes) + 1

        self._tokenize_initial_prompt(prompt, max_new_tokens)
        self._init_context()
        self._init_batch()

        # Do the classification here
        classes_logits = self._do_classification()

        # Free the resources
        self.free_batch()
        self.free_context()

        return compute_cumulative_probabilities(classes_logits), classes_logits

    def free_batch(self):
        """Free the batch."""
        try:
            llama_cpp.llama_batch_free(self.batch)
        except OSError:
            pass

    def free_context(self):
        """Free the context."""
        try:
            llama_cpp.llama_free(self.ctx)
        except OSError:
            pass

    def free_model(self):
        """Free the model."""
        try:
            llama_cpp.llama_free_model(self.model)
        except OSError:
            pass

    def free_backend(self):
        """Free the backend."""
        try:
            llama_cpp.llama_backend_free()
        except OSError:
            pass

    def __del__(self):
        """Free the resources when the object is deleted."""
        self.free_batch()
        self.free_context()
        self.free_model()
        self.free_backend()


if __name__ == "__main__":
    from huggingface_hub import hf_hub_download

    # Load the model from the Hugging Face Hub
    llama_model_path = hf_hub_download(
        "TheBloke/Llama-2-7B-chat-GGUF",
        "llama-2-7b-chat.Q5_K_M.gguf",
        cache_dir="./models",
        revision="main",
    )

    my_classifier = Classifier(
        llama_model_path, ["positive", "negative", "neutral", "another_class"]
    )

    probabilities = my_classifier.classify(
        b"[INST]You must classify the following sentence as "
        b"'positive', 'negative', 'neutral' or 'another_class',"
        b"only respond in lowercase with one of the previously"
        b" mentioned class name:\n"
        b"'You are a loser!'[\\INST]\n"
    )
    print("One shot classification")
    print(probabilities)

    probabilities = my_classifier.classify(
        b"[INST]'You are a loser!'[\\INST]\n")
    print("Zero shot classification")
    print(probabilities)
