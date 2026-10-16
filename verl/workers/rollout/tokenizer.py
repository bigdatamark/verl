# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# [MODULE DOC]
# Defines the abstract tokenizer interface that a hybrid-engine rollout must provide so
# that vLLM can use a non-HF tokenizer. Exports HybridEngineBaseTokenizer only.
"""
The base tokenizer class, required for any hybrid engine based rollout or inference with vLLM.
"""

from abc import ABC, abstractmethod

import numpy as np
import torch

__all__ = ["HybridEngineBaseTokenizer"]


class HybridEngineBaseTokenizer(ABC):
    # [CLASS DOC]
    # Abstract base class mirroring the subset of the HuggingFace tokenizer API vLLM relies
    # on: vocab/special-token properties, encode/decode, id/token conversion, added vocab.
    # Subclasses must implement every abstract member; is_fast defaults to False.
    """the tokenizer property and function name should align with HF's to meet vllm requirement"""

    @property
    @abstractmethod
    def vocab_size(self):
        # [FUNCTION DOC]
        # Abstract property: size of the base vocabulary (int), excluding added tokens.
        """
        `int`: Size of the base vocabulary (without the added tokens).
        """
        pass

    @property
    @abstractmethod
    def pad_token_id(self):
        # [FUNCTION DOC]
        # Abstract property: padding token id, or None when unset.
        """
        `Optional[int]`: Id of the padding token in the vocabulary. Returns `None` if the token has not been set.
        """
        pass

    @property
    @abstractmethod
    def eos_token_id(self):
        # [FUNCTION DOC]
        # Abstract property: end-of-sequence token id, or None when unset.
        """
        `Optional[int]`: Id of the end of sentence token in the vocabulary. Returns `None` if the token has not been
        set.
        """
        pass

    @property
    @abstractmethod
    def all_special_ids(self) -> list[int]:
        # [FUNCTION DOC]
        # Abstract property: list of ids of all special tokens.
        """
        `List[int]`: List the ids of the special tokens(`'<unk>'`, `'<cls>'`, etc.) mapped to class attributes.
        """
        pass

    @property
    @abstractmethod
    def all_special_tokens(self) -> list[str]:
        # [FUNCTION DOC]
        # Abstract property: list of unique special token strings (AddedToken converted to str).
        """
        `List[str]`: A list of the unique special tokens (`'<unk>'`, `'<cls>'`, ..., etc.).

        Convert tokens of `tokenizers.AddedToken` type to string.
        """
        pass

    @abstractmethod
    def encode(self, text):
        # [FUNCTION DOC]
        # Abstract: encode text (string, list of tokens, or ids) into a list of token ids.
        # KNOWN ISSUE: docstring documents a text_pair argument that the signature does not accept.
        """
        Converts a string to a sequence of ids (integer), using the tokenizer and vocabulary.

        Args:
            text (`str`, `List[str]` or `List[int]`):
                The first sequence to be encoded. This can be a string, a list of strings (tokenized string using the
                `tokenize` method) or a list of integers.

            text_pair (`str`, `List[str]` or `List[int]`, *optional*):
                Optional second sequence to be encoded. This can be a string, a list of strings (tokenized string using
                the `tokenize` method) or a list of integers.
        """
        pass

    @abstractmethod
    def decode(
        self,
        token_ids: int | list[int] | np.ndarray | torch.Tensor,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        **kwargs,
    ) -> str:
        # [FUNCTION DOC]
        # Abstract: decode ids (int, list, numpy array or tensor) to a string, optionally skipping
        # special tokens and cleaning tokenization spaces; extra kwargs go to the model-specific
        # decoder.
        """
        Converts a sequence of ids in a string, using the tokenizer and vocabulary with options to remove special
        tokens and clean up tokenization spaces.

        Similar to doing `self.convert_tokens_to_string(self.convert_ids_to_tokens(token_ids))`.

        Args:
            token_ids (`Union[int, List[int], np.ndarray, torch.Tensor]`):
                List of tokenized input ids. Can be obtained using the `__call__` method.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.
            clean_up_tokenization_spaces (`bool`, *optional*):
                Whether or not to clean up the tokenization spaces. If `None`, will default to
                `self.clean_up_tokenization_spaces`.
            kwargs (additional keyword arguments, *optional*):
                Will be passed to the underlying model specific decode method.

        Returns:
            `str`: The decoded sentence.
        """
        pass

    @abstractmethod
    def convert_ids_to_tokens(self, ids: int | list[int], skip_special_tokens: bool = False) -> str | list[str]:
        # [FUNCTION DOC]
        # Abstract: map an id or list of ids to the corresponding token or list of tokens,
        # optionally skipping special tokens.
        """
        Converts a single index or a sequence of indices in a token or a sequence of tokens, using the vocabulary and
        added tokens.

        Args:
            ids (`int` or `List[int]`):
                The token id (or token ids) to convert to tokens.
            skip_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not to remove special tokens in the decoding.

        Returns:
            `str` or `List[str]`: The decoded token(s).
        """
        pass

    @abstractmethod
    def get_added_vocab(self) -> dict[str, int]:
        # [FUNCTION DOC]
        # Abstract: return a mapping from added-token string to id.
        """
        Returns the added tokens in the vocabulary as a dictionary of token to index. Results might be different from
        the fast call because for now we always add the tokens even if they are already in the vocabulary. This is
        something we should change.

        Returns:
            `Dict[str, int]`: The added tokens.
        """
        pass

    @abstractmethod
    def convert_tokens_to_string(self, tokens: list[str]) -> str:
        # [FUNCTION DOC]
        # Abstract: join a list of tokens into a string, removing sub-word artifacts.
        """
        Converts a sequence of tokens in a single string. The most simple way to do it is `" ".join(tokens)` but we
        often want to remove sub-word tokenization artifacts at the same time.

        Args:
            tokens (`List[str]`): The token to join in a string.

        Returns:
            `str`: The joined tokens.
        """
        pass

    @property
    def is_fast(self):
        # [FUNCTION DOC]
        # Property: whether this is a fast (Rust) tokenizer; always False here.
        return False
