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
# Shared infrastructure for the model merger: CLI argument parsing, the ModelMergerConfig dataclass, and the
# abstract BaseModelMerger which loads the HF config and implements saving HF model/tokenizer/processor, generation
# config, LoRA adapters, and uploading to the HuggingFace Hub. Backend subclasses implement loading/merging shards.

import argparse
import os
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import torch
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM, AutoModelForTokenClassification, GenerationConfig

from verl.utils import hf_processor, hf_tokenizer
from verl.utils.transformers_compat import drop_tied_target_keys, get_auto_model_for_vision2seq

from .output_validation import validate_hf_model_output

AutoModelForVision2Seq = get_auto_model_for_vision2seq()


def parse_args():
    # [FUNCTION DOC]
    # Behavior: builds an argparse parser with required subcommand "merge" or "test". Shared options: --backend
    # (fsdp|megatron, required), --local_dir, --tie-word-embedding, --trust-remote-code, --is-value-model,
    # --use_cpu_initialization (flags). merge adds --target_dir (default "tmp"), --hf_upload_path, --private;
    # test adds required --test_hf_dir. Parses sys.argv.
    # Outputs: argparse Namespace (operation holds the subcommand name).
    # Exceptions: argparse exits the process on invalid arguments.
    parser = argparse.ArgumentParser(description="verl model merger")
    subparsers = parser.add_subparsers(dest="operation", required=True, help="Specify 'merge' or 'test' operation.")

    base_op_parser = argparse.ArgumentParser(add_help=False)
    base_op_parser.add_argument(
        "--backend", type=str, required=True, choices=["fsdp", "megatron"], help="The backend of the model"
    )
    base_op_parser.add_argument("--local_dir", type=str, default=None, help="Path to the saved model checkpoints.")
    base_op_parser.add_argument(
        "--tie-word-embedding",
        action="store_true",
        help="Whether to tie word embedding weights (currently only Megatron supported)",
    )
    base_op_parser.add_argument("--trust-remote-code", action="store_true", help="Whether to trust remote code")
    base_op_parser.add_argument(
        "--is-value-model",
        action="store_true",
        help="Whether the model is a value model (currently only Megatron supported)",
    )
    base_op_parser.add_argument(
        "--use_cpu_initialization",
        action="store_true",
        help="Whether to use CPU initialization for the model. This is useful for large models that cannot "
        "fit into GPU memory during initialization.",
    )

    merge_parser = subparsers.add_parser("merge", parents=[base_op_parser], help="Merge model checkpoints and save.")
    merge_parser.add_argument(
        "--target_dir", default="tmp", type=str, help="Directory to save the merged huggingface model"
    )
    merge_parser.add_argument(
        "--hf_upload_path", default=None, type=str, help="Hugging Face repository ID to upload the model"
    )
    merge_parser.add_argument(
        "--private", action="store_true", help="Whether to upload the model to a private Hugging Face repository"
    )

    test_parser = subparsers.add_parser(
        "test", parents=[base_op_parser], help="Test merged model against a reference Hugging Face model"
    )
    test_parser.add_argument(
        "--test_hf_dir", type=str, required=True, help="Path to the reference Hugging Face model directory for testing"
    )

    args = parser.parse_args()
    return args


@dataclass
class ModelMergerConfig:
    # [CLASS DOC]
    # Dataclass holding all merger options (see docstring). hf_upload is derived, not passed to the constructor.
    """Configuration for model merger operations.

    Args:
        operation (str): Operation type - 'merge' or 'test'.
        backend (str): Backend type for the model ('fsdp' or 'megatron').
        target_dir (Optional[str]): Directory to save the merged huggingface model. Defaults to "tmp".
        hf_upload_path (Optional[str]): Hugging Face repository ID to upload the model. Defaults to None.
        private (bool): Whether to upload the model to a private Hugging Face repository. Defaults to False.
        test_hf_dir (Optional[str]): Path to the reference Hugging Face model directory for testing. Defaults to None.
        tie_word_embedding (bool): Whether to tie word embedding weights (currently only Megatron
            supported). Defaults to False.
        trust_remote_code (bool): Whether to trust remote code. Defaults to False.
        is_value_model (bool): Whether the model is a value model (currently only Megatron
            supported). Defaults to False.
        local_dir (Optional[str]): Path to the saved model checkpoints. Defaults to None.
        hf_model_config_path (Optional[str]): Path to HuggingFace model configuration files. Defaults to None.
        hf_upload (bool): Whether to upload to HuggingFace (computed automatically). Not for initialization.
        use_cpu_initialization (bool): Whether to use CPU initialization for large models. Defaults to False.
    """

    operation: str  # 'merge' or 'test'
    backend: str
    target_dir: Optional[str] = "tmp"
    hf_upload_path: Optional[str] = None
    private: bool = False
    test_hf_dir: Optional[str] = None
    tie_word_embedding: bool = False
    trust_remote_code: bool = False
    is_value_model: bool = False
    local_dir: Optional[str] = None
    hf_model_config_path: Optional[str] = None
    hf_upload: bool = field(init=False)
    use_cpu_initialization: bool = False

    def __post_init__(self):
        # [FUNCTION DOC]
        # Behavior: sets hf_upload true only for operation "merge" with a non-empty hf_upload_path; for "test" forces
        # target_dir and hf_upload_path to None and private to False.
        # Side effects: mutates fields.
        self.hf_upload = self.operation == "merge" and bool(self.hf_upload_path)
        if self.operation == "test":
            self.target_dir = None
            self.hf_upload_path = None
            self.private = False


def _default_hf_model_config_path(backend: str, local_dir: str) -> str:
    # [FUNCTION DOC]
    # HELPER: Returns the directory holding HF config/tokenizer files: local_dir/model/huggingface for megatron,
    # otherwise local_dir/huggingface.
    # Exceptions: TypeError from path joining if local_dir is None.
    """Return the default path to HuggingFace config/tokenizer artifacts.

    - FSDP: ``<local_dir>/huggingface`` (unchanged, single-tree layout).
    - Megatron (v2 layout): ``<local_dir>/model/huggingface`` — the HF tree
      is nested under ``model/`` alongside the optional ``model/dist_ckpt/``
      shards.  Checkpoints produced before the v2 refactor will therefore
      not resolve here; run ``scripts/migrate_megatron_checkpoint_layout.py``
      first.
    """
    if backend == "megatron":
        return os.path.join(local_dir, "model", "huggingface")
    return os.path.join(local_dir, "huggingface")


def generate_config_from_args(args: argparse.Namespace) -> ModelMergerConfig:
    # [FUNCTION DOC]
    # Behavior: builds a ModelMergerConfig from parsed args, computing hf_model_config_path from backend and local_dir.
    # For merge, passes target_dir, upload path and private and creates target_dir; for test, passes test_hf_dir and
    # clears merge-only fields.
    # Outputs: ModelMergerConfig.
    # Exceptions: NotImplementedError for an unknown operation.
    # Side effects: creates target_dir on disk for merge.
    common_config_args = {
        "operation": args.operation,
        "backend": args.backend,
        "tie_word_embedding": args.tie_word_embedding,
        "trust_remote_code": args.trust_remote_code,
        "is_value_model": args.is_value_model,
        "local_dir": args.local_dir,
        "hf_model_config_path": _default_hf_model_config_path(args.backend, args.local_dir),
        "use_cpu_initialization": args.use_cpu_initialization,
    }

    if args.operation == "merge":
        config = ModelMergerConfig(
            **common_config_args,
            target_dir=args.target_dir,
            hf_upload_path=args.hf_upload_path,
            private=args.private,
            test_hf_dir=None,
        )
        os.makedirs(config.target_dir, exist_ok=True)
    elif args.operation == "test":
        config = ModelMergerConfig(
            **common_config_args,
            test_hf_dir=args.test_hf_dir,
            # the following args are not used by test operation
            target_dir=None,
            hf_upload_path=None,
            private=False,
        )
    else:
        raise NotImplementedError(f"Unknown operation: {args.operation}")
    return config


class BaseModelMerger(ABC):
    # [CLASS DOC]
    # Abstract merger. Holds config, hf_model_config_path and the loaded HF model config (AutoConfig). Subclasses must
    # implement merge_and_save and cleanup.
    """
    Abstract base class for merging distributed model checkpoints into HuggingFace format.

    This class provides common functionality for converting model checkpoints from different
    distributed training backends (FSDP, Megatron) into standard HuggingFace format that
    can be easily loaded and used for inference or further training.

    The merger supports two main operations:
    - merge: Convert and save checkpoints to HuggingFace format
    - test: Validate merged checkpoints against a reference model

    Args:
        config (ModelMergerConfig): Configuration object containing paths, backend type,
            and operation parameters.

    Attributes:
        config (ModelMergerConfig): The configuration object passed during initialization.
        hf_model_config_path (str): Path to the HuggingFace model configuration files.
        model_config (PretrainedConfig): Loaded HuggingFace model configuration.
    """

    def __init__(self, config: ModelMergerConfig):
        # [FUNCTION DOC]
        # Behavior: stores config and loads the AutoConfig from hf_model_config_path honoring trust_remote_code.
        # Exceptions: propagates HF loading errors if the config is missing.
        # Side effects: reads config files from disk.
        self.config = config
        self.hf_model_config_path = config.hf_model_config_path
        self.model_config = AutoConfig.from_pretrained(
            self.hf_model_config_path, trust_remote_code=self.config.trust_remote_code
        )

    def get_transformers_auto_model_class(self):
        # [FUNCTION DOC]
        # Behavior: picks the transformers auto class for the model. If the config has auto_map with an entry whose value
        # contains the first architecture name (remote code), uses that entry's key: AutoModelForCausalLM,
        # AutoModelForTokenClassification, or the vision2seq class for AutoModelForVision2Seq / AutoModelForImageTextToText.
        # Otherwise infers from the first architecture name: contains ForTokenClassification, ForCausalLM, or
        # ForConditionalGeneration (vision2seq class), checked in that order.
        # Outputs: an auto model class.
        # Exceptions: NotImplementedError for unknown auto class or architecture.
        has_remote_code = hasattr(self.model_config, "auto_map") and any(
            self.model_config.architectures[0] in val for val in self.model_config.auto_map.values()
        )
        if has_remote_code:
            auto_class = next(
                k for k, v in self.model_config.auto_map.items() if self.model_config.architectures[0] in v
            )
            match auto_class:
                case "AutoModelForCausalLM":
                    return AutoModelForCausalLM
                case "AutoModelForTokenClassification":
                    return AutoModelForTokenClassification
                case "AutoModelForVision2Seq":
                    return AutoModelForVision2Seq
                case "AutoModelForImageTextToText":
                    return AutoModelForVision2Seq
                case _:
                    raise NotImplementedError(f"Unknown auto class {auto_class}")
        else:
            if "ForTokenClassification" in self.model_config.architectures[0]:
                return AutoModelForTokenClassification
            elif "ForCausalLM" in self.model_config.architectures[0]:
                return AutoModelForCausalLM
            elif "ForConditionalGeneration" in self.model_config.architectures[0]:
                return AutoModelForVision2Seq

            raise NotImplementedError(f"Unknown architecture {self.model_config.architectures}")

    def patch_model_generation_config(self, model):
        # [FUNCTION DOC]
        # Behavior: if the model can generate and a generation_config.json exists alongside the HF config, replaces the
        # model's generation_config with it.
        # Inputs: model - HF model. Outputs: the same model.
        # Side effects: mutates model; may print a warning when the file is missing.
        """
        The generation_config created from model config may be different to the pretrained model,
        this may lead to error when generating: https://github.com/verl-project/verl/issues/1246

        This function patch the generation_config created from model config to the pretrained model.
        """
        if model.can_generate():
            generation_config = self.load_generation_config()
            if generation_config is not None:
                model.generation_config = generation_config
        return model

    def load_generation_config(self, warn_if_missing: bool = True):
        # [FUNCTION DOC]
        # Behavior: returns the GenerationConfig loaded from hf_model_config_path if generation_config.json exists, else
        # None (printing a warning when warn_if_missing is true).
        # Inputs: warn_if_missing - whether to print the missing-file warning.
        # Outputs: GenerationConfig or None.
        """Load the generation config stored alongside the pretrained model config."""
        generation_config_path = os.path.join(self.hf_model_config_path, "generation_config.json")
        if not os.path.isfile(generation_config_path):
            if warn_if_missing:
                print(
                    f"Warning: Generation config file not found in {self.hf_model_config_path}, using a "
                    f"generation config created from the model config."
                )
            return None
        return GenerationConfig.from_pretrained(self.hf_model_config_path)

    def save_generation_config(self, target_dir):
        # [FUNCTION DOC]
        # Behavior: writes the source generation config into target_dir; if none exists, deletes any stale
        # generation_config.json in target_dir.
        # Inputs: target_dir - output directory.
        # Side effects: writes or removes files.
        """Preserve the pretrained generation config in a merged model directory."""
        generation_config = self.load_generation_config(warn_if_missing=False)
        if generation_config is not None:
            generation_config.save_pretrained(target_dir)
        else:
            stale_config_path = os.path.join(target_dir, "generation_config.json")
            if os.path.isfile(stale_config_path):
                os.remove(stale_config_path)

    def _load_lora_train_meta(self) -> Optional[dict[str, object]]:
        # [FUNCTION DOC]
        # HELPER: Reads optional LoRA training metadata from local_dir/lora_train_meta.json.
        # Behavior: returns None if local_dir is unset, the file is missing, or unreadable (warns). Otherwise extracts
        # "r" and "lora_alpha" coerced to int and "task_type" if a string, warning and skipping invalid values; a null
        # task_type is silently ignored.
        # Outputs: dict with any of r, lora_alpha, task_type, or None if empty.
        # Side effects: reads file; emits warnings.
        if not self.config.local_dir:
            return None

        meta_path = os.path.join(self.config.local_dir, "lora_train_meta.json")
        if not os.path.exists(meta_path):
            return None

        import json

        try:
            with open(meta_path, encoding="utf-8") as f:
                lora_meta = json.load(f)
        except Exception as e:
            warnings.warn(f"Failed to read LoRA metadata from {meta_path}: {e}", stacklevel=2)
            return None

        result = {}
        if "r" in lora_meta:
            try:
                result["r"] = int(lora_meta["r"])
            except (TypeError, ValueError):
                warnings.warn(f"Invalid LoRA rank in {meta_path}: {lora_meta['r']}", stacklevel=2)

        if "lora_alpha" in lora_meta:
            try:
                result["lora_alpha"] = int(lora_meta["lora_alpha"])
            except (TypeError, ValueError):
                warnings.warn(f"Invalid lora_alpha in {meta_path}: {lora_meta['lora_alpha']}", stacklevel=2)

        if "task_type" in lora_meta:
            task_type = lora_meta["task_type"]
            if task_type is None:
                pass
            elif isinstance(task_type, str):
                result["task_type"] = task_type
            else:
                warnings.warn(f"Invalid task_type in {meta_path}: {task_type}", stacklevel=2)

        return result if len(result) > 0 else None

    def save_lora_adapter(self, state_dict: dict[str, torch.Tensor]):
        # [FUNCTION DOC]
        # Behavior: extracts all state_dict entries whose names contain "lora_" (removing them from state_dict), renames
        # ".default.weight" to ".weight", collects target module names (the third-to-last dotted component), infers rank
        # as the minimum dimension of the last LoRA tensor, and overrides rank/alpha/task_type with metadata when present
        # (positive rank only; warns on rank mismatch). Warns if alpha ends up 0. Builds a peft LoraConfig dict, converts
        # enum values of task_type/peft_type to plain values and target_modules to a list, writes adapter_config.json and
        # adapter_model.safetensors under target_dir/lora_adapter. Finally renames remaining base keys by stripping
        # "base_model.model." and mapping ".base_layer.weight"/".base_layer.bias" to ".weight"/".bias".
        # Inputs: state_dict - mapping name -> tensor (mutated in place).
        # Outputs: path of the adapter directory, or None when no LoRA params exist (state_dict untouched in that case).
        # Side effects: mutates state_dict; writes files; emits warnings.
        """
        Save lora adapter to safetensors.

        Returns:
            lora_path: str, the path to the lora adapter. None if no lora adapter found.

        Note:
            This function change the 'state_dict' in place.
        """
        lora_params_names = [name for name in state_dict.keys() if "lora_" in name]

        if len(lora_params_names) == 0:
            return None

        import json
        from typing import OrderedDict

        import peft
        from safetensors.torch import save_file

        lora_params = OrderedDict()
        target_modules = set()
        lora_key = None

        for name in lora_params_names:
            lora_key = name.replace(".default.weight", ".weight")
            target_modules.add(lora_key.split(".")[-3])
            lora_params[lora_key] = state_dict.pop(name)

        inferred_lora_rank = min(lora_params[lora_key].shape[0], lora_params[lora_key].shape[1])
        lora_meta = self._load_lora_train_meta()

        lora_rank = inferred_lora_rank
        lora_alpha = 0
        task_type = None

        if lora_meta is not None:
            meta_rank = lora_meta.get("r")
            if meta_rank is not None and meta_rank > 0:
                if meta_rank != inferred_lora_rank:
                    warnings.warn(
                        f"LoRA rank mismatch between metadata ({meta_rank}) and adapter weights "
                        f"({inferred_lora_rank}); using metadata rank.",
                        stacklevel=2,
                    )
                lora_rank = meta_rank

            meta_alpha = lora_meta.get("lora_alpha")
            if meta_alpha is not None:
                lora_alpha = meta_alpha

            meta_task_type = lora_meta.get("task_type")
            if meta_task_type is not None:
                task_type = meta_task_type

        if lora_alpha == 0:
            warnings.warn(
                "LoRA alpha metadata is missing or equals 0; falling back to lora_alpha=0. "
                "Please verify checkpoint LoRA metadata (lora_train_meta.json).",
                stacklevel=2,
            )

        peft_dict = {
            "r": lora_rank,
            "lora_alpha": lora_alpha,
            "target_modules": list(target_modules),
        }
        if task_type is not None:
            peft_dict["task_type"] = task_type
        peft_config = peft.LoraConfig(**peft_dict).to_dict()
        peft_config["task_type"] = (
            peft_config["task_type"].value
            if hasattr(peft_config["task_type"], "value")
            else (peft_config["task_type"] or None)
        )
        peft_config["peft_type"] = (
            peft_config["peft_type"].value
            if hasattr(peft_config["peft_type"], "value")
            else (peft_config["peft_type"] or None)
        )
        peft_config["target_modules"] = list(peft_config["target_modules"])

        lora_path = os.path.join(self.config.target_dir, "lora_adapter")
        os.makedirs(lora_path, exist_ok=True)
        with open(os.path.join(lora_path, "adapter_config.json"), "w", encoding="utf-8") as f:
            json.dump(peft_config, f, ensure_ascii=False, indent=4)
        save_file(lora_params, os.path.join(lora_path, "adapter_model.safetensors"))

        for name in list(state_dict.keys()):
            key = (
                name.replace("base_model.model.", "")
                .replace(".base_layer.weight", ".weight")
                .replace(".base_layer.bias", ".bias")
            )
            state_dict[key] = state_dict.pop(name)

        return lora_path

    def save_hf_model_and_tokenizer(self, state_dict: dict[str, torch.Tensor]):
        # [FUNCTION DOC]
        # Behavior: creates the model with empty weights in bfloat16 from the HF config using the chosen auto class,
        # materializes it on CPU, patches its generation config, saves any LoRA adapter (stripping LoRA keys), drops tied
        # target keys from the state dict, saves the model with the provided state_dict to target_dir, then saves the
        # processor and tokenizer (if available) and validates the output directory.
        # Inputs: state_dict - merged full state dict (mutated).
        # Exceptions: NotImplementedError from auto class selection; RuntimeError from output validation.
        # Side effects: writes model/tokenizer/processor/adapter files; prints progress.
        auto_model_class = self.get_transformers_auto_model_class()
        with init_empty_weights():
            model = auto_model_class.from_config(
                self.model_config, torch_dtype=torch.bfloat16, trust_remote_code=self.config.trust_remote_code
            )
        model.to_empty(device="cpu")
        model = self.patch_model_generation_config(model)

        lora_path = self.save_lora_adapter(state_dict)
        if lora_path:
            print(f"Saving lora adapter to {lora_path}")

        drop_tied_target_keys(state_dict, model, self.model_config)

        print(f"Saving model to {self.config.target_dir}")
        model.save_pretrained(self.config.target_dir, state_dict=state_dict)
        del state_dict
        del model

        processor = hf_processor(self.hf_model_config_path, trust_remote_code=self.config.trust_remote_code)
        tokenizer = hf_tokenizer(self.hf_model_config_path, trust_remote_code=self.config.trust_remote_code)
        if processor is not None:
            print(f"Saving processor to {self.config.target_dir}")
            processor.save_pretrained(self.config.target_dir)
        if tokenizer is not None:
            print(f"Saving tokenizer to {self.config.target_dir}")
            tokenizer.save_pretrained(self.config.target_dir)

        validate_hf_model_output(self.config.target_dir)

    def upload_to_huggingface(self):
        # [FUNCTION DOC]
        # Behavior: creates (exist_ok) the Hub repo hf_upload_path with the configured privacy, then uploads target_dir as
        # a model repo.
        # Exceptions: on repo creation - PermissionError for 401, RepositoryNotFoundError for 404, ConnectionError for other
        # HTTP errors or network failures; on upload - PermissionError for 401, RuntimeError for other HTTP errors,
        # ConnectionError on network failures, FileNotFoundError on OSError, RuntimeError for anything else.
        # Side effects: network calls to the HuggingFace Hub.
        import requests
        from huggingface_hub import HfApi
        from huggingface_hub.utils import HfHubHTTPError, RepositoryNotFoundError

        api = HfApi()
        try:
            # Attempt to create repository
            api.create_repo(repo_id=self.config.hf_upload_path, private=self.config.private, exist_ok=True)
        except HfHubHTTPError as e:
            # Handle authentication/API errors
            if e.response.status_code == 401:
                raise PermissionError(
                    "Hugging Face authentication failed. Verify your token is valid and has write permissions."
                ) from e
            elif e.response.status_code == 404:
                raise RepositoryNotFoundError(f"Repository path not found: {self.config.hf_upload_path}") from e
            else:
                raise ConnectionError(f"Failed to create repository ({e.response.status_code}): {e}") from e
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError("Network connection failed. Check your internet connection.") from e

        try:
            # Attempt folder upload
            api.upload_folder(folder_path=self.config.target_dir, repo_id=self.config.hf_upload_path, repo_type="model")
        except HfHubHTTPError as e:
            if e.response.status_code == 401:
                raise PermissionError("Authentication failed during upload. Token may have expired.") from e
            else:
                raise RuntimeError(f"Upload failed ({e.response.status_code}): {e}") from e
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError("Network interruption during upload. Try again with stable connection.") from e
        except OSError as e:
            raise FileNotFoundError(f"Local folder error: {self.config.target_dir} - {str(e)}") from e
        except Exception as e:
            raise RuntimeError(f"Unexpected error during upload: {str(e)}") from e

    @abstractmethod
    def merge_and_save(self):
        # [FUNCTION DOC]
        # Abstract: subclasses load backend shards, merge, and either save (merge) or compare (test).
        # Exceptions: raises NotImplementedError in the base.
        raise NotImplementedError("Subclasses should implement this method")

    @abstractmethod
    def cleanup(self):
        # [FUNCTION DOC]
        # Abstract: subclasses release resources (e.g. distributed process groups).
        # Exceptions: raises NotImplementedError in the base.
        raise NotImplementedError("Subclasses should implement this method to clean up resources if needed")
