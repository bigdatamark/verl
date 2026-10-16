# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
# Single-GPU closed-loop script (see module docstring) applying a sparse weight delta to a real sglang engine via
# verl's custom delta weight loader and then running a dense verify flush that must report zero mismatches.
# At import time prepends VERL_PATH (default a developer-specific path) to sys.path and reads MODEL_KIND (qwen2 or nemotron_h).
# KNOWN ISSUE: the default VERL_PATH "/home/changyi/verl_ab_pre" is developer-specific.

"""Closed-loop delta apply test against a REAL sglang engine on tiny models.

randomize -> perturb -> sparse delta flush -> engine.update_weights_from_tensor
(the production ``load_format=<delta loader FQN>`` path) -> idempotence verify
flush asserting ZERO mismatched elements. This exercises everything the
``_FakeModel`` CPU tests cannot: sglang's name remapping, fused-tensor slicing
(qkv / gate_up), TP shard loaders, and multi-stage transform loaders (mamba's
``A = -exp(A_log)``). ``MODEL_KIND=nemotron_h`` needs mamba-ssm/causal-conv1d.

Run on 1 GPU::

    MODEL_KIND=qwen2 python tests/special_distributed/test_sglang_delta_loop.py
    MODEL_KIND=nemotron_h python tests/special_distributed/test_sglang_delta_loop.py
"""

import json
import os
import pathlib
import sys

import torch

sys.path.insert(0, os.environ.get("VERL_PATH", "/home/changyi/verl_ab_pre"))

MODEL_KIND = os.environ.get("MODEL_KIND", "qwen2")


def _build_tiny_hf_dir() -> str:
    # [FUNCTION DOC]
    # HELPER: Ensures a tiny random bf16 checkpoint exists at /tmp/tiny_<MODEL_KIND>_delta_loop_v2 and returns the path.
    # Behavior: builds a small qwen2 config (2 layers, hidden 256, real Qwen vocab size) or, for nemotron_h, first stubs
    # transformers hub-kernel loaders to no-ops and builds a small hybrid Mamba config; raises ValueError for other kinds.
    # If model.safetensors is absent, instantiates and saves the model and saves the Qwen2.5-7B-Instruct tokenizer into the directory.
    # Side effects: monkey-patches transformers (nemotron_h), writes files, may download a tokenizer.
    # KNOWN ISSUE: the cache check is existence-only, so a stale directory from a different config is reused.
    from transformers import AutoConfig

    cfg_dir = f"/tmp/tiny_{MODEL_KIND}_delta_loop_v2"
    if MODEL_KIND == "qwen2":
        hf_config = AutoConfig.for_model(
            "qwen2",
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=151936,
            max_position_embeddings=1024,
            tie_word_embeddings=False,
        )
        hf_config.architectures = ["Qwen2ForCausalLM"]
    elif MODEL_KIND == "nemotron_h":
        import transformers.integrations.hub_kernels as _hk

        _hk.lazy_load_kernel = lambda *a, **k: None
        import transformers.models.nemotron_h.modeling_nemotron_h as _mnh

        for _sym in ("lazy_load_kernel", "get_kernel"):
            if hasattr(_mnh, _sym):
                setattr(_mnh, _sym, lambda *a, **k: None)
        hf_config = AutoConfig.for_model(
            "nemotron_h",
            hidden_size=256,
            intermediate_size=512,
            num_hidden_layers=4,
            hybrid_override_pattern="M*M-",
            num_attention_heads=4,
            num_key_value_heads=2,
            attention_head_dim=64,
            ssm_state_size=64,
            conv_kernel=4,
            expand=2,
            mamba_num_heads=8,
            mamba_head_dim=64,
            n_groups=2,
            chunk_size=128,
            vocab_size=151936,
            max_position_embeddings=1024,
            tie_word_embeddings=False,
        )
        hf_config.architectures = ["NemotronHForCausalLM"]
    else:
        raise ValueError(MODEL_KIND)

    if not pathlib.Path(cfg_dir, "model.safetensors").exists():
        from transformers import AutoModelForCausalLM, AutoTokenizer

        m = AutoModelForCausalLM.from_config(hf_config).to(torch.bfloat16)
        m.save_pretrained(cfg_dir, safe_serialization=True)
        del m
        # the engine needs a real tokenizer; any cached one with a matching
        # vocab budget works -- the weights are random anyway.
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
        tok.save_pretrained(cfg_dir)
    return cfg_dir


def _load_named(cfg_dir: str) -> dict[str, torch.Tensor]:
    # [FUNCTION DOC]
    # HELPER: Loads model.safetensors from the directory and returns a mapping name to bf16 CUDA tensor.
    from safetensors.torch import load_file

    return {k: v.to(torch.bfloat16).cuda() for k, v in load_file(f"{cfg_dir}/model.safetensors").items()}


def _spec_tensor(spec: dict) -> torch.Tensor:
    # [FUNCTION DOC]
    # HELPER: Serializes a mapping to JSON and returns its UTF-8 bytes as a uint8 tensor.
    return torch.frombuffer(bytearray(json.dumps(spec).encode()), dtype=torch.uint8)


def _sparse_flush(old: dict, new: dict):
    # [FUNCTION DOC]
    # HELPER: Builds the "indices" delta wire payload (see docstring) from old and new name-to-tensor mappings.
    # Behavior: for each new tensor computes changed flat indices/values relative to old; skips unchanged tensors; records a
    # DeltaParam with byte offsets into a positions buffer (4 bytes per int32 index) and element offsets into a values
    # buffer. Concatenates int32 indices viewed as bytes and the values, computes a checksum, and builds a spec
    # (encoding "indices", params as dicts, checksum).
    # Outputs: list of (name, tensor) pairs: "__delta_spec__", "__positions__", "__values__".
    # Exceptions: errors from concatenating empty lists if no tensor changed.
    """Encode changed elements exactly like the sender's indices wire."""
    from verl.checkpoint_engine.delta_sync import DeltaParam, checksum
    from verl.checkpoint_engine.delta_sync.sparse_gather import shard_delta_indices

    params, idx_pieces, val_pieces = [], [], []
    pos_off = val_off = 0
    for name, t_new in new.items():
        lidx, lval = shard_delta_indices(t_new.reshape(-1), old[name].reshape(-1), 0)
        nnz = int(lidx.numel())
        if nnz == 0:
            continue
        params.append(
            DeltaParam(
                name=name,
                dtype="bfloat16",
                shape=list(t_new.shape),
                pos_start=pos_off,
                pos_end=pos_off + nnz * 4,
                pos_width=4,
                val_start=val_off,
                val_end=val_off + nnz,
            )
        )
        idx_pieces.append(lidx.to(torch.int32))
        val_pieces.append(lval)
        pos_off += nnz * 4
        val_off += nnz
    positions = torch.cat(idx_pieces).contiguous().view(torch.uint8)
    values = torch.cat(val_pieces)
    spec = {
        "encoding": "indices",
        "params": [vars(p) for p in params],
        "checksum": int(checksum(positions, values)),
    }
    return [("__delta_spec__", _spec_tensor(spec)), ("__positions__", positions), ("__values__", values)]


def _verify_flush(expected: dict):
    # [FUNCTION DOC]
    # HELPER: Builds a dense "verify" payload containing every expected tensor's full flattened values with value offsets,
    # no positions, flags verify and is_last, and a checksum over empty positions and the values.
    # Outputs: list of ("__delta_spec__", spec tensor) and ("__values__", values).
    from verl.checkpoint_engine.delta_sync import DeltaParam, checksum

    params, pieces, val_off = [], [], 0
    for name, t in expected.items():
        flat = t.contiguous().reshape(-1)
        params.append(
            DeltaParam(
                name=name,
                dtype="bfloat16",
                shape=list(t.shape),
                pos_start=0,
                pos_end=0,
                pos_width=4,
                val_start=val_off,
                val_end=val_off + flat.numel(),
            )
        )
        pieces.append(flat)
        val_off += flat.numel()
    values = torch.cat(pieces)
    spec = {
        "encoding": "dense",
        "verify": True,
        "is_last": True,
        "params": [vars(p) for p in params],
        "checksum": int(checksum(torch.empty(0, dtype=torch.uint8), values)),
    }
    return [("__delta_spec__", _spec_tensor(spec)), ("__values__", values)]


def main():
    # [FUNCTION DOC]
    # Behavior: builds/loads the tiny checkpoint, starts an sglang Engine with the delta loader registered (no CUDA graphs or
    # radix cache, seed 1234), perturbs ~1% of elements of each tensor (seed 11, noise std 0.05), sends the sparse delta from
    # the original to the perturbed weights, then the verify flush against the perturbed weights (the loader fails on any
    # mismatch), prints a pass message and shuts the engine down.
    # Side effects: GPU engine startup, prints.
    import sglang as sgl

    from verl.workers.rollout.sglang_rollout.delta_loader import LOADER_FQN

    cfg_dir = _build_tiny_hf_dir()
    named = _load_named(cfg_dir)

    engine = sgl.Engine(
        model_path=cfg_dir,
        mem_fraction_static=0.35,
        disable_cuda_graph=True,
        disable_radix_cache=True,
        trust_remote_code=True,
        custom_weight_loader=[LOADER_FQN],
        random_seed=1234,
    )

    torch.manual_seed(11)
    perturbed = {}
    for name, t in named.items():
        p = t.clone()
        flat = p.reshape(-1)
        k = max(1, flat.numel() // 100)  # ~1% of elements per tensor
        pos = torch.randperm(flat.numel(), device=flat.device)[:k]
        flat[pos] += torch.randn(k, device=flat.device, dtype=flat.dtype) * 0.05
        perturbed[name] = p

    # sparse delta: server state (checkpoint) -> perturbed
    engine.update_weights_from_tensor(_sparse_flush(named, perturbed), load_format=LOADER_FQN, flush_cache=False)
    # idempotence verify against the expected end state: must report 0 mismatches
    engine.update_weights_from_tensor(_verify_flush(perturbed), load_format=LOADER_FQN, flush_cache=False)

    print(f"DELTA-LOOP [{MODEL_KIND}]: sparse apply + verify sweep PASSED")
    engine.shutdown()


if __name__ == "__main__":
    main()
