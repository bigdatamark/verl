# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
# Unit tests for verl.models.mcore.model_forward_fused (imported as mff): selection between Megatron's native
# output_processor hook mode and the legacy forward-replacement mode in patch/unpatch_fused_forward, the model-engine
# fused forward argument plumbing, traversal of Megatron Bridge wrappers (DDP -> Float16Module -> GPTModel), and
# fused_output_processor behavior. Requires megatron-core; runs on CPU via uninitialized modules and monkeypatches.

import inspect
from types import MethodType, SimpleNamespace

import pytest
import torch
from megatron.core.distributed.distributed_data_parallel import DistributedDataParallel
from megatron.core.models.gpt.gpt_model import GPTModel
from megatron.core.transformer.module import Float16Module

from verl.models.mcore import model_forward_fused as mff


def _new_uninitialized_model(model_cls=GPTModel):
    # [FUNCTION DOC]
    # HELPER: Creates an instance of model_cls (default GPTModel) without running its constructor, only initializing the
    # base torch Module state.
    model = object.__new__(model_cls)
    torch.nn.Module.__init__(model)
    return model


def test_mcore_gpt_forward_has_native_output_processor_contract():
    # [FUNCTION DOC]
    # Asserts the installed GPTModel.forward accepts output_processor and output_processor_context parameters.
    parameters = inspect.signature(GPTModel.forward).parameters
    assert {"output_processor", "output_processor_context"}.issubset(parameters)


def test_native_hook_selection_preserves_forward_and_is_idempotent(monkeypatch):
    # [FUNCTION DOC]
    # Behavior: counts calls to mff.inspect.signature, then patches twice, unpatches twice, patches again on a native
    # GPTModel; asserts the signature was inspected only once (cached), mode attribute is hook mode, forward is the
    # original method, and no forward_backup attribute was created.
    model = _new_uninitialized_model()
    original_forward = model.forward.__func__
    signature_calls = 0
    original_signature = inspect.signature

    def counted_signature(callable_):
        # [FUNCTION DOC]
        # HELPER: Increments the enclosing counter and delegates to the real signature function.
        nonlocal signature_calls
        signature_calls += 1
        return original_signature(callable_)

    monkeypatch.setattr(mff.inspect, "signature", counted_signature)

    mff.patch_fused_forward(model)
    mff.patch_fused_forward(model)
    mff.unpatch_fused_forward(model)
    mff.unpatch_fused_forward(model)
    mff.patch_fused_forward(model)

    assert signature_calls == 1
    assert getattr(model, mff._FUSED_FORWARD_MODE_ATTR) == mff._HOOK_MODE
    assert model.forward.__func__ is original_forward
    assert not hasattr(model, "forward_backup")


def test_legacy_fallback_patch_and_unpatch_are_idempotent():
    # [FUNCTION DOC]
    # Behavior: with a GPTModel subclass whose forward lacks output_processor, asserts patching sets legacy mode, swaps
    # forward to the fused GPT forward with a forward_backup kept stable across repeated patches; repeated unpatching
    # restores the original forward and removes the backup; and a second patch/unpatch cycle works.
    class LegacyGPTModel(GPTModel):
        # [CLASS DOC]
        # HELPER: GPTModel subclass with an old-style forward signature.
        def forward(self, input_ids=None):
            # [FUNCTION DOC]
            # HELPER: Returns input_ids unchanged.
            return input_ids

    model = _new_uninitialized_model(LegacyGPTModel)
    original_forward = model.forward.__func__

    mff.patch_fused_forward(model)
    backup = model.forward_backup
    mff.patch_fused_forward(model)

    assert getattr(model, mff._FUSED_FORWARD_MODE_ATTR) == mff._LEGACY_MODE
    assert model.forward.__func__ is mff._fused_GPTModel_forward
    assert model.forward_backup is backup

    mff.unpatch_fused_forward(model)
    mff.unpatch_fused_forward(model)
    assert model.forward.__func__ is original_forward
    assert not hasattr(model, "forward_backup")

    mff.patch_fused_forward(model)
    assert model.forward.__func__ is mff._fused_GPTModel_forward
    mff.unpatch_fused_forward(model)
    assert model.forward.__func__ is original_forward


def test_engine_hook_context_and_current_thd_arguments(monkeypatch):
    # [FUNCTION DOC]
    # Behavior: replaces preprocess_thd_engine with a recorder and calls the engine fused forward for a hook-mode fake
    # model (temperature 0.7, cp layout "dualpipev"): asserts the output passes through, no temperature kwarg is sent,
    # output_processor is fused_output_processor with a context carrying temperature 0.7, and two preprocess calls
    # (input ids, then labels with need_roll) received fp8 padding True (config fp8 set), min_local_rows 64 (CSA window
    # for the dsv4_hybrid attention variant), no length bucket, the cp layout and no local cp size. For a legacy-mode
    # model asserts temperature 0.5 is passed directly and no processor kwargs are sent.
    input_ids = torch.tensor([[1, 2, 3]])
    labels = torch.tensor([[2, 3, 4]])
    preprocess_calls = []

    def fake_preprocess(value, **kwargs):
        # [FUNCTION DOC]
        # HELPER: Records kwargs and returns (value, "packed", None).
        preprocess_calls.append(kwargs)
        return value, "packed", None

    monkeypatch.setattr(mff, "preprocess_thd_engine", fake_preprocess)

    class Model:
        # [CLASS DOC]
        # HELPER: Callable fake model with pre_process True, post_process False and an fp8/CSA config; records call kwargs.
        pre_process = True
        post_process = False
        config = SimpleNamespace(
            fp8="hybrid",
            csa_window_size=64,
            experimental_attention_variant="dsv4_hybrid",
        )

        def __init__(self, mode):
            # [FUNCTION DOC]
            # HELPER: Sets the fused-forward mode attribute and an empty calls list.
            setattr(self, mff._FUSED_FORWARD_MODE_ATTR, mode)
            self.calls = []

        def __call__(self, **kwargs):
            # [FUNCTION DOC]
            # HELPER: Records kwargs and returns an object with log_probs [1.0] and entropy [2.0].
            self.calls.append(kwargs)
            return SimpleNamespace(log_probs=torch.tensor([1.0]), entropy=torch.tensor([2.0]))

    hook_model = Model(mff._HOOK_MODE)
    output = mff.fused_forward_model_engine()(hook_model, input_ids, labels, {}, 0.7, True, 0, "dualpipev")

    hook_kwargs = hook_model.calls[-1]
    assert output.log_probs.tolist() == [1.0]
    assert "temperature" not in hook_kwargs
    assert hook_kwargs["output_processor"] is mff.fused_output_processor
    assert hook_kwargs["output_processor_context"].temperature == pytest.approx(0.7)
    assert preprocess_calls == [
        {
            "pre_process": True,
            "use_fp8_padding": True,
            "min_local_rows": 64,
            "pad_to_length_bucket": None,
            "cp_layout": "dualpipev",
            "local_cp_size": None,
        },
        {
            "pre_process": True,
            "need_roll": True,
            "use_fp8_padding": True,
            "min_local_rows": 64,
            "pad_to_length_bucket": None,
            "cp_layout": "dualpipev",
            "local_cp_size": None,
        },
    ]

    legacy_model = Model(mff._LEGACY_MODE)
    mff.fused_forward_model_engine()(legacy_model, input_ids, labels, {}, 0.5, True, 0)
    legacy_kwargs = legacy_model.calls[-1]
    assert legacy_kwargs["temperature"] == pytest.approx(0.5)
    assert "output_processor" not in legacy_kwargs
    assert "output_processor_context" not in legacy_kwargs


class _OutputLayer(torch.nn.Module):
    # [CLASS DOC]
    # HELPER: Module holding a 4 x 3 all-ones weight parameter.
    def __init__(self):
        # [FUNCTION DOC]
        # HELPER: Creates the weight parameter.
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(4, 3))


class _Decoder(torch.nn.Module):
    # [CLASS DOC]
    # HELPER: Identity decoder returning hidden_states.
    def forward(self, *, hidden_states, **_kwargs):
        # [FUNCTION DOC]
        # HELPER: Returns the hidden_states keyword argument.
        return hidden_states


def test_megatron_bridge_wrapper_chain_reaches_native_hook(monkeypatch):
    # [FUNCTION DOC]
    # Behavior: hand-assembles an uninitialized GPTModel (minimal config, untied output layer, identity decoder, stubbed
    # _preprocess returning fixed hidden states) wrapped in Float16Module and DistributedDataParallel; fakes
    # linear_cross_entropy and TP group. Patches the DDP wrapper and calls it with the fused output processor and a 0.7
    # temperature context. Asserts the inner forward is unchanged (hook mode, no backup), outputs come from the fake
    # kernel, and the kernel saw temperature 0.7 and the output layer weight.
    """Exercise the Float16Module -> DDP stack built by Megatron Bridge."""
    model = _new_uninitialized_model()
    model.config = SimpleNamespace(
        fine_grained_activation_offloading=False,
        moe_paged_stash=False,
        mtp_num_layers=0,
        use_mup=False,
        sequence_parallel=False,
    )
    model.share_embeddings_and_output_weights = False
    model.mtp_process = False
    model.post_process = True
    model.output_layer = _OutputLayer()
    model.decoder = _Decoder()
    model.pg_collection = SimpleNamespace(tp=None, cp=None)
    hidden_states = torch.ones(2, 1, 3)

    def preprocess(_self, **_kwargs):
        # [FUNCTION DOC]
        # HELPER: Returns the fixed hidden states followed by five None values.
        return hidden_states, None, None, None, None, None

    model._preprocess = MethodType(preprocess, model)

    mixed_precision_wrapper = object.__new__(Float16Module)
    torch.nn.Module.__init__(mixed_precision_wrapper)
    mixed_precision_wrapper.module = model
    mixed_precision_wrapper.pg_collection = SimpleNamespace(pp=SimpleNamespace(rank=lambda: 0, size=lambda: 1))
    mixed_precision_wrapper.vp_stage = None
    mixed_precision_wrapper.vp_size = None
    mixed_precision_wrapper.float16_convertor = lambda value: value

    ddp_wrapper = object.__new__(DistributedDataParallel)
    torch.nn.Module.__init__(ddp_wrapper)
    ddp_wrapper.module = mixed_precision_wrapper

    seen = {}

    def fake_linear_cross_entropy(hidden, weight, labels, temperature, reduction, group):
        # [FUNCTION DOC]
        # HELPER: Records all arguments into seen and returns (ones of 2, twos of 2).
        seen.update(
            hidden=hidden,
            weight=weight,
            labels=labels,
            temperature=temperature,
            reduction=reduction,
            group=group,
        )
        return torch.ones(2), torch.ones(2) * 2

    monkeypatch.setattr(mff, "linear_cross_entropy", fake_linear_cross_entropy)
    monkeypatch.setattr(mff.parallel_state, "get_tensor_model_parallel_group", lambda: None)

    original_forward = model.forward.__func__
    mff.patch_fused_forward(ddp_wrapper)
    output = ddp_wrapper(
        input_ids=torch.tensor([[1, 2]]),
        position_ids=torch.tensor([[0, 1]]),
        attention_mask=None,
        labels=torch.tensor([1, 2]),
        output_processor=mff.fused_output_processor,
        output_processor_context=mff.FusedOutputProcessorContext(temperature=0.7),
        fp32_output=False,
    )

    assert model.forward.__func__ is original_forward
    assert not hasattr(model, "forward_backup")
    assert output.log_probs.tolist() == [1.0, 1.0]
    assert output.entropy.tolist() == [2.0, 2.0]
    assert seen["temperature"] == pytest.approx(0.7)
    assert seen["weight"] is model.output_layer.weight


def test_output_processor_preserves_config_logger_payload(monkeypatch):
    # [FUNCTION DOC]
    # Behavior: with config logging enabled for the given config, asserts fused_output_processor logs a payload with keys
    # input_ids, position_ids, attention_mask, decoder_input, logprobs, entropy (in that order and identity-preserved),
    # the same config, and prefix "input_and_logits".
    input_ids = object()
    position_ids = object()
    attention_mask = object()
    decoder_input = object()
    log_probs = torch.tensor([1.0])
    entropy = torch.tensor([2.0])
    config = SimpleNamespace(sequence_parallel=False)
    logged = {}

    def fake_linear_cross_entropy(*_args, **_kwargs):
        # [FUNCTION DOC]
        # HELPER: Returns the preset log_probs and entropy tensors.
        return log_probs, entropy

    def fake_log_config_to_disk(config_arg, payload, prefix):
        # [FUNCTION DOC]
        # HELPER: Records config, payload and prefix into logged.
        logged.update(config=config_arg, payload=payload, prefix=prefix)

    monkeypatch.setattr(mff, "linear_cross_entropy", fake_linear_cross_entropy)
    monkeypatch.setattr(mff.parallel_state, "get_tensor_model_parallel_group", lambda: None)
    monkeypatch.setattr(mff, "has_config_logger_enabled", lambda config_arg: config_arg is config)
    monkeypatch.setattr(mff, "log_config_to_disk", fake_log_config_to_disk)

    mff.fused_output_processor(
        hidden_states=torch.tensor([[1.0]]),
        output_layer=SimpleNamespace(weight=torch.tensor([[1.0]])),
        output_weight=None,
        labels=torch.tensor([0]),
        context=mff.FusedOutputProcessorContext(temperature=1.0),
        config=config,
        input_ids=input_ids,
        position_ids=position_ids,
        attention_mask=attention_mask,
        decoder_input=decoder_input,
    )

    assert list(logged["payload"]) == [
        "input_ids",
        "position_ids",
        "attention_mask",
        "decoder_input",
        "logprobs",
        "entropy",
    ]
    assert logged["payload"]["input_ids"] is input_ids
    assert logged["payload"]["position_ids"] is position_ids
    assert logged["payload"]["attention_mask"] is attention_mask
    assert logged["payload"]["decoder_input"] is decoder_input
    assert logged["payload"]["logprobs"] is log_probs
    assert logged["payload"]["entropy"] is entropy
    assert logged["config"] is config
    assert logged["prefix"] == "input_and_logits"


@pytest.mark.parametrize(
    ("sequence_parallel", "use_tied_weight"),
    [(True, True), (False, False)],
)
def test_output_processor_gathers_before_kernel_and_resolves_weight(monkeypatch, sequence_parallel, use_tied_weight):
    # [FUNCTION DOC]
    # Behavior: asserts that with sequence parallel the hidden states are gathered before the kernel (event order), that
    # the kernel uses the tied output_weight when given else the output layer weight, receives labels and temperature
    # 0.9, and that outputs are the kernel's log probs and entropy.
    # Inputs: sequence_parallel, use_tied_weight parametrized jointly.
    hidden_states = torch.tensor([[1.0]])
    gathered_hidden_states = torch.tensor([[3.0]])
    tied_weight = torch.tensor([[5.0]])
    output_layer_weight = torch.tensor([[7.0]])
    labels = torch.tensor([0])
    events = []
    seen = {}

    def fake_gather(value):
        # [FUNCTION DOC]
        # HELPER: Records "gather", asserts it got the original hidden states, returns the gathered tensor.
        events.append("gather")
        assert value is hidden_states
        return gathered_hidden_states

    def fake_linear_cross_entropy(hidden, weight, labels_arg, temperature, reduction, group):
        # [FUNCTION DOC]
        # HELPER: Records the event and inputs, returns tensors [11.0] and [13.0].
        events.append("linear_cross_entropy")
        seen.update(hidden=hidden, weight=weight, labels=labels_arg, temperature=temperature)
        return torch.tensor([11.0]), torch.tensor([13.0])

    monkeypatch.setattr(mff, "gather_from_sequence_parallel_region", fake_gather)
    monkeypatch.setattr(mff, "linear_cross_entropy", fake_linear_cross_entropy)
    monkeypatch.setattr(mff.parallel_state, "get_tensor_model_parallel_group", lambda: None)

    output_weight = tied_weight if use_tied_weight else None
    output = mff.fused_output_processor(
        hidden_states=hidden_states,
        output_layer=SimpleNamespace(weight=output_layer_weight),
        output_weight=output_weight,
        labels=labels,
        context=mff.FusedOutputProcessorContext(temperature=0.9),
        config=SimpleNamespace(sequence_parallel=sequence_parallel),
    )

    expected_hidden = gathered_hidden_states if sequence_parallel else hidden_states
    expected_weight = tied_weight if use_tied_weight else output_layer_weight
    assert events == (["gather"] if sequence_parallel else []) + ["linear_cross_entropy"]
    assert seen["hidden"] is expected_hidden
    assert seen["weight"] is expected_weight
    assert seen["labels"] is labels
    assert seen["temperature"] == pytest.approx(0.9)
    assert output.log_probs.tolist() == [11.0]
    assert output.entropy.tolist() == [13.0]
