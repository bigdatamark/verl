# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# FSDP-Turbo variant of the FSDP LM-head engine: delegates parallel-state setup, module wrapping, context parallelism and gradient clipping to the external fsdp_turbo library. Registered as backend "fsdp_turbo" for language models on cuda and npu.

import torch

from verl.utils.fsdp_utils import fsdp2_load_full_state_dict

from ..base import EngineRegistry
from .transformer_impl import FSDPEngineWithLMHead


@EngineRegistry.register(model_type="language_model", backend="fsdp_turbo", device=["cuda", "npu"])
class FSDPTurboEngineWithLMHead(FSDPEngineWithLMHead):
    # [CLASS DOC]
    # Engine subclassing FSDPEngineWithLMHead that wraps the HF model with FSDPTurbo and uses FSDP-Turbo's parallel state for data-parallel and Ulysses (CP) groups.
    # Attributes added: fsdp_turbo_config (FSDPTurboConfig dataclass), _parallel_state (fsdp_turbo parallel state object).
    def _init_device_mesh(self):
        # [FUNCTION DOC]
        # HELPER: Runs the parent device-mesh init and then initializes FSDP-Turbo parallel state.
        super()._init_device_mesh()
        self._init_parallel_state()

    def _init_parallel_state(self):
        # [FUNCTION DOC]
        # HELPER: Builds FSDPTurboConfig from engine_config.turbo_config, copies engine_config.offload_policy into the fsdp plan's cpu_offload, initializes the global FSDP-Turbo parallel state and stores it; if Turbo's ulysses group is enabled, processes the Ulysses config.
        # Side effects: global parallel-state initialization (process groups); lazy imports fsdp_turbo.
        from fsdp_turbo.distributed.parallel_state import get_parallel_state, init_parallel_state
        from fsdp_turbo.fsdp_turbo_config import FSDPTurboConfig, _dict_to_dataclass

        self.fsdp_turbo_config = _dict_to_dataclass(FSDPTurboConfig, self.engine_config.turbo_config)
        self.fsdp_turbo_config.distributed.fsdp_plan.cpu_offload = self.engine_config.offload_policy
        init_parallel_state(self.fsdp_turbo_config)
        self._parallel_state = get_parallel_state()
        if self._is_ulysses_enabled():
            self._process_ulysses_config()

    def _build_module(self):
        # [FUNCTION DOC]
        # HELPER: Builds the HF module via the parent with ulysses_sequence_parallel_size temporarily forced to 1 (so verl's VLM monkey patch does not split sequences before Turbo's CP split), restoring it afterwards even on error. Returns the module.
        # Do not let verl's Qwen VLM monkey patch slice the text model before
        # FSDP-Turbo's post-fusion model patch does the CP split.
        cp_size = self.ulysses_sequence_parallel_size
        self.ulysses_sequence_parallel_size = 1
        try:
            return super()._build_module()
        finally:
            self.ulysses_sequence_parallel_size = cp_size

    def _build_fsdp_module(self, module):
        # [FUNCTION DOC]
        # HELPER: Wraps a module with FSDPTurbo and loads the original full weights into it.
        # Behavior: captures the module's full state dict, wraps it, and if offload_policy or forward_only is configured disables manual param/optimizer offload, enables the FSDP2 CPU offload policy flag; then loads the full state dict (no device mesh, offload policy flag passed through).
        # Inputs: module - unwrapped HF module.
        # Outputs: the wrapped model.
        # Side effects: sets _is_offload_param, _is_offload_optimizer and _uses_fsdp2_cpu_offload_policy attributes.
        from fsdp_turbo.fsdp_turbo import FSDPTurbo

        full_state = module.state_dict()
        module = FSDPTurbo(self.fsdp_turbo_config, module).model
        offload_policy = None
        if self.engine_config.offload_policy or self.engine_config.forward_only:
            self._is_offload_param = False
            self._is_offload_optimizer = False
            offload_policy = True
            self._uses_fsdp2_cpu_offload_policy = True
        fsdp2_load_full_state_dict(module, full_state, None, offload_policy)

        return module

    def _is_ulysses_enabled(self):
        # [FUNCTION DOC]
        # HELPER: Returns whether FSDP-Turbo's "ulysses" group is enabled.
        return self._parallel_state.is_group_enable("ulysses")

    def _process_ulysses_config(self):
        # [FUNCTION DOC]
        # HELPER: Switches the engine to FSDP-Turbo context parallelism.
        # Behavior: forces HF attention implementation to "eager", and replaces ulysses size/group with Turbo's, setting use_ulysses_sp true.
        # Exceptions: ValueError if verl's own ulysses_sequence_parallel_size is already above 1.
        # Side effects: mutates model_config.hf_config and engine attributes.
        if self.ulysses_sequence_parallel_size > 1:
            raise ValueError(
                "Do not enable both FSDP-Turbo CP and verl Ulysses SP. "
                "Use fsdp_kwargs.distributed.ulysses_parallel_size for Turbo CP "
                "and set ulysses_sequence_parallel_size=1."
            )

        self.model_config.hf_config._attn_implementation = "eager"
        self.ulysses_sequence_parallel_size = self._parallel_state.get_ulysses_group_size()
        self.ulysses_parallel_group = self._parallel_state.get_ulysses_group()
        self.use_ulysses_sp = True

    def get_data_parallel_rank(self):
        # [FUNCTION DOC]
        # Returns the data-parallel rank from Turbo parallel state, falling back to the parent before it is initialized.
        if not hasattr(self, "_parallel_state"):
            return super().get_data_parallel_rank()
        return self._parallel_state.get_data_rank()

    def get_data_parallel_size(self):
        # [FUNCTION DOC]
        # Returns the data-parallel world size from Turbo parallel state, falling back to the parent before it is initialized.
        if not hasattr(self, "_parallel_state"):
            return super().get_data_parallel_size()
        return self._parallel_state.get_data_group_size()

    def get_data_parallel_group(self):
        # [FUNCTION DOC]
        # Returns the data-parallel process group from Turbo parallel state, falling back to the parent before it is initialized.
        if not hasattr(self, "_parallel_state"):
            return super().get_data_parallel_group()
        return self._parallel_state.get_data_group()

    def is_mp_src_rank_with_outputs(self):
        # [FUNCTION DOC]
        # Returns whether this rank should report outputs: parent behavior before initialization; otherwise true only for Ulysses rank 0 when ulysses is enabled, else always true.
        if not hasattr(self, "_parallel_state"):
            return super().is_mp_src_rank_with_outputs()
        if self._is_ulysses_enabled():
            is_collect = self._parallel_state.get_ulysses_rank() == 0
        else:
            is_collect = True
        return is_collect

    def optimizer_step(self):
        # [FUNCTION DOC]
        # Clips gradients with FSDP-Turbo's clip_grad_norm and steps the optimizer.
        # Behavior: if a grad scaler exists, unscales first and lets the scaler step/update (it skips non-finite steps); otherwise if grad norm is non-finite prints a warning and zeroes grads instead of stepping. When QAT is enabled, invalidates cached quantization scales afterwards.
        # Outputs: grad norm as Python float.
        # Exceptions: AssertionError if optimizer_config.clip_grad is None.
        # Side effects: updates parameters/optimizer state; prints to stdout.
        """
        Clip gradients, skip update if non-finite, and step optimizer.

        Returns:
            grad_norm (float): Norm of gradients before clipping.
        """
        assert self.optimizer_config.clip_grad is not None

        # getattr fallback: some subclasses (e.g. VeOmniEngine) bypass FSDPEngine.__init__.
        scaler = getattr(self, "scaler", None)

        # Unscale gradients before clip so the clip threshold is applied to true gradient
        # magnitudes, not scaled ones. scaler.step() will skip the update if any grad is inf/nan.
        if scaler is not None:
            scaler.unscale_(self.optimizer)

        from fsdp_turbo.training.clip_grads import clip_grad_norm

        grad_norm_value = clip_grad_norm(model=self.module, max_norm=self.optimizer_config.clip_grad)
        grad_norm = torch.tensor(grad_norm_value, device=next(self.module.parameters()).device, dtype=torch.float32)

        if scaler is not None:
            # scaler handles inf/nan skipping internally via _check_inf_per_device.
            scaler.step(self.optimizer)
            scaler.update()
        else:
            # if grad_norm is not finite, skip the update
            if not torch.isfinite(grad_norm):
                print(f"WARN: grad_norm is not finite: {grad_norm}")
                self.optimizer.zero_grad()
            else:
                self.optimizer.step()

        if self._qat_enabled:
            from verl.utils.qat.core import invalidate_all_scales

            invalidate_all_scales(self.module)

        return grad_norm.item()
