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
# CPU tests for verl.utils.omega_conf_to_dataclass converting OmegaConf configs (including nested ones with a
# _target_ key) into BaseConfig dataclasses, plus a smoke test that scripts/print_cfg.py runs.

import unittest
from dataclasses import dataclass, field

from omegaconf import OmegaConf

from verl.base_config import BaseConfig
from verl.utils import omega_conf_to_dataclass


@dataclass
class TestDataclass(BaseConfig):
    # [CLASS DOC]
    # HELPER: Simple BaseConfig dataclass with hidden_size (default 0) and activation (default "relu").
    hidden_size: int = 0
    activation: str = "relu"


@dataclass
class TestTrainConfig(BaseConfig):
    # [CLASS DOC]
    # HELPER: Nested BaseConfig dataclass with batch_size, a TestDataclass model and an override_config mapping.
    batch_size: int = 0
    model: TestDataclass = field(default_factory=TestDataclass)
    override_config: dict = field(default_factory=dict)


_cfg_str = """train_config:
  _target_: tests.utils.test_config_on_cpu.TestTrainConfig
  batch_size: 32
  model:
    hidden_size: 768
    activation: relu
  override_config: {}"""


class TestConfigOnCPU(unittest.TestCase):
    # [CLASS DOC]
    # unittest case for OmegaConf-to-dataclass conversion on CPU.
    """Test cases for configuration utilities on CPU.

    Test Plan:
    1. Test basic OmegaConf to dataclass conversion for simple nested structures
    2. Test nested OmegaConf to dataclass conversion for complex hierarchical configurations
    3. Verify all configuration values are correctly converted and accessible
    """

    def setUp(self):
        # [FUNCTION DOC]
        # HELPER: Creates self.config from the module-level YAML string (train_config with _target_ TestTrainConfig,
        # batch_size 32, model hidden_size 768 / activation relu, empty override_config).
        self.config = OmegaConf.create(_cfg_str)

    def test_omega_conf_to_dataclass(self):
        # [FUNCTION DOC]
        # Behavior: converts the model sub-config to TestDataclass with an explicit class and asserts values and type.
        # Inputs: self. Outputs: none.
        sub_cfg = self.config.train_config.model
        cfg = omega_conf_to_dataclass(sub_cfg, TestDataclass)
        self.assertEqual(cfg.hidden_size, 768)
        self.assertEqual(cfg.activation, "relu")
        assert isinstance(cfg, TestDataclass)

    def test_nested_omega_conf_to_dataclass(self):
        # [FUNCTION DOC]
        # Behavior: converts the whole train_config to TestTrainConfig and asserts top-level and nested values and that
        # the nested model is a TestDataclass instance. Inputs: self. Outputs: none.
        cfg = omega_conf_to_dataclass(self.config.train_config, TestTrainConfig)
        self.assertEqual(cfg.batch_size, 32)
        self.assertEqual(cfg.model.hidden_size, 768)
        self.assertEqual(cfg.model.activation, "relu")
        assert isinstance(cfg, TestTrainConfig)
        assert isinstance(cfg.model, TestDataclass)


class TestPrintCfgCommand(unittest.TestCase):
    # [CLASS DOC]
    # unittest case running the print_cfg.py script.
    """Test suite for the print_cfg.py command-line tool."""

    def test_command_with_override(self):
        # [FUNCTION DOC]
        # Behavior: runs "python3 scripts/print_cfg.py" as a subprocess (relative to the current directory, so must be run
        # from repo root), asserts exit code 0 and that stdout mentions "critic" and "profiler". Despite the name, no
        # overrides are passed. Inputs: self. Outputs: none. Side effects: spawns a subprocess.
        """Test that the command runs without error when overriding config values."""
        import subprocess

        # Run the command
        result = subprocess.run(
            ["python3", "scripts/print_cfg.py"],
            capture_output=True,
            text=True,
        )

        # Verify the command exited successfully
        self.assertEqual(result.returncode, 0, f"Command failed with stderr: {result.stderr}")

        # Verify the output contains expected config information
        self.assertIn("critic", result.stdout)
        self.assertIn("profiler", result.stdout)


if __name__ == "__main__":
    unittest.main()
