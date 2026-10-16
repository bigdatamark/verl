# Copyright 2025 Bytedance Ltd. and/or its affiliates

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# [MODULE DOC]
# Tests for should_save_ckpt_esi, which decides whether to save a checkpoint ahead of an expiring
# capacity block (ESI = expiration-sensitive instance) based on the MLP_ or SAGEMAKER_
# CURRENT_CAPACITY_BLOCK_EXPIRATION_TIMESTAMP environment variables, the max step duration and an
# optional redundant_time margin.
# KNOWN ISSUE: tests set os.environ variables without restoring them, so state leaks between tests and
# into other test modules; outcomes can depend on execution order (e.g. a leftover MLP timestamp during
# the AWS test).
import os
import time
from datetime import datetime, timedelta
from unittest import TestCase

from verl.utils.checkpoint.checkpoint_manager import should_save_ckpt_esi


class TestShouldSaveCkptEsi(TestCase):
    # [CLASS DOC]
    # unittest TestCase grouping the expiration-timestamp scenarios.
    def test_no_expiration_timestamp(self):
        # [FUNCTION DOC]
        # Removes both env vars; expects False for max_steps_duration=100. Side effects: mutates os.environ.
        """Test case when no expiration timestamp is set"""
        os.environ.pop("MLP_CURRENT_CAPACITY_BLOCK_EXPIRATION_TIMESTAMP", None)
        os.environ.pop("SAGEMAKER_CURRENT_CAPACITY_BLOCK_EXPIRATION_TIMESTAMP", None)
        self.assertFalse(should_save_ckpt_esi(100))

    def test_mlp_expiration_valid(self):
        # [FUNCTION DOC]
        # MLP expiration 90s from now with 30s step duration; expects True (inside the save window).
        # Side effects: sets the MLP env var (not restored).
        """Test valid MLP expiration timestamp requiring save"""
        current_time = time.time()
        os.environ["MLP_CURRENT_CAPACITY_BLOCK_EXPIRATION_TIMESTAMP"] = str(current_time + 90)
        self.assertTrue(should_save_ckpt_esi(30))  # max_steps_duration=30 seconds

    def test_mlp_expiration_passed(self):
        # [FUNCTION DOC]
        # MLP expiration 10s in the past; expects False. Side effects: sets MLP env var.
        """Test expired MLP timestamp"""
        current_time = time.time()
        os.environ["MLP_CURRENT_CAPACITY_BLOCK_EXPIRATION_TIMESTAMP"] = str(current_time - 10)
        self.assertFalse(should_save_ckpt_esi(30))

    def test_mlp_invalid_timestamp(self):
        # [FUNCTION DOC]
        # Non-numeric MLP timestamp "invalid"; expects False (parse failure handled, not raised).
        # Side effects: sets MLP env var.
        """Test invalid MLP timestamp format"""
        os.environ["MLP_CURRENT_CAPACITY_BLOCK_EXPIRATION_TIMESTAMP"] = "invalid"
        self.assertFalse(should_save_ckpt_esi(30))

    def test_mlp_expiration_not_reached(self):
        # [FUNCTION DOC]
        # MLP expiration 200s away with 30s step duration; expects False (too early to save).
        # Side effects: sets MLP env var.
        """Test MLP expiration timestamp with insufficient remaining time"""
        current_time = time.time()
        os.environ["MLP_CURRENT_CAPACITY_BLOCK_EXPIRATION_TIMESTAMP"] = str(current_time + 200)
        self.assertFalse(should_save_ckpt_esi(30))  # max_steps_duration=30

    def test_aws_expiration_not_reached(self):
        # [FUNCTION DOC]
        # SageMaker expiration 100 minutes away (integer epoch seconds) with 30-minute step duration; expects False.
        # Side effects: sets SAGEMAKER env var; MLP var from earlier tests may still be set.
        """Test AWS expiration timestamp with sufficient remaining time"""
        now = datetime.now()
        expiration = now + timedelta(minutes=100)  # Exceeds 90-minute threshold
        os.environ["SAGEMAKER_CURRENT_CAPACITY_BLOCK_EXPIRATION_TIMESTAMP"] = str(int(expiration.timestamp()))
        self.assertFalse(should_save_ckpt_esi(30 * 60))

    def test_redundant_time(self):
        # [FUNCTION DOC]
        # MLP expiration 120s away, step duration 30 and redundant_time 30; expects True (comment says total
        # required is 60+30+30=120). Side effects: sets MLP env var.
        """Test redundant_time parameter effect"""
        current_time = time.time()
        # Total required: 60+30+30=120 seconds
        os.environ["MLP_CURRENT_CAPACITY_BLOCK_EXPIRATION_TIMESTAMP"] = str(current_time + 120)
        self.assertTrue(should_save_ckpt_esi(30, redundant_time=30))

    def test_zero_max_steps_duration(self):
        # [FUNCTION DOC]
        # MLP expiration 60s away with step duration 0; expects False. Side effects: sets MLP env var.
        """Test zero max_steps_duration"""
        current_time = time.time()
        os.environ["MLP_CURRENT_CAPACITY_BLOCK_EXPIRATION_TIMESTAMP"] = str(current_time + 60)
        self.assertFalse(should_save_ckpt_esi(0))
