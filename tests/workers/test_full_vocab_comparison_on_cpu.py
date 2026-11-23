# Copyright 2026 verl contributors
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest

from examples.on_policy_distillation_trainer.summarize_full_vocab_comparison import (
    ACCURACY,
    METRICS,
    compare,
    parse_log,
    summarize,
)


def test_ansi_duplicate_lines_and_reward_are_not_accuracy(tmp_path):
    path = tmp_path / "train.log"
    line = f"\x1b[36m(worker pid=1)\x1b[0m step:0 - {ACCURACY}:0.25 - reward:0.9\n"
    path.write_text(line * 2)
    assert parse_log(path) == {0: {ACCURACY: 0.25, "reward": 0.9}}


@pytest.mark.parametrize("value", ["nan", "inf", "-inf"])
def test_nonfinite_metrics_rejected(tmp_path, value):
    path = tmp_path / "train.log"
    path.write_text(f"step:1 - actor/grad_norm:{value}\n")
    with pytest.raises(ValueError, match="Non-finite"):
        parse_log(path)


def test_conflicting_duplicate_steps_rejected(tmp_path):
    path = tmp_path / "train.log"
    path.write_text("step:1 - loss:1\nstep:1 - loss:2\n")
    with pytest.raises(ValueError, match="Conflicting"):
        parse_log(path)


def test_incomplete_run_rejected():
    with pytest.raises(ValueError, match="Expected steps"):
        summarize({0: {ACCURACY: 0.25}, 2: {ACCURACY: 0.5}}, 2)


def test_baseline_mismatch_and_truncation_reported():
    full = {0: {ACCURACY: 0.25}, 1: {ACCURACY: 0.5, **dict.fromkeys(METRICS.values(), 1)}}
    topk = {**full, 0: {ACCURACY: 0.5}}
    report = compare(full, topk, 1)
    assert not report["baseline_accuracy_matches"]
    assert any("Baseline accuracy differs" in item for item in report["limitations"])
    assert any("token cap" in item for item in report["limitations"])


def test_missing_accuracy_not_replaced_with_reward():
    with pytest.raises(ValueError, match="answer accuracy"):
        summarize({0: {"reward": 0.9}, 1: {ACCURACY: 0.5}}, 1)
