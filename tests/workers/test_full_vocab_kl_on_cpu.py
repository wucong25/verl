# Copyright 2026 verl contributors
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

import pytest
import torch
import torch.nn.functional as F

from verl.trainer.distillation.full_vocab import full_vocab_forward_kl


@pytest.mark.parametrize("chunk_size", [1, 3, 128])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_value_and_masked_gradient(chunk_size, dtype):
    torch.manual_seed(17)
    student = torch.randn(2, 5, 19, dtype=dtype, requires_grad=True)
    teacher = torch.randn_like(student).requires_grad_()
    mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 1, 0]])
    result = full_vocab_forward_kl(student, teacher, chunk_size=chunk_size)
    log_p = teacher.detach().float().log_softmax(-1)
    reference = F.kl_div(student.float().log_softmax(-1), log_p, log_target=True, reduction="none").sum(-1)
    torch.testing.assert_close(result, reference, atol=3e-6, rtol=3e-6)
    expected_grad = torch.autograd.grad((reference * mask).sum(), student)[0]
    (result * mask).sum().backward()
    torch.testing.assert_close(student.grad, expected_grad, atol=1e-6, rtol=1e-5)
    assert teacher.grad is None


def test_shift_invariance_and_zero_kl():
    logits = torch.tensor([[1000.0, -1000.0, 999.0]], requires_grad=True)
    result = full_vocab_forward_kl(logits, logits.detach() + 100.0)
    torch.testing.assert_close(result, torch.zeros_like(result), atol=1e-6, rtol=0)
    result.sum().backward()
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits), atol=1e-6, rtol=0)


def test_empty_tokens():
    logits = torch.empty(0, 11, requires_grad=True)
    result = full_vocab_forward_kl(logits, logits.detach())
    assert result.shape == (0,)
    result.sum().backward()
    assert logits.grad.shape == logits.shape


@pytest.mark.parametrize("chunk_size", [0, -1, True, 1.5])
def test_invalid_chunk(chunk_size):
    with pytest.raises(ValueError):
        full_vocab_forward_kl(torch.zeros(1, 2), torch.zeros(1, 2), chunk_size=chunk_size)


def test_different_vocabularies():
    with pytest.raises(ValueError, match="identical aligned"):
        full_vocab_forward_kl(torch.zeros(2, 7), torch.zeros(2, 9))


def test_padded_vocab_columns_have_zero_gradient():
    torch.manual_seed(3)
    student = torch.randn(2, 7, requires_grad=True)
    teacher = torch.randn(2, 7)
    reference = full_vocab_forward_kl(student, teacher)
    padded_student = F.pad(student.detach(), (0, 3), value=-torch.inf).requires_grad_()
    padded_teacher = F.pad(teacher, (0, 3), value=-torch.inf)
    actual = full_vocab_forward_kl(padded_student, padded_teacher)
    torch.testing.assert_close(actual, reference)
    actual.sum().backward()
    reference.sum().backward()
    torch.testing.assert_close(padded_student.grad[:, :7], student.grad)
    assert torch.count_nonzero(padded_student.grad[:, 7:]).item() == 0


def test_full_vocab_config_rejects_policy_gradient_and_clamping():
    from verl.workers.config import DistillationLossConfig

    valid = dict(loss_mode="forward_kl_full", use_policy_gradient=False, loss_max_clamp=None, log_prob_min_clamp=None)
    assert DistillationLossConfig(**valid).loss_settings.use_full_vocab
    for override in ({"use_policy_gradient": True}, {"loss_max_clamp": 10.0}, {"log_prob_min_clamp": -10.0}):
        with pytest.raises(ValueError):
            DistillationLossConfig(**(valid | override))


@pytest.mark.parametrize("strategy", ["fsdp", "fsdp2", "fsdp_turbo"])
def test_full_vocab_loss_matches_dense_reference(strategy):
    from types import SimpleNamespace

    from tensordict import TensorDict

    from verl.trainer.distillation.full_vocab_loss import compute_full_vocab_loss

    torch.manual_seed(23)
    teacher = torch.randn(5, 13).log_softmax(-1)
    offsets = torch.tensor([0, 3, 5])
    data = TensorDict({"teacher_logprobs": torch.nested.nested_tensor_from_jagged(teacher, offsets)}, batch_size=[2])
    student = torch.randn(1, 5, 13, requires_grad=True)
    config = SimpleNamespace(strategy=strategy)
    distill = SimpleNamespace(distillation_loss=SimpleNamespace(full_vocab_chunk_size=2))
    actual = compute_full_vocab_loss(config, distill, data, student, "thd")["distillation_losses"]
    expected = F.kl_div(student.log_softmax(-1), teacher.unsqueeze(0), log_target=True, reduction="none").sum(-1)
    torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
    actual.sum().backward()
    assert torch.isfinite(student.grad).all()
