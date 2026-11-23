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
"""Exact forward KL for aligned dense or vocabulary-parallel logits.

Teacher and student must use identical token IDs and identical vocabulary
partitioning. Sequence alignment and response masking belong to the caller.
This primitive does not implement cross-tokenizer distillation.
"""

import torch
import torch.distributed as dist


def _log_probs(logits, group):
    values = logits.float()
    maximum = values.amax(-1, keepdim=True)
    if group is not None:
        dist.all_reduce(maximum, op=dist.ReduceOp.MAX, group=group)
    shifted = values - maximum
    denominator = shifted.exp().sum(-1, keepdim=True)
    if group is not None:
        dist.all_reduce(denominator, group=group)
    return shifted - denominator.log()


class _FullVocabularyKL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, student, teacher, group, chunk_size):
        ctx.save_for_backward(student, teacher)
        ctx.group = group
        ctx.chunk_size = chunk_size
        width = student.shape[-1]
        student_rows = student.reshape(-1, width)
        teacher_rows = teacher.reshape(-1, width)
        output = torch.empty(student_rows.shape[0], device=student.device, dtype=torch.float32)
        for start in range(0, student_rows.shape[0], chunk_size):
            end = start + chunk_size
            log_q = _log_probs(student_rows[start:end], group)
            log_p = _log_probs(teacher_rows[start:end], group)
            p = log_p.exp()
            terms = torch.where(p > 0, p * (log_p - log_q), 0.0)
            value = terms.sum(-1)
            if group is not None:
                dist.all_reduce(value, group=group)
            output[start:end] = value
        return output.reshape(student.shape[:-1])

    @staticmethod
    def backward(ctx, grad_output):
        student, teacher = ctx.saved_tensors
        width = student.shape[-1]
        student_rows = student.reshape(-1, width)
        teacher_rows = teacher.reshape(-1, width)
        gradient = torch.empty_like(student_rows)
        weights = grad_output.reshape(-1, 1)
        for start in range(0, student_rows.shape[0], ctx.chunk_size):
            end = start + ctx.chunk_size
            q = _log_probs(student_rows[start:end], ctx.group).exp()
            p = _log_probs(teacher_rows[start:end], ctx.group).exp()
            gradient[start:end] = ((q - p) * weights[start:end]).to(student.dtype)
        return gradient.reshape_as(student), None, None, None


def full_vocab_forward_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    *,
    vocab_parallel_group=None,
    chunk_size: int = 128,
) -> torch.Tensor:
    """Return per-token KL(teacher || student) at temperature one.

    Both inputs contain logits over the same support; masked vocabulary
    columns may be negative infinity in both inputs. For Megatron TP,
    supply the TP process group and matching local vocabulary shards on every
    rank. All ranks must use identical token shapes and chunk sizes. The output
    is replicated across TP ranks; the backward returns only the local logit
    gradient, without multiplying by TP size.

    Recomputing probabilities in backward bounds intermediate fp32 storage to
    chunk_size * local_vocab_size. Input logits remain live until backward.
    Teacher logits are detached. Padding vocabulary columns must be masked to negative infinity
    before calling; token padding is handled by masking the returned losses.
    """
    if student_logits.shape != teacher_logits.shape:
        raise ValueError("Full-vocabulary KL requires identical aligned logit shapes.")
    if student_logits.ndim < 2 or student_logits.shape[-1] == 0:
        raise ValueError("Expected logits with a nonempty vocabulary dimension.")
    if student_logits.device != teacher_logits.device:
        raise ValueError("Teacher and student logits must be on the same device.")
    if not student_logits.is_floating_point() or not teacher_logits.is_floating_point():
        raise TypeError("Logits must be floating-point tensors.")
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    return _FullVocabularyKL.apply(student_logits, teacher_logits.detach(), vocab_parallel_group, chunk_size)
