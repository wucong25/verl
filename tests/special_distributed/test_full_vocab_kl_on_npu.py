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

"""Run with torchrun --nproc-per-node=2 --master-addr=127.0.0.1 on an idle NPU pair."""

import json
import os

import torch
import torch.distributed as dist
import torch_npu  # noqa: F401

from verl.trainer.distillation.full_vocab import full_vocab_forward_kl


def main():
    rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(rank)
    dist.init_process_group("hccl")
    try:
        world = dist.get_world_size()
        device = torch.device("npu", rank)
        torch.manual_seed(37)
        student_cpu = torch.randn(2, 5, 32 * world)
        teacher_cpu = torch.randn_like(student_cpu)
        student = student_cpu.chunk(world, -1)[rank].contiguous().to(device).requires_grad_()
        teacher = teacher_cpu.chunk(world, -1)[rank].contiguous().to(device)
        mask = torch.tensor([[1, 1, 0, 0, 0], [1, 1, 1, 1, 0]], device=device)
        result = full_vocab_forward_kl(student, teacher, vocab_parallel_group=dist.group.WORLD, chunk_size=3)
        (result * mask).sum().backward()
        reference_student = student_cpu.requires_grad_()
        log_p = teacher_cpu.log_softmax(-1)
        expected = (log_p.exp() * (log_p - reference_student.log_softmax(-1))).sum(-1)
        (expected * mask.cpu()).sum().backward()
        expected_gradient = reference_student.grad.chunk(world, -1)[rank]
        torch.testing.assert_close(result.cpu(), expected.detach(), atol=2e-5, rtol=2e-5)
        torch.testing.assert_close(student.grad.cpu(), expected_gradient, atol=2e-5, rtol=2e-5)
        print(
            json.dumps(
                {
                    "rank": rank,
                    "world_size": world,
                    "device": str(device),
                    "max_loss_error": (result.cpu() - expected.detach()).abs().max().item(),
                    "max_gradient_error": (student.grad.cpu() - expected_gradient).abs().max().item(),
                    "status": "passed",
                }
            ),
            flush=True,
        )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
