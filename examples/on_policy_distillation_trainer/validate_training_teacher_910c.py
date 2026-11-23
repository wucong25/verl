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
"""Real-weight training-engine forward smoke; run with torchrun on free NPUs."""

import json
import os
import sys
from pathlib import Path


def main():
    sys.dont_write_bytecode = True
    ROOT = Path(__file__).resolve().parents[2]
    os.chdir(ROOT)
    for variable, directory in {
        "TMPDIR": "tmp",
        "HF_HOME": "hf",
        "XDG_CACHE_HOME": "cache",
        "TORCH_HOME": "torch",
        "TRITON_CACHE_DIR": "triton",
        "ASCEND_WORK_PATH": "ascend",
        "ASCEND_CACHE_PATH": "ascend_cache",
        "ASCEND_PROCESS_LOG_PATH": "ascend_logs",
    }.items():
        target = ROOT / ".full_opd_runtime" / directory
        target.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(target)
    os.environ["ASCEND_SLOG_PRINT_TO_STDOUT"] = "1"
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"

    # CANN may remove its extension build directory after importing an operator.
    # Avoid concurrent builds from different backends using the same directory.
    os.environ["TORCH_EXTENSIONS_DIR"] = str(
        ROOT / ".full_opd_runtime" / ("extensions_" + os.environ.get("BACKEND", "fsdp2"))
    )
    import numpy as np
    import torch
    import torch.distributed as dist
    import torch_npu  # noqa: F401
    from tensordict import TensorDict

    from verl.trainer.config import CheckpointConfig
    from verl.utils import tensordict_utils as tu
    from verl.workers.config import FSDPEngineConfig, HFModelConfig, McoreEngineConfig
    from verl.workers.engine import EngineRegistry

    backend = os.environ.get("BACKEND", "fsdp2")
    if backend not in {"fsdp2", "megatron"}:
        raise ValueError("BACKEND must be fsdp2 or megatron")
    torch.npu.set_device(int(os.environ.get("LOCAL_RANK", 0)))
    dist.init_process_group("hccl")
    try:
        model_config = HFModelConfig(
            path=os.environ.get("MODEL_PATH", "/home/Qwen3.5-35B-A3B"),
            use_remove_padding=False,
            use_fused_kernels=False,
            enable_gradient_checkpointing=False,
            override_config={"attn_implementation": "eager"},
        )
        shared = dict(
            strategy=backend,
            forward_only=True,
            use_remove_padding=False,
            use_fused_kernels=False,
            use_dynamic_bsz=False,
            infer_micro_batch_size_per_gpu=1,
        )
        if backend == "megatron":
            engine_config = McoreEngineConfig(
                **shared,
                dtype="bfloat16",
                tensor_model_parallel_size=2,
                pipeline_model_parallel_size=1,
                expert_model_parallel_size=8,
                expert_tensor_parallel_size=1,
                context_parallel_size=1,
                override_transformer_config={
                    "attention_backend": "auto",
                    "use_flash_attn": True,
                    "use_triton_gdn": False,
                    "use_ascend_gdn": False,
                    "moe_permute_fusion": False,
                    "moe_grouped_gemm": True,
                    "moe_token_dispatcher_type": "alltoall",
                    "use_naive_l2norm": True,
                },
            )
        else:
            engine_config = FSDPEngineConfig(
                **shared,
                model_dtype="bf16",
                use_torch_compile=False,
            )
        engine = EngineRegistry.new(
            model_type="language_model",
            backend=backend,
            model_config=model_config,
            engine_config=engine_config,
            optimizer_config=None,
            checkpoint_config=CheckpointConfig(),
        )
        engine.initialize()
        tokens = model_config.tokenizer.encode("The capital of France is", add_special_tokens=False)
        ids = torch.tensor(tokens, device="npu")
        offsets = torch.tensor([0, len(tokens)], device="npu")
        data = TensorDict(
            {
                "input_ids": torch.nested.nested_tensor_from_jagged(ids, offsets),
                "position_ids": torch.nested.nested_tensor_from_jagged(
                    torch.arange(len(tokens), device="npu"),
                    offsets,
                ),
                "loss_mask": torch.ones(1, len(tokens), device="npu"),
            },
            batch_size=[1],
        )
        tu.assign_non_tensor(
            data,
            return_full_logits=True,
            use_remove_padding=False,
            use_fused_kernels=False,
            temperature=1.0,
            use_dynamic_bsz=False,
            micro_batch_size_per_gpu=1,
            compute_loss=False,
            return_model_output=True,
        )
        with engine.eval_mode(), torch.no_grad():
            output = engine.infer_batch(data, loss_function=None)
        if engine.is_mp_src_rank_with_outputs() and engine.get_data_parallel_rank() == 0:
            logits = output["model_output"]["full_logits"].values()
            assert logits.shape == (len(tokens), 248320), logits.shape
            assert torch.isfinite(logits).all().item()
            report = {
                "backend": backend,
                "model": model_config.path,
                "shape": list(logits.shape),
                "status": "passed",
                "max_memory_gb": torch.npu.max_memory_allocated() / 2**30,
            }
            np.save(Path(".full_opd_runtime") / (backend + "_teacher_logits.npy"), logits.float().cpu().numpy())
            Path(".full_opd_runtime", backend + "_teacher_forward.json").write_text(json.dumps(report, indent=2))
            print(json.dumps(report), flush=True)
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
