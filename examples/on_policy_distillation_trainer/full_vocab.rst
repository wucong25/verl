Full-vocabulary on-policy distillation
=====================================

This branch reimplements the idea in https://github.com/verl-project/verl/pull/7375
on local main (909bb913), obtaining teacher logits from verl training engines.

Configuration
-------------

Set ``distillation.distillation_loss.loss_mode=forward_kl_full`` and each
teacher's ``inference.name=training_engine``. ``training_engine`` accepts the
corresponding engine configuration; ``training_model`` accepts HFModelConfig
overrides. The worker sets the teacher path and forward-only mode.

The existing inference TP/DP/PP product reserves devices per replica.
For eight devices, set ``inference.tensor_model_parallel_size=8``.
Actual engine parallelism is configured separately under ``training_engine``;
Megatron can use TP=2 and EP=8 within that eight-device replica.
Requests are serialized within each replica and distributed round-robin
across independent replica pools.

Teacher and student must share identical token IDs and compatible multimodal
preprocessing. The example preflight also compares tokenizer JSON and checks
all indexed weight shards. Cross-tokenizer distillation is not implemented.

The objective is KL(teacher || student) at temperature one. Full mode requires
``use_policy_gradient=False``, ``log_prob_min_clamp=null``,
``loss_max_clamp=null``, and fused output kernels disabled on both sides.
Response masking retains causal alignment: row t predicts token t+1.
Rollout temperature controls sampling but does not rescale full KL.

Implementation
--------------

* FSDP exports raw logits and restores packed sequence order, including
  Ulysses gathering when enabled.
* Megatron gathers TP teacher vocabulary shards and restores CP token order
  using its existing postprocessor. Student KL reduces normalizers and losses
  across TP while keeping gradients local. Padded vocabulary columns are masked.
* Dynamic CP and fused output kernels explicitly reject full teacher requests.
* Both classic and TransferQueue agent paths preserve image inputs and M-RoPE
  positions. CPU arrays cross Ray boundaries to avoid device-specific tensor
  serialization dependencies.
* KL probabilities are recomputed in backward. ``full_vocab_chunk_size`` bounds
  intermediate fp32 storage, but dense teacher payloads and student logits
  still require O(tokens * vocabulary) memory. This initial implementation
  favors correctness over teacher batching and communication efficiency.

910C comparison
---------------

Run inside container ``04bb453293be`` from ``/home/w00848261/verl``.
The script keeps outputs and caches within that repository, reads model
directories without modifying them, uses eight student/rollout devices plus
eight FSDP teacher devices, and disables checkpoint saving.

* Teacher: ``/home/Qwen3.5-35B-A3B``
* Student: ``/home/w00848261/Qwen3.5-0.8B``
* Shared vocabulary: 248320

Run both modes from the same initial student weights::

    LOSS_MODE=forward_kl_full SEED=42 STEPS=20 \
      bash examples/on_policy_distillation_trainer/run_qwen35_full_opd_910c.sh

    LOSS_MODE=forward_kl_topk TOPK=64 SEED=42 STEPS=20 \
      bash examples/on_policy_distillation_trainer/run_qwen35_full_opd_910c.sh

Both runs use identical teacher forward, learning rate, data subset, seed and
update count. Thinking is disabled so the 512-token answer limit allows
answer evaluation. The default geometry3k training batch has eight examples;
validation uses 32 examples before training and every five updates.

The example reward adapter preserves the original geometry reward but reports
raw answer accuracy and format score separately. The stock scalar geometry
reward mixes correctness and formatting and must not be reported as accuracy.

Compare validation accuracy and elapsed time. Full KL and truncated top-k
loss have different definitions and their numeric values are not directly
comparable. A short single-seed run on 32 validation examples cannot establish
a general improvement. Report ties and regressions as well as improvements.

FSDP Turbo execution was excluded at the user's request. Its inherited FSDP
path is not evidence of real-hardware validation.

Validation
----------

CPU tests::

    python -m pytest -q \
      tests/workers/test_full_vocab_kl_on_cpu.py \
      tests/workers/test_engine_full_vocab_on_cpu.py \
      tests/workers/test_training_teacher_on_cpu.py

Distributed primitive parity, on an idle NPU pair::

    torchrun --nproc-per-node=2 --master-addr=127.0.0.1 --master-port=29671 \
      tests/special_distributed/test_full_vocab_kl_on_npu.py

Logs are retained under ``.full_opd_runtime``. Runtime directories, Ray state
and temporary caches are excluded locally from Git. No PR is submitted.

Real-weight engine forward smoke on eight free devices::

    BACKEND=fsdp2 python -m torch.distributed.run --nproc-per-node=8 \
      --master-addr=127.0.0.1 --master-port=29681 \
      examples/on_policy_distillation_trainer/validate_training_teacher_910c.py

    BACKEND=megatron python -m torch.distributed.run --nproc-per-node=8 \
      --master-addr=127.0.0.1 --master-port=29681 \
      examples/on_policy_distillation_trainer/validate_training_teacher_910c.py

The Megatron smoke uses TP=2, EP=8, BSHD inputs and disables GDN and MoE
permutation fusion to use the operators available in this container.

Completed v2 pilot (2026-11-19)
-------------------------------

Both FSDP2 runs completed 20 updates with exit code zero, seed 42 and
64 validation examples (overriding the launcher default of 32).
The original comparison_manifest.json records the source hashes used for v2;
all recorded hashes matched when the results were collected.

============================== ============ ============
Metric                         Full         Top-64
============================== ============ ============
Initial answer accuracy        3/64         6/64
Final answer accuracy          3/64         3/64
Mean training step (s)         51.123       51.736
Mean actor update (s)           5.573        4.415
Actor peak allocated (GiB)      4.748        4.683
Actor peak reserved (GiB)       6.049        6.291
Training response cap rate      92.50%       96.25%
============================== ============ ============

These results do not establish a quality improvement: initial validation
accuracy differs, both final accuracies tie, and most training responses
reach the 512-token cap. The cause of the baseline discrepancy has not been
isolated. The raw v2 outputs were not saved, preventing a per-example audit.
The launcher now disables validation shuffling, explicitly selects greedy
validation, and saves validation and training generations inside each run
folder. These changes require a fresh hardware run; they are not a proven
fix for the discrepancy. Use a new RUN_TAG to preserve the original v2 logs.
Before another quality comparison, inspect the saved baseline generations,
verify identical examples and outputs, and inspect truncation. If increasing
the answer budget, change it for both modes and rerun both baselines.

Actor memory numbers are logged worker allocator statistics, not whole-device
peaks and not the combined teacher/rollout footprint. Timings depend on the
sampled responses; the small end-to-end timing difference does not establish
a general throughput advantage. Megatron has real-weight teacher forward
validation, but this training comparison used FSDP2 only.

Reproduce the summary from the retained logs::

    python examples/on_policy_distillation_trainer/summarize_full_vocab_comparison.py \
      --full .full_opd_runtime/forward_kl_full_seed42_steps20_v2/train.log \
      --topk .full_opd_runtime/forward_kl_topk_seed42_steps20_v2/train.log \
      --steps 20 --output .full_opd_runtime/comparison_report_v2.json

The summarizer rejects missing steps, conflicting duplicate metrics and
non-finite metrics. Its report marks baseline differences and high truncation;
it never substitutes the mixed training reward for answer accuracy.
