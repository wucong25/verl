#!/usr/bin/env bash
# Same teacher training-engine forward for full-vocabulary and top-k OPD.
set -euo pipefail
REPO_ROOT=/home/w00848261/verl
cd "$REPO_ROOT"
TEACHER_MODEL=${TEACHER_MODEL:-/home/Qwen3.5-35B-A3B}
STUDENT_MODEL=${STUDENT_MODEL:-/home/w00848261/Qwen3.5-0.8B}
LOSS_MODE=${LOSS_MODE:-forward_kl_full}
SEED=${SEED:-42}
STEPS=${STEPS:-100}
RESPONSE_LENGTH=${RESPONSE_LENGTH:-8192}
PROMPT_LENGTH=${PROMPT_LENGTH:-1024}
MAX_MODEL_LEN=$((PROMPT_LENGTH + RESPONSE_LENGTH))
TOPK=${TOPK:-64}
RUN_NAME="${RUN_TAG:-${LOSS_MODE}_seed${SEED}_steps${STEPS}_response${RESPONSE_LENGTH}}"
if [[ -z "${TEACHER_ENGINE_CONFIG:-}" ]]; then
    TEACHER_ENGINE_CONFIG='{strategy:fsdp2,use_remove_padding:false,model_dtype:bf16,use_torch_compile:false,fsdp_size:8}'
fi
RUN_DIR="$REPO_ROOT/.full_opd_runtime/$RUN_NAME"
mkdir -p "$RUN_DIR" "$REPO_ROOT/.tmp" "$REPO_ROOT/.ray"
export TMPDIR="$REPO_ROOT/.tmp"
export XDG_CACHE_HOME="$REPO_ROOT/.full_opd_runtime/cache"
export HF_HOME="$REPO_ROOT/.full_opd_runtime/hf"
export TORCH_HOME="$REPO_ROOT/.full_opd_runtime/torch"
export TORCH_EXTENSIONS_DIR="$REPO_ROOT/.full_opd_runtime/torch_extensions"
export TRITON_CACHE_DIR="$REPO_ROOT/.full_opd_runtime/triton"
export ASCEND_WORK_PATH="$REPO_ROOT/.full_opd_runtime/ascend"
export ASCEND_CACHE_PATH="$REPO_ROOT/.full_opd_runtime/ascend_cache"
export ASCEND_PROCESS_LOG_PATH="$REPO_ROOT/.full_opd_runtime/ascend_logs"
export RAY_TMPDIR="$REPO_ROOT/.ray"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HCCL_CONNECT_TIMEOUT=1500
export HCCL_HOST_SOCKET_PORT_RANGE=60000-60050
export HCCL_NPU_SOCKET_PORT_RANGE=61000-61050
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export WANDB_MODE=disabled
# Toolkit setup scripts may reference unset variables.
set +u
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /usr/local/Ascend/nnal/atb/set_env.sh
set -u
python3 examples/on_policy_distillation_trainer/full_vocab_preflight.py \
    --teacher "$TEACHER_MODEL" --student "$STUDENT_MODEL" > "$RUN_DIR/preflight.json"
ARGS=(
    algorithm.adv_estimator=grpo
    algorithm.use_kl_in_reward=False
    data.train_files=/home/w00848261/geo3k/train.parquet
    data.val_files=/home/w00848261/geo3k/test.parquet
    data.train_batch_size=8 data.train_max_samples=128 data.val_max_samples=32
    "data.max_prompt_length=$PROMPT_LENGTH" "data.max_response_length=$RESPONSE_LENGTH"
    +data.apply_chat_template_kwargs.enable_thinking=False
    data.validation_shuffle=False
    data.filter_overlong_prompts=True data.truncation=error data.shuffle=True "data.seed=$SEED"
    "actor_rollout_ref.model.path=$STUDENT_MODEL"
    actor_rollout_ref.model.use_remove_padding=False
    actor_rollout_ref.model.use_fused_kernels=False
    actor_rollout_ref.model.enable_gradient_checkpointing=True
    +actor_rollout_ref.model.override_config.attn_implementation=eager
    actor_rollout_ref.actor.strategy=fsdp2
    actor_rollout_ref.actor.use_torch_compile=False
    actor_rollout_ref.actor.optim.lr=1e-6
    actor_rollout_ref.actor.ppo_mini_batch_size=8
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1
    actor_rollout_ref.actor.use_dynamic_bsz=False
    actor_rollout_ref.actor.use_kl_loss=False
    actor_rollout_ref.actor.entropy_coeff=0
    actor_rollout_ref.actor.fsdp_config.fsdp_size=8
    actor_rollout_ref.actor.fsdp_config.param_offload=True
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
    actor_rollout_ref.rollout.name=vllm
    actor_rollout_ref.rollout.tensor_model_parallel_size=1
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4
    actor_rollout_ref.rollout.n=1 actor_rollout_ref.rollout.temperature=1.0
    actor_rollout_ref.rollout.val_kwargs.do_sample=False
    actor_rollout_ref.rollout.val_kwargs.temperature=0
    actor_rollout_ref.rollout.val_kwargs.n=1
    "actor_rollout_ref.rollout.seed=$SEED"
    actor_rollout_ref.rollout.enforce_eager=True
    actor_rollout_ref.rollout.enable_prefix_caching=False
    actor_rollout_ref.rollout.free_cache_engine=True
    "actor_rollout_ref.rollout.max_model_len=$MAX_MODEL_LEN"
    "actor_rollout_ref.rollout.max_num_batched_tokens=$MAX_MODEL_LEN"
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1
    +actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=0
    distillation.enabled=True distillation.n_gpus_per_node=8 distillation.nnodes=1
    "distillation.teacher_models.teacher_model.model_path=$TEACHER_MODEL"
    distillation.teacher_models.teacher_model.key=hiyouga/geometry3k
    distillation.teacher_models.teacher_model.num_replicas=1
    distillation.teacher_models.teacher_model.inference.name=training_engine
    distillation.teacher_models.teacher_model.inference.tensor_model_parallel_size=8
    "++distillation.teacher_models.teacher_model.training_engine=$TEACHER_ENGINE_CONFIG"
    '++distillation.teacher_models.teacher_model.training_model={override_config:{attn_implementation:eager}}'
    "distillation.distillation_loss.loss_mode=$LOSS_MODE"
    "distillation.distillation_loss.topk=$TOPK"
    distillation.distillation_loss.use_task_rewards=False
    distillation.distillation_loss.use_policy_gradient=False
    distillation.distillation_loss.loss_max_clamp=null
    distillation.distillation_loss.log_prob_min_clamp=null
    reward.custom_reward_function.path=$REPO_ROOT/examples/on_policy_distillation_trainer/geometry_accuracy.py
    trainer.device=npu trainer.n_gpus_per_node=8 trainer.nnodes=1
    trainer.logger=[console] trainer.project_name=qwen35_full_opd
    "trainer.experiment_name=$RUN_NAME"
    "trainer.default_local_dir=$RUN_DIR/checkpoints"
    "trainer.validation_data_dir=$RUN_DIR/validation"
    "trainer.rollout_data_dir=$RUN_DIR/rollouts"
    trainer.resume_mode=disable trainer.val_before_train=True
    trainer.test_freq=5 trainer.save_freq=-1 trainer.total_epochs=100
    "trainer.total_training_steps=$STEPS"
    ray_kwargs.ray_init.runtime_env.py_executable=null
    "+ray_kwargs.ray_init._temp_dir=$REPO_ROOT/.ray"
    "hydra.run.dir=$RUN_DIR/hydra"
)
python3 -m verl.trainer.main_ppo "${ARGS[@]}" "$@" 2>&1 | tee "$RUN_DIR/train.log"
