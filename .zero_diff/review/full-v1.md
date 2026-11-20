# 全模型严格训推对齐配方 full-v1（候选，尚未执行）
## 目标与已确认范围
用户：启用batch不变性；验收Megatron原生log-prob重算、带梯度forward，以及两次真实更新后的在线权重同步；本轮不做独立SFT/数学准确率；性能暂不设上限，先实现严格零差异。主拓扑16 NPU，训练TP2/PP2/EP8/ETP1/CP1，推理TP8/EP8。
本配方验收有限测试集合的原始FP32 log-prob逐元素相等，不宣称任意输入或未测试配置都已得到证明。

## 可复现入口
准备文件：.zero_diff/full_online_001.py
原生worker桥接：.zero_diff/online_worker_008.py
审计：.zero_diff/weight_audit_007.py
数据清单：.zero_diff/case_manifest_full_dataset.json
启动：.zero_diff/run_full_v1.sh（人工确认前不执行）
容器26e3c8f4e2f9fec566cb4609405adf061d9d75bb86098f091aa36561caa245ae；
实际镜像sha256:bdc20cc234a54139fe8b66c871e7b7c625bf0288b783c77a01ec14b00a269c6c。
源码基点：verl 622cd6c12b2796a3b541fe50babebf8aa3076799；vllm ced6857afa0ea7b2e3f0846a62e1394e90f15607；vllm-ascend 0c002b4b44cd2e7e12a094e6f778b62ed793afba。
未提交改动以每run/source.json的SHA256及source-tree精确源码快照为准。/vllm无修改。
仅允许三个用户指定源码目录及其中的诊断文件；模型和数据只读，不修改外部Megatron、Bridge、CANN或其他安装包。

## 配置
- 模型：/mnt/weight/Qwen3.5-35B-A3B，全部40层。
- 数据：DAPO原始parquet第0/101/1392/2218行的真实token前缀，长度65/257/513/1024；不是重复拼接token。每条生成16响应token，ignore_eos=True。此长度用于数值覆盖，不用于完整数学解题。
- 推理：一次混合4条，TP8/EP8，最大模型长度1152，max_num_seqs=4，max_num_batched_tokens=2304，gpu_memory_utilization=0.3。BF16主体/激活/KV，无量化、无MTP；eager，prefix cache和chunked prefill关闭，PCP1；rl_config.enabled=True强制ND布局。
- 推理对齐选项：training_tensor_parallel_size=2，training_router_math/training_expert_math/training_attention_math/training_logprob_math=True。推理worker VLLM_BATCH_INVARIANT=1。
- 训练：每轮四条响应复制两次形成8样本，DP4，动态packed microbatch，max_token_len_per_gpu及infer_max_token_len_per_gpu=4096；参数/优化器offload=True。
- 训练主体参数BF16，GDN A_log保留原FP32；GDN norm以原生BF16零中心delta存储，forward和导出均用FP32的delta+1。全模型30个norm已只读检查可无损转换。
- 训练override_mcore_model_config：preserve_gdn_a_log_fp32=True，preserve_gdn_norm_export_fp32=True，npu_batch_invariant_matmul=True，npu_batch_invariant_reduction=True。
- 训练override_transformer_config：attention_backend=auto，use_flash_attn=True，use_triton_gdn=False，use_ascend_gdn=True，moe_grouped_gemm=True，moe_token_dispatcher_type=alltoall，moe_permute_fusion=True，hidden_dropout=attention_dropout=0。
- 原生Megatron optimizer，lr=1e-5，lr_decay_steps=2；执行恰好两次optimizer_step。
- 数值探针使用原生ppo_loss、全1非零advantages和同版本old_log_probs，use_kl_loss=False；用于确保有效反向和参数变化。它不是完整DAPO奖励/GRPO数学训练，也不报告奖励或准确率改善。
- TinkerTrainingWorker原生forward_backward及engine.optimizer_step在同一train_mode中调用，保持梯度存活到step；不替换模型、反向、优化器算法。
- 在线同步：原生AutoBridge conversion tasks导出，256MiB BucketedWeightSender/device IPC；同一组8个vLLM驻留worker sleep/wake更新，PID和model_id必须不变。无磁盘导出冷启动替代。

## 验收
1. 零更新初次同步：所有推理参数哈希不变。
2. theta0/theta1/theta2：每轮64个响应token的decode↔批量prefill、批量↔单条prefill严格0差异；128个有效训练响应token的原生重算↔rollout、带梯度forward↔rollout、两条原生路径互比严格0；均finite，dtype保留FP32，无后处理舍入。
3. 两次更新的梯度范数finite且>0，训练参数字节哈希实际改变；两次IPC同步完成、驻留身份不变。theta2完成更新后的最终探针，不执行第三次更新。
4. 保存resolved training/inference config、每轮原始log-prob张量、权重哈希、同步收据、源码快照和耗时；所有异常记入result.json，脚本启动器最终必须检查passed=True。
5. 推理耗时包含生成、批量和逐条prefill；native耗时包含重算、forward/backward及前两轮optimizer，不能当作纯算子耗时或生产吞吐。同步耗时单列。
6. 未覆盖Graph、prefix cache、chunked scheduling、量化、MTP、更高并发、其他拓扑、checkpoint restore、SFT与数学质量；Rust Bench 8K/1K基准未完成，不能声称性能相对基线达标。

## 代理证据
- proxy-vllm-028 / proxy-megatron-012：初始128前缀+16响应，推理16/16、原生双路径各64/64严格0。
- proxy-online-008：真实两步/驻留IPC，theta0/1/2原生双路径各64/64均0；PD/mixed每轮16/16均0；初次同步全参数哈希不变。梯度范数76.90190124511719、81.9739761352539；两次同步4.770683s、4.829573s。
- proxy-cases-inference-001：四长度混batch的64个有效响应，PD及单条/批量prefill均0；总诊断耗时40.842058s。
- proxy-online-009：已通过。theta0/1/2每轮PD/mixed各64/64、原生recompute/gradforward各128/128均严格0；初次同步全参数不变；两步梯度范数56.03482437133789、45.78289794921875，全部16rank参数实际改变，驻留身份不变。两次同步5.032431s、4.853402s；native诊断16.370210s/1.979226s/1.860398s，推理诊断12.818745s/4.381022s/4.021117s。

## 人工检查点
参考https://github.com/Inferact/zero-mismatch-rl/blob/main/skill/README.md的C9：提交具体配方评审，确认版本后才启动全规模。本文件尚未确认。
用户已明确“暂不设上限，先实现严格零差异”；本轮先按该要求验证数值闭环，未把未完成的BI=0基准作为性能验收分母。


## 实现修订记录（2026-11-20）
full-v1 的模型、数据、拓扑、精度、更新次数和验收条件不变。全模型首次运行 full-online-001 在更新前 PD 检查失败；它不构成通过证据。后续数值定位带来两项实现修复：

- 真实第8层同Q/K/V的FA短query出现1个BF16 ULP差异。缓存路径query不足16且KV至少16时在query左侧补零到16，保留右下因果位置并裁掉dummy输出。proxy-online-010已在全部三个权重版本验证原生重算和带梯度forward严格0，且两次有效optimizer与驻留IPC通过独立审计。
- 冷启动单独1024前缀在第0层GDN的solve_tril末块出现较大不稳定值。逐阶段证据定位到现有merge16→32/merge32→64内核；确切底层编译器/缓冲区机制尚未判定。推理非packed求解输入末尾增加一个64行全零保护块，求解后裁回真实长度，不改变真实chunk系统、不将保护块传入W/U或状态计算。proxy8-solve-012中单条冷启动、混批和history全部8层逐元素0；GDN测试20 passed。保护块后的两步原生闭环由proxy-online-011验证（本记录写入时运行中）。

上述修订只修改授权目录内适配层；没有改写外部Megatron/MindSpeed/CANN，也没有新增Triton算子。全40层最终验收待full-online-002；失败记录和各次源码快照保留，不把代理通过等同于全模型通过。
