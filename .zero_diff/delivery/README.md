# Qwen3.5-35B-A3B 训推严格零差异交付

2026-11-20（Asia/Shanghai）。**已在用户确认的 full-v1 范围通过全 40 层验收**，运行目录：
`/home/w00848261/verl/.zero_diff/runs/full-online-002/`。

## 结果

以下比较直接使用原始 FP32 log-prob、非空真实 response mask；没有舍入、容差放宽或用一个后端的输出替代另一后端。

| 权重版本 | decode ↔ prefill | 混批 ↔ 单条 prefill | Megatron 原生重算 ↔ rollout | Megatron 带梯度 forward ↔ rollout |
|---|---:|---:|---:|---:|
| θ₀ | 64/64 相等 | 64/64 相等 | 128/128 相等 | 128/128 相等 |
| θ₁（更新1后） | 64/64 相等 | 64/64 相等 | 128/128 相等 | 128/128 相等 |
| θ₂（更新2后） | 64/64 相等 | 64/64 相等 | 128/128 相等 | 128/128 相等 |

每项 mismatch=0、max_abs=0、全部有限；两条 Megatron 路径互比也全部相等。
两次原生 optimizer 更新的梯度范数为 280.6439514160156、43.78923416137695；每步全部 16 个训练 rank 都有参数哈希改变。在线 IPC 同步实际完成，8 个 vLLM worker 的 PID/model identity 在三轮中保持不变。初次无更新同步前后，8 个推理 rank 全参数哈希相等。

独立审计 `independent-audit.json` 重新读取了全部 `dual-theta*.pt` 和 `inference-theta*.pt`，以及权重哈希/同步记录。`model-geometry.json` 核验原权重配置、16 个训练 rank 各 20 个语言层（PP2）、8 个推理 rank 各 40 个语言层。`source-current-audit.json` 确认生产源码及实验脚本与运行快照一致。

## 已验收配置与边界

- 容器：26e3c8f4e2f9fec566cb4609405adf061d9d75bb86098f091aa36561caa245ae；镜像 digest：sha256:bdc20cc234a54139fe8b66c871e7b7c625bf0288b783c77a01ec14b00a269c6c。
- 原权重：/mnt/weight/Qwen3.5-35B-A3B；40 层，hidden=2048，256 experts，BF16。
- 训练 Megatron：16 NPU，TP2/PP2/EP8/ETP1/CP1；推理 vLLM：TP8/EP8。
- batch invariance 开启；eager；无 prefix cache、chunked prefill、MTP 或量化。
- 真实 DAPO 前缀长度 65/257/513/1024，每条生成 16 token；训练为四条响应重复两次，共 8 样本。具体数据行和 token 在 `.zero_diff/case_manifest_full_dataset.json`。
- 原生 PPO loss、全 1 非零 advantages、lr=1e-5，完成两次真实数值更新。此项是数值一致性探针，不是 DAPO 奖励优化或数学准确率验收。没有生成最终训练 checkpoint；原始权重目录只读。
- 结论覆盖以上已测配置和样本，不是任意长度/批次/并行布局的普遍证明。未做独立 SFT、数学质量或 RustBench 性能基线验收。

## 实测耗时

| 权重版本 | 生成+混批/单条前向诊断 | 原生重算+带梯度前向/反向（前两轮含更新） | 更新后 IPC 同步 |
|---|---:|---:|---:|
| θ₀ | 13.823 s | 18.370 s | 12.021 s |
| θ₁ | 11.749 s | 11.373 s | 11.836 s |
| θ₂ | 11.721 s | 13.002 s | — |

这是本验收探针的阶段墙钟耗时，不是纯生成吞吐，也不含模型加载和全部权重哈希审计。没有 BI=0 基线分母，不报告加速比。

## 主要改动

- 对齐 GDN、注意力、MoE、路由、RoPE、归一化、矩阵乘及词表规约的运算顺序。
- 保留 GDN A_log 和在线导出的有效 norm gamma 的 FP32 精度，修复真实更新后权重同步的数值损失。
- 在 vLLM 端为短 query 保持训练注意力的计算形状；为 GDN 三角求解加入零保护块，消除已复现的冷启动末块异常。未改外部 MindSpeed/CANN，未新增 Triton 内核。
- Megatron 保留原生前向、反向与优化器路径，适配只在配置显式开启时生效。

最终补丁在本目录 `verl.patch` 和 `vllm-ascend.patch`；`vllm.patch` 为空。`patch-manifest.json` 给出基准 commit、大小与 SHA256。两个非空补丁均通过当前树的 `git apply --reverse --check`，未进行提交或 PR。旧补丁保存在 `before-terminal-guards/`。

验证包括历史各算子 NPU 检查、修复后的 GDN 20 项回归、8 层两步闭环 proxy-online-011 和全 40 层两步闭环 full-online-002。完整运行证据优先于本摘要。

## 复核与复现

以下命令在上述容器内执行：

```bash
cd /home/w00848261/verl
source .zero_diff/env.sh
python .zero_diff/audit_receipt.py .zero_diff/runs/full-online-002
python .zero_diff/audit_full_geometry.py .zero_diff/runs/full-online-002
```

如需重新执行完整 16 NPU 验收，选择未使用的 run 名：

```bash
cd /home/w00848261/verl
bash .zero_diff/run_full_acceptance.sh full-v1-repeat-001
```

启动脚本自动 source CANN 和 ATB，拒绝覆盖已有 run 目录，保存源码快照并在最后执行两项独立审计。实际完成全模型验收的固定入口为 `.zero_diff/run_full_v1_002.sh`。可重复运行入口仅参数化 run 路径和 IPC job；已通过 shell 语法、CLI help 及归一化 AST 等价检查，没有为路径参数化再重复一次完整 NPU 实验。

配置细节见 `.zero_diff/review/full-v1.md`；实现合同见 `.zero_diff/review/numerical-contract.md`。历史失败记录完整保留。
