# 数值合同与修改位置

| 模块/环节 | 推理侧对齐 | 训练侧与反向 |
|---|---|---|
| 密集GEMM | 复用vllm-ascend现有BI linear_persistent固定K累加 | opt-in NPU dispatcher复用同一kernel；linear经原生matmul/add autograd，dX/dW/dbias已有独立FP64对照 |
| GDN | AscendC chunk算术、FP32状态；每请求保留chunk入口并重放不满64token尾段，避免prefill/decode算法不同 | 复用原生Megatron/MindSpeed GDN及fla_npu backward；verl仅注册镜像中缺失的兼容schema |
| GDN A_log | exp及门控使用FP32边界 | 混合精度wrapper保留原FP32参数及其identity |
| GDN norm | FP32有效gamma；npu_rmsnorm后BF16边界及FP32silu门控 | 原生BF16零中心delta；在线导出使用conversion tasks包装，FP32加1，避免BF16再次舍入 |
| 残差/归一化 | BF16残差相加，npu_rmsnorm；Gemma型gamma在FP32中计算1+weight | 原生模型和backward复用 |
| Q/K与mRoPE | FP32频率、BF16trig及乘加、3轴交错部分旋转；qknorm与训练一致 | 原生模型和backward复用 |
| full attention | 读取原有KV cache，使用与训练相同的CANN npu_fusion_attention TND算术 | 原生FA及backward复用 |
| MoE路由 | FP32top8后top8内softmax，固定原生路由/概率dtype边界 | 原生router及backward复用 |
| MoE专家与共享专家 | 概率在FC2前加权，稳定专家顺序合并；共享专家投影/门控按原生BF16边界 | 原生grouped GEMM、token dispatch及backward复用 |
| 行并行投影 | 物理TP8上按训练TP2的两段BF16乘法和一次加法；每次读取当前权重，避免更新后使用旧缓存 | 原生TP2模型/通信/backward复用 |
| 词表log-prob | 原FP32logits，按训练TP2分词表，两级固定1024分块sum；沿用原生CE操作顺序 | 仅在原生CE作用域内替换FP32词表sum；原生CE saved tensors、collectives及backward不变 |
| 在线同步 | 原生产IPC接收、load_weights、专家重排/卷积处理；驻留sleep/wake | 原生Bridge导出，GDN norm导出局部FP32修复；原生产BucketedWeightSender |

所有新增精度路径由配方显式开关启用。未修改/vllm和外部Megatron/Bridge/MindSpeed/TE/CANN安装目录。未实现新的Triton kernel；矩阵和词表sum复用当前vllm-ascend BI已有实现。默认BI=0性能未进行正式基准验收。

## 测试范围与边界
新增算子测试覆盖GDN、残差norm、GDN norm及L2norm、router、专家合并、共享/普通row-parallel、mRoPE、FA、log-prob和fixed sum。
训练侧线性反向测试独立比较FP64参考的输入/权重梯度；词表sum测试验证作用域、CPU fallback、退出恢复和autograd。
43个修改/新增Python文件AST及三个仓库git diff --check通过。
模型级严格数值结果以runs/*/result.json和原始.pt为准；proxy-online-008/009均实际完成两次非零更新和驻留在线同步。
未独立验证全模型梯度与另一个训练实现逐参数误差，未做SFT或数学质量评价；不得将有限非零梯度或forward零差异当作这些结论。

## 回归命令
使用.zero_diff/env.sh先source CANN/ATB、设置允许目录内缓存。叶目录pytest隔离上级e2e conftest，避免依赖镜像中没有的modelscope：
```bash
leaf=/vllm-ascend/tests/e2e/nightly/single_node/ops/singlecard_ops
python -m pytest --confcutdir="$leaf" -o cache_dir=/home/w00848261/verl/.zero_diff/cache/pytest "$leaf/test_gdn_training_math.py" "$leaf/test_residual_norm_batch_invariant.py" "$leaf/test_gdn_norm_training_math.py" "$leaf/test_gdn_l2norm_training_math.py" "$leaf/test_training_router_math.py" "$leaf/test_training_expert_math.py" "$leaf/test_training_row_parallel.py" "$leaf/test_training_mrope.py" "$leaf/test_training_attention.py" "$leaf/test_training_logprobs.py" "$leaf/test_training_reduction.py"
```
上述测试此前分组实际通过，整条合并命令仅供复现；不要把合并命令记为已经执行。运行时应独占相应NPU，避免与模型级测量混跑。


## 全模型最终验收补充（2026-11-20）

全 40 层 full-online-002 已完成并通过独立审计。θ₀/θ₁/θ₂ 的原始 FP32 rollout、PD/混批及 Megatron 原生重算/带梯度 forward 均逐元素相等；两次非零 optimizer 更新、驻留 IPC 同步和全模型几何审计通过。详细范围、阶段耗时和复核命令见 ../delivery/README.md。

此前代理之后增加两项推理适配：缓存 FA 短 query 左补零至16行再裁回，以保留真实右下因果位置及训练 kernel 的舍入路径；非 packed GDN solve 输入末尾增加64行零保护块，原求解后裁回，再进入原W/U与状态计算。前者由真实相同Q/K/V重放和6项FA测试支持；后者由worker子阶段捕获、冷启动单条/混批/长历史逐层检查和20项GDN测试支持。三角系统仍按原64-token chunk独立求解，未改变真实token的数学依赖。底层merge kernel末块缺陷的编译器/内存机制尚未判定，本项目只在授权适配层隔离其影响，没有修改外部库或加入新Triton实现。
