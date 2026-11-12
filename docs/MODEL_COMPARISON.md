# 模型对比报告（U-Net / SegFormer / VMamba）

> 数据来源：`outputs/model_comparison.json`（`scripts/evaluate.py` 在验证集上的真实输出）。
> 所有数值均来自本机 CPU 实测，未做任何虚构或外推。

---

## 1. 实验设置（三模型同规模对比）

| 项 | 值 |
|----|----|
| 数据集 | LoveDA（官方 7 类语义分割） |
| 规模 | 100 张训练 / 20 张验证（快速验证规模，CPU 限制） |
| 输入尺寸 | 256×256（训练 RandomCrop / 验证 Resize） |
| 优化器 | Adam，lr=1e-4，weight_decay=1e-4 |
| 损失 | CrossEntropy(ignore=255) + 0.5×soft Dice |
| 训练 | 8 epoch，batch=8，num_workers=0 |
| 设备 | CPU-only（无 GPU） |

---

## 2. 结果对比

| 模型 | 参数量 | 权重内存 | 模型体积 | 推理(ms) | Acc | Prec | Recall | mIoU | mDice |
|------|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| U-Net (ResNet18, 随机初始化) | 14.33M | 57.3MB | 57.4MB | 384.76 | 0.4740 | 0.2996 | 0.2749 | 0.1536 | 0.2266 |
| **SegFormer-B0 (ImageNet 预训练)** | **3.72M** | **14.9MB** | **14.9MB** | **318.35** | **0.6170** | **0.4576** | **0.4765** | **0.3136** | **0.4261** |
| VMamba | — | — | — | — | — | — | — | — | — |

> VMamba 指标栏留空：本机 CPU-only 无法安装 `mamba-ssm`（无 CUDA 内核回退），
> 未训练即不填写指标，避免伪造。

---

## 3. 结论

1. **SegFormer-B0 全面胜出**：参数量仅 U-Net 的 1/4（3.72M vs 14.33M），
   推理更快（318ms vs 385ms），mDice 近 2 倍（0.4261 vs 0.2266）。
2. **预训练骨干的价值**：U-Net 用随机初始化 ResNet18，在 100 图小数据上欠拟合；
   SegFormer 复用 ImageNet 预训练 MiT-B0 骨干，迁移学习效果显著。
3. **生产选择**：B 模块 API 默认加载 **SegFormer-B0**（轻量、快、准），符合 CPU 部署约束。

---

## 4. CPU 环境限制说明

- 本机 **无 GPU/CUDA**，所有训练/推理在 CPU 完成，8 epoch（100 图）单模型训练需数十分钟，
  故采用快速验证规模；绝对指标偏低是「随机/小数据 + CPU」的预期，不影响模型间相对对比。
- 推理耗时在单线程 CPU 上测得（batch=1，256×256 输入），未做算子级优化。

---

## 5. VMamba 为何未训练

- `mamba-ssm` 与 `causal-conv1d` 是 CUDA 内核库，**无 CPU 回退**；本机无 `nvcc`/CUDA toolkit，
  `pip install` 实测失败。
- 已提供 `models/vmamba.py` 结构参考（导入守卫，GPU 机器可直接接入训练/评估循环），
  详见 `docs/VMAMBA_NOTE.md`。
- 在 GPU 机器上的后续步骤：`pip install mamba-ssm causal-conv1d` → 新建
  `configs/model_vmamba.yaml` 与 `configs/train_vmamba.yaml` → 接入 `scripts/evaluate.py`
  的 `configs/compare.yaml`。

---

## 6. 复现命令

```bash
cd image_analysis
# 评估 + 生成 model_comparison.json
HF_ENDPOINT=https://hf-mirror.com python scripts/evaluate.py --compare configs/compare.yaml
```
