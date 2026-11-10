# VMamba（阶段 5）说明

## 结论

VMamba **无法在本机（CPU-only Windows，DL 环境）训练或推理**，原因如下，故在三模型
对比表中 VMamba 列为「未训练（需 GPU）」。

## 为什么装不上

- `mamba-ssm` 是 CUDA 内核库：选择性扫描（selective scan）与 `causal-conv1d` 均为
  CUDA 算子，没有 CPU 回退实现。
- 安装实测：`pip install mamba-ssm` 在构建阶段即失败（`ModuleNotFoundError: No module
  named 'torch'` 于隔离构建环境，根因是 setup.py 需要 torch/CUDA 编译环境；本机无
  `nvcc`、无 CUDA toolkit）。
- `transformers` 虽能提供 CPU 可跑的 Transformer 模型（SegFormer），但没有 Mamba/SSM 的
  CPU 实现。

## 本机可用的替代

- **结构参考**：`models/vmamba.py`（VSS 块 + U-Net 风格 VMamba，`mamba_ssm` 导入守卫，
  在 GPU 机器上可直接接入训练/评估循环）。
- **本地现成资产**：`D:\Mamba-UNet` 是一套完整可跑的 Mamba 分割工程，含
  - `code/networks/segmamba.py`（SegMamba）、`code/networks/vision_mamba.py`（Vim）
  - `mamba/mamba_ssm/` 源码、`configs/vmamba_tiny.yaml`
  - 已训练权重 `D:\mambaunet\*.pth`（Dice 最高 ≈ 0.89，为医学影像任务，与梯田地类
    语义不同，**不能直接用于本项目的梯田分割**，仅供结构/流程参考）。

## 在 GPU 机器上训练 VMamba 的步骤

```bash
# 1. 装依赖（需 NVIDIA GPU + CUDA）
pip install mamba-ssm causal-conv1d

# 2. 加入对比配置后训练/评估
#    configs/compare.yaml 增加：
#      - name: vmamba
#        model_config: configs/model_vmamba.yaml
#        checkpoint: checkpoints_vmamba/vmamba_best.pth
#    并新建 configs/model_vmamba.yaml: {name: vmamba, num_classes: 7}
python training/train.py --config configs/train_vmamba.yaml \
    --model-config configs/model_vmamba.yaml
python scripts/evaluate.py --compare configs/compare.yaml
```

## 三模型对比表说明

| 模型 | 本机状态 |
|------|----------|
| U-Net (ResNet18) | ✅ 已训练（CPU） |
| SegFormer-B0 (MiT-B0) | ✅ 已训练（CPU，hf-mirror 预训练骨干） |
| VMamba | ❌ 未训练（CPU-only 无法装 mamba-ssm，需 GPU） |

> 诚实性约束：VMamba 的指标栏不能填写虚假数值；在无 GPU 的前提下应留空并注明原因。
