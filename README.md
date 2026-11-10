# 图像智能分析模块（B 模块）

贵州省从江县加榜梯田智慧农业综合监测系统 —— 图像智能分析子系统。

负责：遥感影像及农田图像预处理、梯田/农田语义分割、植被覆盖分析、异常识别、
可视化与 REST API 输出。

---

## 当前状态（阶段 1–7 完成，阶段 8 收尾）

| 阶段 | 内容 | 状态 |
|------|------|------|
| 1 | 环境勘察 + 目录骨架 + 数据下载 | ✅ |
| 2 | 数据读取/预处理/切片/增强/统计/可视化 | ✅ 9 测试 |
| 3 | U-Net 基线（ResNet18） | ✅ |
| 4 | SegFormer-B0（MiT-B0 预训练） | ✅ |
| 5 | VMamba（结构代码 + 文档，CPU 无法训练） | ✅ |
| 6 | 植被/异常分析（RGB 指数） | ✅ 10 测试 |
| 7 | FastAPI REST API | ✅ curl 实测 |
| 8 | 测试 / 文档 / 集成 | 进行中 |

**测试**：`pytest tests/` → 19 通过。

---

## 模型对比结果（验证集 20 张，256×256，CPU）

| 模型 | 参数(M) | 权重内存(MB) | 推理(ms) | Acc | Prec | Recall | mIoU | mDice |
|------|---------|--------------|----------|-----|-------|--------|------|-------|
| U-Net (ResNet18, 随机初始化) | 14.33 | 57.3 | 384.8 | 0.474 | 0.300 | 0.275 | 0.154 | 0.227 |
| **SegFormer-B0 (预训练)** | **3.72** | **14.9** | **318.4** | **0.617** | **0.458** | **0.477** | **0.314** | **0.426** |
| VMamba | — | — | — | — | — | — | — | 需 GPU（未训练） |

> 快速验证规模：100 训练图 / 20 验证图，8 epoch，batch 8，Adam lr 1e-4。
> SegFormer 以 1/4 参数量、更快推理、近 2× mDice 胜出，体现预训练骨干的价值。
> VMamba 依赖 `mamba-ssm`（CUDA），本机 CPU-only 无法训练，指标栏留空不编造。

---

## 关键约定

- **类别映射（官方 LoveDA）**：掩膜取值 0-7，`0=no-data(忽略)、1=background、
  2=building、3=road、4=water、5=barren、6=forest、7=agriculture`。训练重映射
  1-7→0-6、0→255(ignore)。详见 `dataset_manifest.md` 与 `datasets/dataset_loader.py`。
- **中文路径**：OpenCV `imread/imwrite` 不支持 `D:\大创\...`，统一用
  `np.fromfile+imdecode` / `imencode+tofile`。
- **无 NIR 波段**：植被分析使用 RGB 植被指数（VARI/ExG/GLI/NDI），**不称真实 NDVI**。
- **诚实性约束**：未实际计算的字段不填充；不虚构实验数据；未训练模型不填指标。

---

## 环境

- 本机 **CPU-only、无 GPU/CUDA**；conda 环境 `DL`（Python 3.11.15，torch 2.13.0+cpu）。
- 关键依赖：opencv / rasterio / albumentations / smp / timm / transformers / fastapi / uvicorn。

---

## 目录结构

```
image_analysis/
├── configs/            # dataset.yaml / train*.yaml / model*.yaml / compare.yaml
├── data/raw/loveda/    # LoveDA（Train/Val/Test）
├── datasets/           # 读取、预处理、增强、切片、统计
├── models/             # unet / segformer / vmamba（工厂 build_model）
├── training/           # train.py / losses.py / metrics.py
├── analysis/           # vegetation.py / anomaly.py
├── visualization/      # segmentation_visualizer / heatmap
├── api/                # FastAPI main.py
├── scripts/            # stage2_inspect / stage3_predict / stage6_analysis / evaluate
├── outputs/            # segmentation / analysis / api / model_comparison.json
├── checkpoints*/       # 模型权重（unet_best.pth / segformer_best.pth）
├── tests/              # 19 个单元测试
└── docs/               # API_DOCUMENTATION.md / VMAMBA_NOTE.md
```

---

## 使用方法

```bash
cd D:/大创/image_analysis
PY="D:/Anaconda3/envs/DL/python.exe"

# 训练 U-Net 基线
$PY training/train.py --config configs/train.yaml --model-config configs/model.yaml

# 训练 SegFormer（需设镜像下载预训练骨干）
HF_ENDPOINT=https://hf-mirror.com $PY training/train.py \
    --config configs/train_segformer.yaml --model-config configs/model_segformer.yaml

# 模型对比评估（输出 outputs/model_comparison.json）
HF_ENDPOINT=https://hf-mirror.com $PY scripts/evaluate.py --compare configs/compare.yaml

# 分割结果可视化（预测/叠加/逐图指标）
$PY scripts/stage3_predict.py --n 3

# 植被/异常分析
$PY scripts/stage6_analysis.py --input data/raw/loveda/Val --n 3

# 启动 REST API
HF_ENDPOINT=https://hf-mirror.com $PY -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

API 文档见 `docs/API_DOCUMENTATION.md`，Swagger 界面见 `http://localhost:8000/docs`。

---

## 待办 / 依赖外部

- **加榜梯田真实 RGB 影像**：由 A 同学提供后放 `data/raw/jiabang_rgb/`，植被/异常
  阈值需据此校准。
- **VMamba 训练**：需 GPU 机器（`pip install mamba-ssm causal-conv1d`），见
  `docs/VMAMBA_NOTE.md`。
