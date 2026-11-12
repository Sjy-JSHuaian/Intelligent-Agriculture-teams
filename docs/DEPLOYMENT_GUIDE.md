# B 模块部署指南（Windows + Conda）

> 目标：在 Windows 11 + Conda 环境下复现 B 模块全部功能（数据/训练/推理/API）。
> 本机为 **CPU-only、无 CUDA**，本指南不含任何 GPU 步骤。

---

## 1. 环境要求

| 项 | 要求 |
|----|------|
| 操作系统 | Windows 10/11 |
| Python | 3.11（推荐 3.11.x，本机 3.11.15） |
| 包管理 | Conda（环境名 `DL`） |
| 深度学习 | PyTorch 2.13.0 **+cpu**（CPU 版） |
| 网络 | 国内需可访问 HuggingFace 镜像（`hf-mirror.com`），否则 SegFormer 预训练骨干无法下载 |

---

## 2. 创建并激活环境

```bash
conda create -n DL python=3.11 -y
conda activate DL
```

安装 CPU 版 PyTorch：

```bash
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cpu
```

> ⚠️ 如 torchvision 报 `_imaging` 导入错误，需重装 pillow：`pip install --force-reinstall pillow==13.0.0`。

---

## 3. 安装依赖

在 `image_analysis/` 根目录：

```bash
pip install -r requirements.txt
```

关键依赖（已在本机验证）：`opencv-python`、`rasterio`、`albumentations`、
`segmentation-models-pytorch`、`timm`、`transformers`、`fastapi`、`uvicorn`、
`python-multipart`、`pytest`、`numpy`、`scipy`、`scikit-learn`、`matplotlib`。

> 国内安装 Transformers/HuggingFace 权重时设置镜像：
> `set HF_ENDPOINT=https://hf-mirror.com`（或 `$env:HF_ENDPOINT="https://hf-mirror.com"`）。

---

## 4. 数据放置

```
image_analysis/data/raw/
├── loveda/                 # LoveDA（已下载，约 18G）
│   ├── Train/  (Urban/Rural 各 images_png + masks_png)
│   ├── Val/
│   └── Test/
└── jiabang_rgb/            # 加榜梯田 RGB（待 A 同学提供）
```

- LoveDA 下载：官方 Zenodo 在国内被 DNS 屏蔽，本项目用 **ModelScope 镜像**
  （`OmniData/LoveDA`）克隆。
- 若自行准备，需保证目录含 `images_png/` 与 `masks_png/`（数据集加载器按此递归查找）。

---

## 5. 模型权重

| 模型 | 权重路径 | 大小 |
|------|---------|------|
| U-Net (ResNet18) | `checkpoints/unet_best.pth` | 57.4 MB |
| SegFormer-B0 | `checkpoints_segformer/segformer_best.pth` | 14.9 MB |

- 若从零开始，训练生成权重：

```bash
# U-Net
python training/train.py --config configs/train.yaml --model-config configs/model.yaml

# SegFormer（需设 HF 镜像）
HF_ENDPOINT=https://hf-mirror.com python training/train.py \
    --config configs/train_segformer.yaml --model-config configs/model_segformer.yaml
```

- VMamba 权重在本机**不可生成**（需 GPU + `mamba-ssm`），见 `VMAMBA_NOTE.md`。

---

## 6. 启动服务

```bash
cd image_analysis

# 方式一：统一入口（先检查再启动）
python scripts/run_all.py --start-api --host 0.0.0.0 --port 8000

# 方式二：直接 uvicorn
HF_ENDPOINT=https://hf-mirror.com python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
```

验证：浏览器打开 `http://localhost:8000/docs`，或 `curl http://localhost:8000/health`。

---

## 7. 运行测试

```bash
pytest tests/ -v        # 期望 37 passed
```

---

## 8. 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| 中文路径图片读写失败 | OpenCV `imread/imwrite` 不支持中文路径 | 本项目已用 `np.fromfile+imdecode` / `imencode+tofile`，无需额外处理 |
| SegFormer 下载超时 | HuggingFace 被墙 | 设 `HF_ENDPOINT=https://hf-mirror.com` |
| `ModuleNotFoundError: mamba_ssm` | CPU-only 无 CUDA | 正常，VMamba 本机不可训练，见 `VMAMBA_NOTE.md` |
| `/predict` 返回 503 | 权重缺失 | 检查 `checkpoints_segformer/segformer_best.pth` 是否存在 |
| matplotlib 中文标题方框 | 无中文字体 | 已设 `font.sans-serif = Microsoft YaHei/SimHei` |
| 训练很慢 | CPU-only | 属预期，用快速验证规模（100 图/8 epoch） |

---

## 9. 给 A/C 同学的交付物

- 代码：`image_analysis/`（B 模块）
- 权重：`checkpoints*/`（U-Net + SegFormer）
- 接口文档：`docs/API_INTEGRATION.md`
- 数据格式：`docs/DATA_FORMAT_SPEC.md`
- 模型对比：`docs/MODEL_COMPARISON.md`
