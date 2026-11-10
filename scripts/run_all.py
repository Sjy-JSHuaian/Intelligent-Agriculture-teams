"""统一运行入口：环境/目录/权重/配置/数据 检查 + 启动 FastAPI。

用法（在 image_analysis 根目录运行）：
    python scripts/run_all.py                # 仅检查
    python scripts/run_all.py --start-api    # 检查通过后启动 FastAPI
    python scripts/run_all.py --host 0.0.0.0 --port 8000 --start-api

说明：
- 只做只读检查与提示，**不会**下载模型、修改系统环境或删除文件。
- 检查结果分 OK / WARN / FAIL 三档，FAIL 会给出修复建议。
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")


def _line(status: str, name: str, detail: str) -> None:
    print(f"[{status}] {name:28s} {detail}")


def check_python() -> dict:
    ver = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    ok_ = sys.version_info[:2] == (3, 11)
    _line("OK" if ok_ else "WARN", "Python", f"{ver}（建议 3.11）")
    return {"ok": ok_}


def check_torch() -> dict:
    try:
        import torch
        cuda = torch.cuda.is_available()
        _line("OK", "PyTorch", f"{torch.__version__}  CUDA={cuda}")
        if cuda:
            print("    注意：本机为 CPU-only 项目，检测到 CUDA，训练仍可运行但非预期环境。")
        return {"ok": True, "cuda": cuda}
    except Exception as e:  # noqa: BLE001
        _line("FAIL", "PyTorch", f"导入失败: {e}")
        return {"ok": False}


def check_dependencies() -> dict:
    pkgs = {
        "opencv": "cv2", "numpy": "numpy", "albumentations": "albumentations",
        "yaml": "yaml", "smp": "segmentation_models_pytorch",
        "transformers": "transformers", "fastapi": "fastapi", "uvicorn": "uvicorn",
        "multipart": "multipart", "rasterio": "rasterio",
    }
    all_ok = True
    for label, mod in pkgs.items():
        try:
            importlib.import_module(mod)
            _line("OK", label, "已安装")
        except Exception as e:  # noqa: BLE001
            all_ok = False
            _line("WARN", label, f"缺失/失败: {e}")
    return {"ok": all_ok}


def check_directories() -> dict:
    dirs = {
        "数据目录": ROOT / "data" / "raw" / "loveda",
        "输出目录": ROOT / "outputs",
        "配置目录": ROOT / "configs",
        "U-Net 权重目录": ROOT / "checkpoints",
        "SegFormer 权重目录": ROOT / "checkpoints_segformer",
    }
    all_ok = True
    for label, p in dirs.items():
        if p.exists():
            _line("OK", label, str(p.relative_to(ROOT)))
        else:
            all_ok = False
            _line("WARN", label, f"不存在: {p.relative_to(ROOT)}")
    return {"ok": all_ok}


def check_weights() -> dict:
    weights = {
        "U-Net 最佳权重": ROOT / "checkpoints" / "unet_best.pth",
        "SegFormer 最佳权重": ROOT / "checkpoints_segformer" / "segformer_best.pth",
    }
    all_ok = True
    for label, p in weights.items():
        if p.exists():
            _line("OK", label, f"{p.stat().st_size / 1e6:.1f} MB")
        else:
            all_ok = False
            _line("WARN", label, f"缺失: {p.relative_to(ROOT)}（需训练或用已有权重）")
    return {"ok": all_ok}


def check_configs() -> dict:
    cfgs = ["dataset.yaml", "model.yaml", "model_segformer.yaml",
            "train.yaml", "train_segformer.yaml", "compare.yaml"]
    all_ok = True
    for c in cfgs:
        p = ROOT / "configs" / c
        if p.exists():
            _line("OK", f"configs/{c}", "")
        else:
            all_ok = False
            _line("WARN", f"configs/{c}", "缺失")
    return {"ok": all_ok}


def check_dataset() -> dict:
    root = ROOT / "data" / "raw" / "loveda"
    if not root.exists():
        _line("WARN", "LoveDA 数据", f"目录缺失: {root.relative_to(ROOT)}")
        return {"ok": False}
    found = [d.name for d in root.iterdir() if d.is_dir()]
    has_train = any(s.lower() in ["train"] for s in found)
    has_val = any(s.lower() in ["val", "valid", "validation"] for s in found)
    _line("OK" if has_train else "WARN", "LoveDA Train", "存在" if has_train else "缺失")
    _line("OK" if has_val else "WARN", "LoveDA Val", "存在" if has_val else "缺失")
    _line("OK", "LoveDA 顶层目录", ", ".join(found))
    return {"ok": has_train}


def check_jiabang() -> dict:
    p = ROOT / "data" / "raw" / "jiabang_rgb"
    if p.exists() and any(p.iterdir()):
        _line("OK", "加榜梯田 RGB", f"存在: {p.relative_to(ROOT)}")
        return {"ok": True}
    _line("WARN", "加榜梯田 RGB", "待 A 同学提供（植被/异常阈值需据此校准）")
    return {"ok": False}


def start_api(host: str, port: int) -> None:
    import uvicorn
    print(f"\n启动 FastAPI: http://{host}:{port}  （Swagger: http://{host}:{port}/docs）")
    uvicorn.run("api.main:app", host=host, port=port, reload=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="B 模块统一运行入口")
    parser.add_argument("--start-api", action="store_true", help="检查后启动 FastAPI")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    print("=" * 64)
    print("加榜梯田图像智能分析（B 模块）环境检查")
    print("=" * 64)

    results = {}
    results["python"] = check_python()
    results["torch"] = check_torch()
    results["deps"] = check_dependencies()
    results["dirs"] = check_directories()
    results["weights"] = check_weights()
    results["configs"] = check_configs()
    results["dataset"] = check_dataset()
    results["jiabang"] = check_jiabang()

    print("-" * 64)
    fails = [k for k, v in results.items() if not v.get("ok", False)]
    if fails:
        print(f"存在需注意的项（WARN 不阻塞训练，FAIL 需先解决）：{', '.join(fails)}")
        print("提示：缺失模型权重请运行对应训练脚本；缺失数据请见 dataset_manifest.md。")
    else:
        print("全部检查通过。")

    if args.start_api:
        start_api(args.host, args.port)
    else:
        print("\n启动 API：python scripts/run_all.py --start-api "
              "（或 uvicorn api.main:app --host 0.0.0.0 --port 8000）")


if __name__ == "__main__":
    main()
