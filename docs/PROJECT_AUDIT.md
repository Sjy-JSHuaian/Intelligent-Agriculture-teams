# B 模块项目审查报告（Project Audit）

> 审查时间：2026-11-06（阶段 8）
> 审查范围：`image_analysis/` 全部源码、配置、文档、脚本与测试。

---

## 一、已有功能清单（审查确认）

| 模块 | 文件 | 功能 | 状态 |
|------|------|------|------|
| 数据读取 | `datasets/preprocessing.py` | 中文路径安全读写、归一化、标准化、RGB 指数 | ✅ |
| 数据加载 | `datasets/dataset_loader.py` | LoveDA 配对、类别重映射、统计 | ✅ |
| 数据增强 | `datasets/augmentation.py` | 训练/验证两套 albumentations 流水线 | ✅ |
| 切片 | `datasets/tiling.py` | 大图切块、背景过滤 | ✅ |
| U-Net | `models/unet.py` | smp.Unet(ResNet18) | ✅ 已训练 |
| SegFormer | `models/segformer.py` | transformers MiT-B0 + 7 类解码头 | ✅ 已训练 |
| VMamba | `models/vmamba.py` | 结构参考（mamba-ssm 导入守卫） | ✅ 未训练（CPU 限制） |
| 损失 | `training/losses.py` | CE(ignore255) + soft Dice | ✅ |
| 指标 | `training/metrics.py` | 混淆矩阵 → Acc/Prec/Rec/mIoU/mDice | ✅ |
| 训练 | `training/train.py` | 工厂建模型、checkpoint、曲线 | ✅ |
| 植被 | `analysis/vegetation.py` | VARI/ExG/GLI/NDI + 覆盖度 | ✅ |
| 异常 | `analysis/anomaly.py` | 低绿度/裸地/变化检测 | ✅ |
| 可视化 | `visualization/*` | 掩膜配色/叠加/热力图 | ✅ |
| API | `api/main.py` | FastAPI 8+4 端点 | ✅ |
| 脚本 | `scripts/*` | 数据检查/预测/分析/评估/运行入口 | ✅ |
| 测试 | `tests/*` | 37 个 pytest | ✅ |

---

## 二、审查结论（结构评价）

- **目录结构合理**：`datasets / models / training / analysis / visualization / api / scripts / configs / tests / docs` 分层清晰，职责单一，无循环依赖。
- **无明显重复代码**：各模块函数均被复用（如 `read_image` 统一中文路径处理，`build_model` 统一模型工厂）。
- **路径统一**：源码无硬编码绝对路径，均基于 `ROOT = Path(__file__).resolve().parents[1]` 计算相对路径。
- **设备兼容**：训练/评估均 `cuda if available else cpu`，本机 CPU-only 正确回退。
- **中文路径兼容**：图片读写统一 `np.fromfile+imdecode` / `imencode+tofile`，已用中文路径往返测试覆盖。

---

## 三、发现的问题与修复情况

| # | 问题 | 严重程度 | 位置 | 修复状态 |
|---|------|---------|------|---------|
| 1 | 模型加载失败（权重缺失/加载异常）会导致未捕获异常 → 500，错误信息难理解 | **P0 高** | `api/main.py` | ✅ 已修复：新增 `ModelUnavailable` 异常，权重/配置缺失或加载失败统一返回 **503** + 清晰 message |
| 2 | 缺少「模型信息查询」端点 | **P0 高**（任务要求） | `api/main.py` | ✅ 已新增 `GET /model/info` |
| 3 | 统一返回格式缺 `success` 布尔字段（任务 3 建议结构含 `success`） | P1 中 | `api/main.py` | ✅ 已补齐：成功/失败响应均含 `success`（保留原 `status` 兼容） |
| 4 | 缺少 C 模块 RESTful 风格别名端点 | P1 中 | `api/main.py` | ✅ 已新增 `/segment` `/vegetation/analyze` `/anomaly/detect` `/analysis/full` |
| 5 | 空文件/非法格式错误信息不够明确 | P1 中 | `api/main.py` | ✅ 已修复：`decode_image` 显式区分「空文件」与「无法解码」 |
| 6 | 无 API 集成测试 | P1 中 | `tests/` | ✅ 已新增 `tests/test_api.py`（18 项） |
| 7 | 无统一运行入口 | P1 中 | `scripts/` | ✅ 已新增 `scripts/run_all.py` |
| 8 | `dataset_manifest.md` 状态过期（写「数据未下载」但已下载 18G） | P2 低 | 文档 | ✅ 已更新为实际状态 |
| 9 | `requirements.txt` 注释「尚未安装」过期、列 `gdal`（可选） | P2 低 | 文档 | ⚠️ 保留（gdal 已标注可选，不阻塞；注释属提示性） |
| 10 | `models/vmamba.py` 与 `docs/` 引用 `D:\Mamba-UNet` 本机绝对路径 | P2 低（仅文档/注释） | 注释 | ⚠️ 保留并标注为「本机参考资产」，非运行依赖 |
| 11 | `analysis/vegetation.py::analyze_vegetation` 返回 numpy 数组（不可直接 json 序列化） | P2 低（观察项） | `analysis/vegetation.py` | ✅ 不修改函数语义；API 层与脚本层均已只取可序列化字段 |

---

## 四、修复验证

| 验证项 | 结果 |
|--------|------|
| `pytest tests/ -v` | **37 passed**（19 原有 + 18 新增 API 集成） |
| API 路由注册 | 12 个业务路由（8 原 + 4 别名）全部注册 |
| 真实 SegFormer 推理 | ✅ 加载 3.72M 权重，1024×1024 影像预测 0-6 掩膜正常 |
| `run_all.py` 环境检查 | ✅ 全部 OK（仅「加榜 RGB 待 A 同学」为预期 WARN） |

---

## 五、遗留与建议（不阻塞交付）

1. **加榜梯田真实 RGB 影像**（A 同学）：到位后需校准植被/异常阈值，并验证地理参考（GeoTIFF CRS/仿射变换）保留。
2. **VMamba 训练**：需 GPU 机器（`mamba-ssm` 无 CPU 回退），本机不可行。
3. **多光谱 NIR**：若 A 同学能提供 NIR 波段，可升级为真实 NDVI（当前仅 RGB 指数）。
4. **requirements.txt** 建议后续按实际 `pip freeze` 固化精确版本，便于 A/C 同学复现。
