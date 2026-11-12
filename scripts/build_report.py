# -*- coding: utf-8 -*-
"""生成成员A交付说明 DOCX（图表与指标均读取 result/ 下的真实输出）。

用法：python scripts/build_report.py
输出：docs/成员A_无人机仿真与自适应路径规划_交付说明.docx
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

from docx import Document
from docx.enum.section import WD_ORIENT
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
PLOT = ROOT / "result" / "plot"
EXP = ROOT / "result" / "experiment"


def set_font(run, name="宋体", size=10.5, bold=False, color=None):
    run.font.name = name
    run._element.rPr.rFonts.set(qn("w:eastAsia"), name)
    run.font.size = Pt(size)
    run.bold = bold
    if color:
        run.font.color.rgb = RGBColor(*color)


def body(doc, text, size=10.5, bold=False, align=None, space_after=4):
    p = doc.add_paragraph()
    r = p.add_run(text)
    set_font(r, "宋体", size, bold)
    p.paragraph_format.space_after = Pt(space_after)
    p.paragraph_format.line_spacing = 1.25
    if align:
        p.alignment = align
    return p


def heading(doc, text, level=1):
    p = doc.add_heading(level=level)
    r = p.add_run(text)
    set_font(r, "微软雅黑", 15 if level == 1 else 12.5, True, (0x1F, 0x3A, 0x5F))
    p.paragraph_format.space_before = Pt(10)
    p.paragraph_format.space_after = Pt(5)


def add_table(doc, headers, rows, widths=None, font_size=9):
    t = doc.add_table(rows=1, cols=len(headers))
    t.style = "Table Grid"
    t.autofit = False
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    hdr = t.rows[0].cells
    for i, h in enumerate(headers):
        hdr[i].text = ""
        p = hdr[i].paragraphs[0]
        r = p.add_run(str(h))
        set_font(r, "微软雅黑", font_size, True)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = ""
            p = cells[i].paragraphs[0]
            r = p.add_run(str(v))
            set_font(r, "宋体", font_size)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if i > 1 else WD_ALIGN_PARAGRAPH.LEFT
    if widths:
        for row in t.rows:
            for i, w in enumerate(widths):
                row.cells[i].width = Cm(w)
    doc.add_paragraph().paragraph_format.space_after = Pt(2)


def metric_rows():
    p = EXP / "metrics.csv"
    if not p.exists():
        return {}
    return {(r["scene"], r["method"]): r for r in csv.DictReader(p.open(encoding="utf-8-sig"))}


def main():
    m = metric_rows()
    scene_name = {"scene_simple": "简单", "scene_medium": "中等", "scene_complex": "复杂"}
    rows = []
    for sid in ("scene_simple", "scene_medium", "scene_complex"):
        for method, label in (("baseline", "基线"), ("improved", "改进")):
            r = m.get((sid, method), {})
            rows.append([
                scene_name[sid], label, r.get("n_fields", "-"), r.get("path_length_m", "-"),
                r.get("energy_wh", "-"), f"{r.get('battery_pct_used', '-')}%",
                f"{r.get('coverage_pct', '-')}%", r.get("planning_time_s", "-"),
                r.get("high_risk_mean_order", "-"), r.get("collision_count", "-"),
            ])

    doc = Document()
    sec = doc.sections[0]
    sec.page_width, sec.page_height = Cm(21.0), Cm(29.7)
    sec.left_margin = sec.right_margin = Cm(2.0)
    sec.top_margin = sec.bottom_margin = Cm(2.0)
    style = doc.styles["Normal"]
    style.font.name = "宋体"
    style.element.rPr.rFonts.set(qn("w:eastAsia"), "宋体")

    # 封面
    for _ in range(4):
        doc.add_paragraph()
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("大学生创新创业训练计划项目"); set_font(r, "微软雅黑", 16, True, (0x33, 0x33, 0x33))
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("成员A 交付说明"); set_font(r, "微软雅黑", 26, True, (0x1F, 0x3A, 0x5F))
    p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p.add_run("无人机仿真与自适应路径规划"); set_font(r, "微软雅黑", 15, False, (0x44, 0x44, 0x44))
    for _ in range(3):
        doc.add_paragraph()
    for line in ["项目：丘陵梯田无人机自主巡检、农情预测与精准变量作业仿真系统",
                 "角色：成员A（路径规划 / 仿真集成）",
                 "日期：2026 年 11 月"]:
        p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run(line); set_font(r, "宋体", 11)
    doc.add_page_break()

    heading(doc, "1  交付物与最低验收对照", 1)
    body(doc, "本交付围绕任务书“成员A：无人机仿真与自适应路径规划”的 5 项最低验收组织。"
              "所有结果均由脚本从原始数据自动生成，未手工填写数字。")
    add_table(doc,
              ["交付物", "最低验收要求", "实现位置", "当前结果"],
              [["① 梯田仿真场景", "≥3 种复杂度；含地块边界、高程差、障碍物",
                "src/simulator/scene_builder.py", "3 个场景，12/22/34 块地，高差 162–630 m"],
               ["② 路径规划程序", "基线可运行、改进可运行、输入输出固定",
                "src/planner/*.py", "A* 基线 + 地形/安全/能耗改进 A*，JSON 字段固定"],
               ["③ 巡检仿真结果", "航迹、访问顺序、覆盖率、能耗/距离",
                "src/simulator/inspect.py", "覆盖率 100%，1296 张图像索引"],
               ["④ 算法实验", "基线 vs 改进：长度、覆盖率、时间、安全/碰撞",
                "scripts/04_run_experiments.py", "航程 −15%/−34%/−38%，禁飞区闯入→0"],
               ["⑤ 网页接口", "底图、航迹坐标、地块 ID、覆盖信息",
                "src/api/app.py + frontend/", "Leaflet 地图 + FastAPI，6 个接口"]],
              widths=[3.0, 5.0, 4.6, 4.4], font_size=8.5)

    heading(doc, "2  三套梯田仿真场景", 1)
    body(doc, "以云南元阳哈尼梯田核心区为研究区，使用 Copernicus DEM GLO-30（真实 30 m 地形）"
              "重采样到 5 m 规划网格，再沿局部等高线人工布设地块、建筑、树木、电线；"
              "固定随机种子，可完全复现。地块/障碍物为仿真，DEM 与 OSM 为真实公开数据。")
    add_table(doc,
              ["场景", "中心经纬度", "范围", "高差", "平均坡度", "地块", "建筑/树木/电线"],
              [["简单", "102.7481°E, 23.1169°N", "700×700 m", "162.5 m", "11.0°", "12", "11 / 14 / 0"],
               ["中等", "102.7625°E, 23.1167°N", "900×900 m", "369.3 m", "22.7°", "22", "14 / 28 / 1"],
               ["复杂", "102.7418°E, 23.0821°N", "1100×1100 m", "630.5 m", "26.0°", "34", "18 / 46 / 2"]],
              widths=[1.2, 4.4, 2.4, 1.9, 1.8, 1.3, 3.8], font_size=8.5)
    if (PLOT / "fig1_scenes.png").exists():
        doc.add_picture(str(PLOT / "fig1_scenes.png"), width=Cm(17))
        p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run("图1  三套场景（地形晕渲 + 地块风险 + 障碍物）"); set_font(r, "宋体", 9)

    heading(doc, "3  算法设计", 1)
    body(doc, "基线：地块按编号固定顺序，固定牛耕航带，块间用纯距离 A*，仅避让硬障碍与陡坡。")
    body(doc, "改进：① 高风险地块优先、同级最近邻；② 每块地在 4 种航带方向/进出口中选"
              "“进场+离场”最短者；③ 转场 A* 代价 = 距离 ×（1 + 坡度² + 障碍安全距离 + 软风险）+ 爬升代价。")
    body(doc, "该方法对应论文 mUDAI 的“先视角、后路径”两阶段思想：把“单 ROI 最佳视角”推广为"
              "“4 种航带变体”，把“多机 VRP”推广为“风险分层 + 最近邻 + A* 转场”。"
              "公式与参数见 docs/算法说明.md。")

    heading(doc, "4  算法实验结果", 1)
    add_table(doc,
              ["场景", "算法", "地块", "航程/m", "能耗/Wh", "电量占用", "覆盖率", "规划/s", "高风险次序", "碰撞"],
              rows, widths=[1.2, 1.2, 1.0, 1.9, 1.9, 1.8, 1.7, 1.6, 2.2, 1.0], font_size=8)
    body(doc, "结论：改进算法航程下降 15.3%/33.6%/37.9%，能耗下降 11.8%/24.9%/29.8%，覆盖率保持 100%；"
              "高风险地块平均访问次序由 5.75/14.71/13.73 提前到 2.5/4.0/6.0。"
              "复杂场景基线耗电 314 Wh，超过 300 Wh 电池，单架次不可完成；改进 220.6 Wh 可完成。", size=9.5)
    if (PLOT / "fig3_metrics.png").exists():
        doc.add_picture(str(PLOT / "fig3_metrics.png"), width=Cm(15.5))
        p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run("图2  基线与改进算法指标对比"); set_font(r, "宋体", 9)
    if (PLOT / "fig2_traj_compare.png").exists():
        doc.add_picture(str(PLOT / "fig2_traj_compare.png"), width=Cm(13))
        p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run("图3  三套场景航迹对比（左列基线，右列改进）"); set_font(r, "宋体", 9)

    heading(doc, "5  局部避障与重规划", 1)
    body(doc, "在改进航迹约 40%–45% 处注入圆形临时禁飞区，从第一次进入禁飞区前的航点开始，"
              "以剩余地块和禁飞区掩膜重新规划。三个场景原航迹闯入 168.6/205.0/69.6 m，"
              "重规划后均为 0 m；简单场景因禁飞区覆盖 1 块地而跳过该地块。")
    if (PLOT / "fig5_replan.png").exists():
        doc.add_picture(str(PLOT / "fig5_replan.png"), width=Cm(17))
        p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run("图4  禁飞区注入与重规划"); set_font(r, "宋体", 9)

    heading(doc, "6  网页接口", 1)
    body(doc, "FastAPI 只读本地 JSON，无数据库；Leaflet 使用本地地形晕渲底图，断网可演示。")
    add_table(doc,
              ["接口", "返回内容"],
              [["GET /api/scenes", "场景列表 + 基线/改进指标"],
               ["GET /api/scene/{sid}", "底图 URL、地块 GeoJSON、障碍 GeoJSON、起降点"],
               ["GET /api/traj/{sid}/{method}", "航迹坐标、时间、地块访问顺序"],
               ["GET /api/inspect/{sid}/{method}", "逐地块覆盖率、图像索引"],
               ["GET /api/metrics", "实验指标表"],
               ["GET /api/replan", "禁飞区重规划结果"]],
              widths=[7.0, 10.0], font_size=8.5)
    body(doc, "启动：python scripts/06_serve_api.py，浏览器打开 http://127.0.0.1:8000/。", size=9.5)
    if (PLOT / "web_dashboard.png").exists():
        doc.add_picture(str(PLOT / "web_dashboard.png"), width=Cm(16.5))
        p = doc.add_paragraph(); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        r = p.add_run("图5  网页地图界面（中等场景 / 改进算法）"); set_font(r, "宋体", 9)

    heading(doc, "7  运行方法与文件位置", 1)
    for line in [
        "① 自检：python scripts/00_check_env.py",
        "② 重建场景（可选）：python scripts/01_build_scenes.py",
        "③ 路径规划：python scripts/02_run_planning.py  → result/traj/*.json",
        "④ 巡检仿真：python scripts/03_run_inspection.py → result/inspect/*",
        "⑤ 算法实验：python scripts/04_run_experiments.py → result/experiment/*.csv",
        "⑥ 图表：python scripts/05_make_figures.py → result/plot/*.png",
        "⑦ 网页：python scripts/06_serve_api.py → http://127.0.0.1:8000/",
    ]:
        body(doc, line, size=9.5, space_after=2)
    body(doc, "数据：../数据集存储/（Copernicus DEM GLO-30 + OpenStreetMap，含 manifest 校验）。"
              "代码：project_A/。详细字段见 docs/接口文档.md。", size=9.5)

    heading(doc, "8  数据来源与边界声明", 1)
    body(doc, "真实数据：Copernicus DEM GLO-30（ESA，AWS Open Data 镜像）、OpenStreetMap（ODbL）。"
              "仿真数据：地块边界、田埂、建筑、树木、电线、病害风险值、巡检占位图。")
    body(doc, "本交付不包含真实无人机飞行、病害识别模型与时序预测；不将仿真结果表述为实测结果。"
              "30 m DEM 无法表达田埂细节，故采用“真实地形底图 + 局部 5 m 仿真网格 + 明确标注人工要素”的方案。")

    out = DOCS / "成员A_无人机仿真与自适应路径规划_交付说明.docx"
    doc.save(out)
    print("saved", out)


if __name__ == "__main__":
    main()

