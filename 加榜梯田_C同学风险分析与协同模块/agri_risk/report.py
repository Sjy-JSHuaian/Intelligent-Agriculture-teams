from pathlib import Path
import html
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from .rules import NAMES

def make_report(root,summary,cfg,clean,features,records,thresholds,pdf=False):
    figures=root/'results/figures';figures.mkdir(exist_ok=True)
    fonts=[Path('C:/Windows/Fonts/msyh.ttc'),Path('C:/Windows/Fonts/simsun.ttc')]
    for font in fonts:
        if font.exists():
            font_manager.fontManager.addfont(str(font))
            plt.rcParams['font.family']=font_manager.FontProperties(fname=str(font)).get_name();break
    plt.rcParams.update({'axes.unicode_minus':False,'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    monthly=pd.read_csv(root/'results/monthly_precipitation.csv')
    fig,ax=plt.subplots(figsize=(10,4))
    t=pd.to_datetime(monthly.month)
    ax.plot(t,monthly.chirps_selected_precip_mm_sum,label='CHIRPS：服务选中单格点',color='#26746b',lw=1.7)
    ax.plot(t,monthly.nasa_precip_mm_sum,label='NASA POWER：区域背景',color='#c87832',lw=1.6)
    ax.axvspan(pd.Timestamp('2021-01-01'),pd.Timestamp('2023-12-31'),color='#ddd',alpha=.25,label='阈值参考期')
    ax.set(ylabel='月累计降水 / mm',title='两个降水产品的时间变化（空间范围和日界线不同）')
    ax.legend(loc='upper right',fontsize=9);ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(figures/'monthly_precipitation.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(10,3.5))
    for i,(typ,label) in enumerate(NAMES.items()):
        rows=[r for r in records if r['event_type']==typ and r['date']>='2024-01-01' and r['status']=='candidate']
        ax.scatter(pd.to_datetime([r['date'] for r in rows]),[i]*len(rows),c=['#bb5629' if r['screen_level']==2 else '#388879' for r in rows],s=15,marker='|')
    ax.set_yticks(range(len(NAMES)),list(NAMES.values()));ax.set_xlim(pd.Timestamp('2024-01-01'),pd.Timestamp('2025-12-31'))
    ax.set_title('固定参考阈值后的候选日期：绿=一级关注，橙=二级关注')
    ax.grid(axis='x',alpha=.15);fig.tight_layout();fig.savefig(figures/'candidate_timeline.png',dpi=180);plt.close(fig)
    typ='rainfall';day=summary['case_peak_rain_3d']['date']
    case=next(r for r in records if r['event_type']==typ and r['date']==day)
    year2025=[r for r in summary['annual_screening'] if r['year']==2025]
    count_lines='\n'.join(f"| {r['event_name']} | {r['candidate_days']} | {r['level2_days']} | {r['episodes_starting_in_year']} |" for r in year2025)
    annual_lines='\n'.join(f"| {r['year']} | {r['chirps_mm']:.1f} | {r['nasa_mm']:.1f} |" for r in summary['annual_precipitation'])
    support=summary['nasa_supported_rainfall_days_2024_2025'];rain_n=summary['rainfall_candidate_days_2024_2025']
    md=f'''# 加榜梯田 C 同学：气象风险筛查与协同分析

本次已完成：1826天气象数据核验、滚动指标、分季节参考阈值、五类区域风险筛查、连续候选过程汇总、A模块JSON适配、给B模块的巡检需求格式和本地API。真实田块任务目前为0条，因为尚未取得带日期与田块关联的加榜影像和已核准边界。示例目录中的田块/图像是明确标注的虚构接口测试数据。

## 1. 你现在能说明的结果

2025年的统计如下。二级关注天数包含在候选天数内；不同类型可能同日出现，不能简单相加成为独立灾害数。连续过程是同类信号连续触发的日期段，其结束日期仅在历史回放后知道。

| 类别 | 候选天数 | 其中二级 | 年内开始的连续过程 |
|---|---:|---:|---:|
{count_lines}

2024-2025年CHIRPS集中降雨规则共筛出{rain_n}个候选日，其中{support}日在NASA相应分季节规则下也触发。产品支持只是背景对照，不是灾害真值或准确率。日降水相关系数为{summary['daily_precip_correlation']:.3f}，60个月累计降水相关系数为{summary['monthly_precip_correlation']:.3f}；两者空间尺度与日界线不同，不能据此断言哪个产品更准确。

## 2. 怎样产生结果

使用2021-2023年参考样本；针对每个目标月份，将参考期该月及前后各一个月的数据合并计算分位数。2024年作固定规则检查、2025年保留为时间外分析；此次未按2024或2025表现调阈值。参考期自身的结果仅用于回顾，不能算独立测试。Q90表示参考样本90%分位数，既不是90%发病概率，也不是气象部门预警等级。

| 类型 | 一级关注 | 二级关注 |
|---|---|---|
| 集中降雨 | CHIRPS 3日累计大于0且达到Q90 | 达到Q95 |
| 少雨与土壤偏干 | CHIRPS 14日累计不高于Q10，且ERA5表层7日均值不高于Q20 | 同时30日累计不高于Q10，土壤7日均值不高于Q10 |
| 持续湿润 | NASA 7日均湿度达到Q90，且7天中至少4天降水达到1mm | 湿度达到Q95且仍满足有雨天数 |
| 偏热背景 | NASA日最高温的3日均值达到Q90 | 达到Q95 |
| 偏冷背景 | NASA日最低温的3日均值不高于Q10 | 不高于Q05 |

“至少4天/1mm”是可配置的原型筛查约定，未作农艺验证；土壤含水率并非田面水深，RGB低绿度并非病害。0级仅表示本规则未触发，缺数据则为insufficient_data。每个滚动窗口要求全部天数有效，缺测不补零。30日特征起始29天为空。

## 3. 一个可复核的历史日期

在2024-2025年中，CHIRPS最大3日累计出现在{day}，为{summary['case_peak_rain_3d']['chirps_3d_mm']:.2f}mm；同日期标签下NASA为{summary['case_peak_rain_3d']['nasa_3d_mm']:.2f}mm。该月CHIRPS参考Q90为{case['thresholds']['rain_3d_q90_mm']:.2f}mm，Q95为{case['thresholds']['rain_3d_q95_mm']:.2f}mm。适合回放“降雨背景 - 排水/田埂核查”的流程；不能据此宣称当天发生田埂损毁或洪涝。

| 年份 | CHIRPS选中格点 / mm | NASA背景 / mm |
|---|---:|---:|
{annual_lines}

## 4. 与同学 A、B 怎样契合

A（按你们分工）输出图像植被、异常和分割结果；C负责区域天气、解释和任务需求；B提供已核准田块、地形/灌排资料并规划路线。A仓库自身将图像称为B模块，文档中的字母不用于重新分配你们的职责。

A的POST /analysis/full与/analyze返回data.anomaly.low_vigor_fraction等字段；裸地比例可保留，但不能默认认定为异常耕地。low_vigor_fraction的分母是整图像素，C要求先确认图像是单田块裁剪再做田块级关联。timestamp是处理时间，不能替代captured_at。当前API不输出地理坐标，因此必须补充独立元数据observation_id、plot_id、captured_at及关联核准状态。

代码核对还发现：/predict存盘灰度掩膜在当前实现中是原始编码1-7；/segmentation的pred_mask实际上指向RGB彩色图。不能把后者当类别ID栅格。旧outputs/analysis/summary.json中的3条LoveDA示例VARI超出当前compute_vari裁剪范围；它们仅用于兼容性审查，未与加榜天气混合，也未用来估计灾害。

C给B的输出是inspection_requests：包含真实plot_id、核准后的多边形、日期、排序理由、图像证据ID、待核查主题及未知信息。P1/P2只是解释明确的巡检顺序，不是灾害等级。B仍需做地形跟随、障碍物、电量及起降条件检查；C不输出航点高度、药量或可直接执行的作业处方。B端字段是本次提出的接口约定，尚未与其实际程序联调。

## 5. 你现在怎样开始

1. 先阅读本报告，再看results/risk_daily.csv中的indicator_values和thresholds，逐行复核案例日期。
2. 运行python run_analysis.py可重建全部数据和图；运行python -m pytest -q tests验证核心行为。
3. 运行python -m uvicorn agri_risk.api:app --host 127.0.0.1 --port 8001，在/docs查看接口；A默认8000，避免端口冲突。
4. 与A交换真实图像结果及拍摄元数据；与B交换真实边界、灌排依据。可以先用examples中的synthetic请求验证字段，不能将它写进论文作为实测。
5. 有真实异常/正常标签后再计算漏报、误报、精确率和召回率；目前只报告候选日期、流程和数据质量。全部数据为事后取得的历史产品，未验证实时产品延迟，因此不把回放结果称为实时预测。

## 来源与边界

研究框来自你提供的截图，约6×5公里，坐标系待核准；本次暂按WGS84解释取数。CHIRPS仅1个服务选中有效格点，NASA与ERA5均为区域背景。三种产品日期未统一日界线。

- NASA POWER：https://power.larc.nasa.gov/docs/services/api/temporal/daily/
- ERA5-Land经Open-Meteo：https://open-meteo.com/en/docs/historical-weather-api
- CHIRPS v2经ClimateSERV：https://climateserv.servirglobal.net/；方法doi:10.1038/sdata.2015.66。
- A仓库：https://github.com/Sjy-JSHuaian/intelligent-agriculture ，本次核对提交1a6d26b417ee15d1980b7e1a14e1c292dd527693（2026-11-10读取）。重点文件api/main.py、analysis/anomaly.py、datasets/preprocessing.py、docs/API_INTEGRATION.md。
'''
    (root/'分析说明.md').write_text(md,encoding='utf8')
    if pdf:make_pdf(root,summary,cfg,case,year2025,figures)

def make_pdf(root,summary,cfg,case,year2025,figures):
    from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,Image,PageBreak
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib import colors
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.pdfbase.cidfonts import UnicodeCIDFont
    font=Path('C:/Windows/Fonts/simsun.ttc')
    if font.exists():pdfmetrics.registerFont(TTFont('CJK',str(font),subfontIndex=0))
    else:pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
    face='CJK' if font.exists() else 'STSong-Light'
    styles={
     'title':ParagraphStyle('title',fontName=face,fontSize=23,leading=32,textColor=colors.HexColor('#224e48'),spaceAfter=16),
     'h':ParagraphStyle('h',fontName=face,fontSize=15,leading=22,textColor=colors.HexColor('#224e48'),spaceBefore=9,spaceAfter=10),
     'p':ParagraphStyle('p',fontName=face,fontSize=10.5,leading=17,spaceAfter=9,wordWrap='CJK'),
     'small':ParagraphStyle('small',fontName=face,fontSize=8.5,leading=13,spaceAfter=6,wordWrap='CJK'),
     'cell':ParagraphStyle('cell',fontName=face,fontSize=9.2,leading=14,wordWrap='CJK')}
    story=[]
    def p(text,style='p'):return Paragraph(html.escape(str(text)),styles[style])
    def add(text,style='p'):story.append(p(text,style))
    def table(rows,widths):
        t=Table([[p(v,'cell') for v in row] for row in rows],colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dcebe6')),
          ('VALIGN',(0,0),(-1,-1),'TOP'),('TOPPADDING',(0,0),(-1,-1),7),('BOTTOMPADDING',(0,0),(-1,-1),7),
          ('LINEBELOW',(0,0),(-1,0),.7,colors.HexColor('#608e83')),
          ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f4f7f5')])]))
        story.append(t);story.append(Spacer(1,10))
    add('加榜梯田\n气象风险筛查与协同分析','title')
    add('C 同学可运行模块 | 2021-2025 历史数据 | 2026-11-10 整理','small')
    add('你负责的闭环','h')
    add('整理天气背景 → 筛选值得关注的日期 → 结合 A 的图像和 B 的田块资料解释 → 给 B 提出有依据的巡检需求。')
    add('已核验1826天数据，生成五类区域信号、连续候选过程、Python脚本和REST接口。真实田块任务目前为0条：还没有带拍摄日期、田块关联的加榜影像及核准边界。虚构接口示例已单独标识。')
    add('2025年固定规则筛查结果','h')
    table([['关注类型','候选天数','其中二级','连续过程*']]+[[r['event_name'],r['candidate_days'],r['level2_days'],r['episodes_starting_in_year']] for r in year2025],[224,75,75,101])
    add('* 连续过程按开始年份计数；二级包含在候选天数内，不同类型可能同日出现。数字不是灾害发生次数、病害发生率或模型准确率。','small')
    add('先从“集中降雨后的排水口、沟渠和田埂核查”做第一个实验。正常稻田蓄水不能直接判为内涝，低绿度也不能确诊病害。')
    story.append(PageBreak())
    add('1  看懂你取得的天气数据','title')
    story.append(Image(str(figures/'monthly_precipitation.png'),width=475,height=190))
    table([['年份','CHIRPS 单格点 / mm','NASA 背景 / mm']]+[[r['year'],f"{r['chirps_mm']:.1f}",f"{r['nasa_mm']:.1f}"] for r in summary['annual_precipitation']],[95,190,190])
    add(f"日降水相关系数为{summary['daily_precip_correlation']:.3f}；60个月累计降水相关系数为{summary['monthly_precip_correlation']:.3f}。这描述两个产品的同步变化，不证明任何一套数据更准确。")
    add('NASA、ERA5是区域背景；CHIRPS服务在当前框中仅选中1个有效格点。当地太阳时日、北京时间日和CHIRPS产品日期没有严格统一，不能把同日差值直接当作误差。')
    peak=summary['case_peak_rain_3d']
    add('可复核案例','h')
    add(f"2024-2025年CHIRPS最大3日累计出现在{peak['date']}，为{peak['chirps_3d_mm']:.2f} mm；NASA同日期标签下为{peak['nasa_3d_mm']:.2f} mm。该月参考Q90={case['thresholds']['rain_3d_q90_mm']:.2f} mm，Q95={case['thresholds']['rain_3d_q95_mm']:.2f} mm。它适合演示降雨触发的核查需求；没有证据证明当天发生了灾害。")
    story.append(PageBreak())
    add('2  风险信号怎样算出来','title')
    add('2021-2023年建立阈值：每个月合并参考期该月与前后各一个月计算分位数；2024-2025年固定规则。本次没有用后两年调参，也没有把参考期回顾当成独立测试。')
    table([['类型','一级关注','二级关注'],
      ['集中降雨','CHIRPS 3日累计>0且≥Q90','≥Q95'],
      ['少雨与土壤偏干','14日雨量≤Q10且表层7日含水率≤Q20','再满足30日雨量≤Q10且含水率≤Q10'],
      ['持续湿润','7日均湿度≥Q90，7天中至少4天降雨≥1mm','湿度≥Q95，且满足有雨天数'],
      ['偏热背景','3日平均日最高温≥Q90','≥Q95'],
      ['偏冷背景','3日平均日最低温≤Q10','≤Q05']],[100,215,160])
    story.append(Image(str(figures/'candidate_timeline.png'),width=475,height=166))
    add('Q90是参考样本的90%分位数，不是90%发病概率；“4天/1mm”是未作农艺验证的原型约定。缺测保持未知，不补零；0级仅为本规则未触发。30日窗口开始的29天为空属正常情况。','small')
    add(f"2024-2025年CHIRPS集中降雨候选日共{summary['rainfall_candidate_days_2024_2025']}天，其中{summary['nasa_supported_rainfall_days_2024_2025']}天也得到NASA对应规则支持。这是产品间的近似对照，不是验证准确率。",'small')
    story.append(PageBreak())
    add('3  怎样连接 A、C、B','title')
    table([['模块','提供什么','本次对接状态'],
      ['A：图像分析','RGB植被指数、低绿度/裸地比例、分割结果','已核对代码与JSON格式；未调用同学实际运行的服务'],
      ['C：你负责','区域天气、候选信号、解释、巡检需求','本地历史分析和接口测试已通过'],
      ['B：地形与路径','核准田块、地形/灌排信息、路线和任务可行性','提供建议字段约定；等待其真实程序和数据联调']],[92,210,173])
    add('A 需要补充的元数据','h')
    add('拍摄时间 captured_at（带时区）、唯一观测编号 observation_id、田块编号 plot_id、单田块裁剪标记、图像与田块的关联核准信息。API timestamp是处理时间。没有坐标的PNG不能凭文件名产生经纬度。')
    add('已经处理的兼容性问题','h')
    add('① low_vigor_fraction的分母为整图像素，不能当作患病植被比例。② /predict当前存盘灰度掩膜为1-7，/segmentation的pred_mask指向彩色图，不能混作类别ID。③ 仓库3条旧LoveDA结果VARI超出当前代码裁剪范围，已隔离；不能当成加榜图像。')
    add('C 给 B 的任务需求','h')
    add('包含真实田块ID、多边形、日期、排序理由、图像证据ID和待核查事项。P1/P2是巡检排序：有二级天气信号及本地线索时优先，其余图像候选为一般核查。没有本地证据时保持区域提示。不给出药量，也不直接批准飞行；实际航线由B计算。')
    add('仓库将图像模块称为B、平台称为C；本报告沿用你确认的分工：A图像、B地形路径、C气象风险。','small')
    story.append(PageBreak())
    add('4  你接下来照着做','title')
    for title,text in [
      ('第一步：读一个日期','打开results/risk_daily.csv，找到案例日期，对照indicator_values、thresholds和status解释为什么触发。'),
      ('第二步：复现计算','安装requirements.txt后运行 python run_analysis.py；需要更新PDF时加 --pdf。核心测试：python -m pytest -q tests。'),
      ('第三步：启动接口','python -m uvicorn agri_risk.api:app --host 127.0.0.1 --port 8001，然后打开 http://127.0.0.1:8001/docs。与A默认8000端口区分。'),
      ('第四步：交换真实证据','请A补齐拍摄元数据，请B提供核准田块及灌排依据。examples中的synthetic请求仅用于接口演示，不能写成实测案例。'),
      ('第五步：形成科研结果','有人工核查的异常/正常标签后，再评估漏报和误报；当前报告数据质量、候选日期与流程可行性。不要把规则自身输出当作真值来评估同一规则。')]:
        add(title,'h');add(text)
    add('来源与可追溯性','h')
    add('NASA POWER Daily API；ERA5-Land经Open-Meteo；CHIRPS v2经ClimateSERV（Funk等，2015，doi:10.1038/sdata.2015.66）。源文件及SHA256记录见inputs/provenance.json。研究框坐标系待核准，本次按原数值暂作WGS84解释。','small')
    add('图像参考：https://github.com/Sjy-JSHuaian/intelligent-agriculture；核对提交1a6d26b417ee15d1980b7e1a14e1c292dd527693。完整网址、字段约定和检查说明见分析说明.md及docs目录。','small')
    add('本包是历史回放原型。尚无实时数据延迟核验、病害标签或实测水位；不能宣称具备实时预测准确率或已验证的变量作业处方。','small')
    def footer(canvas,doc):
        canvas.setFont(face,8);canvas.setFillColor(colors.HexColor('#60756e'))
        canvas.drawString(60,28,'加榜梯田 | C 气象风险与协同分析 | 历史回放原型')
        canvas.drawRightString(535,28,str(doc.page))
    SimpleDocTemplate(str(root/'加榜梯田_C同学分析报告.pdf'),pagesize=(595.28,841.89),
      rightMargin=60,leftMargin=60,topMargin=48,bottomMargin=45,
      title='加榜梯田C同学气象风险筛查与协同分析',author='研究项目资料整理').build(story,onFirstPage=footer,onLaterPages=footer)
