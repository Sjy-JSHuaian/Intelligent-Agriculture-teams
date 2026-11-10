"""Trailing weather indicators, not validated disaster or disease labels.
Default wet-day convention: daily precipitation >= 1 mm. Configurable.
"""
from pathlib import Path
import argparse,json
import numpy as np
import pandas as pd

def runs(values,wet_threshold):
    dry=[];wet=[];d=w=0
    for x in values:
        if pd.isna(x):
            d=w=0;dry.append(np.nan);wet.append(np.nan)
        elif x<wet_threshold:
            d+=1;w=0;dry.append(d);wet.append(w)
        else:
            w+=1;d=0;dry.append(d);wet.append(w)
    return dry,wet

def calculate(frame,wet_threshold=1.0):
    frame=frame.copy(); frame['date']=pd.to_datetime(frame['date'])
    frame=frame.sort_values('date').reset_index(drop=True)
    if frame.date.duplicated().any(): raise ValueError('Duplicate dates')
    if frame.empty: raise ValueError('Empty input')
    # Reindex to expose missing calendar dates; never sum across a gap as if contiguous.
    frame=frame.set_index('date').reindex(pd.date_range(frame.date.min(),frame.date.max(),freq='D'))
    out=pd.DataFrame(index=frame.index);out.index.name='date'
    for prefix,col in [('chirps','chirps_selected_precip_mm_day'),('nasa','nasa_precip_mm_day')]:
        out[prefix+'_rain_1d_mm']=frame[col]
        for days in [3,7,14,30]:
            out[f'{prefix}_rain_{days}d_mm']=frame[col].rolling(days,min_periods=days).sum()
        dry,wet=runs(frame[col],wet_threshold)
        out[prefix+'_consecutive_lt_threshold_days']=dry
        out[prefix+'_consecutive_ge_threshold_days']=wet
    out['nasa_rh_7d_mean_pct']=frame.nasa_rh_pct.rolling(7,min_periods=7).mean()
    out['nasa_tmean_7d_mean_c']=frame.nasa_tmean_c.rolling(7,min_periods=7).mean()
    for depth in ['0_7','7_28']:
        col=f'era5_sm_{depth}_m3_m3_daily_mean'
        out[f'era5_sm_{depth}_7d_mean_m3_m3']=frame[col].rolling(7,min_periods=7).mean()
        out[f'era5_sm_{depth}_change_vs_7days_ago_m3_m3']=frame[col]-frame[col].shift(7)
    return out.reset_index()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--package',type=Path,default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--wet-threshold-mm',type=float,default=1.0)
    args=parser.parse_args()
    if args.wet_threshold_mm<=0:raise ValueError('Wet threshold must be positive')
    out=args.package;tab=out/'02_整理后表格'
    result=calculate(pd.read_csv(tab/'J01_入门合并日表.csv'),args.wet_threshold_mm)
    result.to_csv(tab/'J01_历史气象指标_非风险标签.csv',index=False,encoding='utf-8-sig',date_format='%Y-%m-%d',float_format='%.8f')
    meta={'version':'1.0','wet_day_threshold_mm':args.wet_threshold_mm,
      'threshold_purpose':'Descriptive wet/dry day convention, not a drought/flood/disease threshold',
      'rolling_window':'Trailing calendar days, includes current day; complete valid window required',
      'run_length':'Observed run since file start or most recent missing day; left-censored, true runs may be longer',
      'missing_values':'Missing dates reindexed; rolling values blank if incomplete; runs reset after missing observation',
      'change_vs_7days_ago':'Today minus date minus 7 days, not change of two weekly means',
      'daily_alignment':'Each product keeps original day basis; no joint event labels generated',
      'limitations':'Regional background only; no field-specific water depth or confirmed disease/disaster labels',
      'rows':len(result)}
    (out/'05_来源与核验/指标计算口径.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf8')
    print('Created descriptive indicator table:',len(result),'days; no risk labels.')

if __name__=='__main__':main()
