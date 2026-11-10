from pathlib import Path
import json
import numpy as np
import pandas as pd

REQUIRED=['nasa_tmean_c','nasa_tmax_c','nasa_tmin_c','nasa_rh_pct',
 'nasa_precip_mm_day','nasa_wind2m_ms','nasa_solar_mj_m2_day',
 'era5_sm_0_7_m3_m3_daily_mean','era5_sm_7_28_m3_m3_daily_mean',
 'era5_temperature_c_daily_mean','chirps_selected_precip_mm_day']

def load_clean(path):
    df=pd.read_csv(path)
    missing=set(['date',*REQUIRED])-set(df.columns)
    if missing: raise ValueError(f'Missing weather columns: {sorted(missing)}')
    df['date']=pd.to_datetime(df.date,format='%Y-%m-%d',errors='raise')
    if df.date.duplicated().any():raise ValueError('Duplicate weather date; do not silently average')
    if df.empty:raise ValueError('Empty weather input')
    df=df.set_index('date').sort_index()[REQUIRED]
    original_dates=set(df.index)
    df=df.reindex(pd.date_range(df.index.min(),df.index.max(),freq='D'))
    df.index.name='date'; issues=[]
    for c in REQUIRED:
        df[c]=pd.to_numeric(df[c],errors='raise')
        bad=~np.isfinite(df[c]) & df[c].notna()
        for day in df.index[bad]:issues.append({'date':str(day.date()),'column':c,'reason':'non_finite'})
        df.loc[bad,c]=np.nan
    bounds={'nasa_rh_pct':(0,100),'nasa_precip_mm_day':(0,np.inf),
      'chirps_selected_precip_mm_day':(0,np.inf),'nasa_wind2m_ms':(0,np.inf),
      'nasa_solar_mj_m2_day':(0,np.inf),
      'era5_sm_0_7_m3_m3_daily_mean':(0,1),'era5_sm_7_28_m3_m3_daily_mean':(0,1)}
    for c,(low,high) in bounds.items():
        bad=df[c].notna() & ~df[c].between(low,high)
        for day in df.index[bad]:issues.append({'date':str(day.date()),'column':c,'reason':'outside_physical_range'})
        df.loc[bad,c]=np.nan
    bad=(df.nasa_tmin_c>df.nasa_tmean_c)|(df.nasa_tmean_c>df.nasa_tmax_c)
    for day in df.index[bad]:issues.append({'date':str(day.date()),'column':'nasa_temperature','reason':'min_mean_max_inconsistent'})
    df.loc[bad,['nasa_tmin_c','nasa_tmean_c','nasa_tmax_c']]=np.nan
    quality={'rows':len(df),'first_date':str(df.index.min().date()),'last_date':str(df.index.max().date()),
      'inserted_missing_dates':[str(d.date()) for d in df.index if d not in original_dates],
      'missing_after_cleaning':{c:int(df[c].isna().sum()) for c in REQUIRED},'invalid_values':issues,
      'imputation':'None. Missing stays missing; zero rainfall remains zero.',
      'source_day_boundaries':'NASA LST; ERA5 Asia/Shanghai; CHIRPS product date. Not harmonized.'}
    return df.reset_index(),quality

def indicators(df,cfg):
    x=df.set_index('date').copy()
    for src,col in [('chirps','chirps_selected_precip_mm_day'),('nasa','nasa_precip_mm_day')]:
        for n in [1,3,7,14,30]:x[f'{src}_rain_{n}d_mm']=x[col].rolling(n,min_periods=n).sum()
        wet=(x[col]>=cfg['wet_day_mm']).astype(float).where(x[col].notna())
        x[f'{src}_wet_days_7d']=wet.rolling(7,min_periods=7).sum()
        run=[];count=0
        for value in x[col]:
            if pd.isna(value):count=0;run.append(np.nan)
            elif value<cfg['wet_day_mm']:count+=1;run.append(float(count))
            else:count=0;run.append(0.)
        x[f'{src}_observed_dry_run_days']=run
    x['rh_7d_pct']=x.nasa_rh_pct.rolling(7,min_periods=7).mean()
    x['tmax_3d_c']=x.nasa_tmax_c.rolling(3,min_periods=3).mean()
    x['tmin_3d_c']=x.nasa_tmin_c.rolling(3,min_periods=3).mean()
    x['soil_7d_m3_m3']=x.era5_sm_0_7_m3_m3_daily_mean.rolling(7,min_periods=7).mean()
    x['soil_change_7d_m3_m3']=x.era5_sm_0_7_m3_m3_daily_mean-x.era5_sm_0_7_m3_m3_daily_mean.shift(7)
    return x.reset_index()

def write_csv(df,path):
    df.to_csv(path,index=False,encoding='utf-8-sig',date_format='%Y-%m-%d',float_format='%.8f')

def json_safe(obj):
    if isinstance(obj,dict):return {str(k):json_safe(v) for k,v in obj.items()}
    if isinstance(obj,(list,tuple)):return [json_safe(v) for v in obj]
    if isinstance(obj,(pd.Timestamp,)):return obj.isoformat()
    if isinstance(obj,(np.integer,)):return int(obj)
    if isinstance(obj,(float,np.floating)):return float(obj) if np.isfinite(obj) else None
    if isinstance(obj,np.bool_):return bool(obj)
    return obj

def write_json(obj,path):
    Path(path).write_text(json.dumps(json_safe(obj),ensure_ascii=False,indent=2,allow_nan=False),encoding='utf8')
