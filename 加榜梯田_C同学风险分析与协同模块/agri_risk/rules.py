import json
import numpy as np
import pandas as pd

FEATURES=['chirps_rain_3d_mm','chirps_rain_14d_mm','chirps_rain_30d_mm',
          'nasa_rain_3d_mm','soil_7d_m3_m3','rh_7d_pct','tmax_3d_c','tmin_3d_c']
QUANTILES=[.05,.10,.20,.90,.95]
NAMES={'rainfall':'集中降雨关注','water_supply':'少雨与土壤偏干关注',
 'wet_environment':'持续湿润环境关注','warm_anomaly':'偏热背景关注','cold_anomaly':'偏冷背景关注'}
TOPICS={'rainfall':'核查排水口、沟渠和田埂','water_supply':'核查灌溉水源、沟渠通水与田面水位',
 'wet_environment':'核查长势与叶片异常，结合人工农艺判断',
 'warm_anomaly':'核查作物生育期、供水及热胁迫迹象','cold_anomaly':'核查作物生育期与低温影响迹象'}

def fit_thresholds(frame,cfg):
    ref=frame[frame.date.between(cfg['reference_start'],cfg['reference_end'])]
    if ref.empty:raise ValueError('No reference-period observations')
    result=[]
    for month in range(1,13):
        months=[((month-1+i)%12)+1 for i in range(-cfg['seasonal_month_half_window'],cfg['seasonal_month_half_window']+1)]
        group=ref[ref.date.dt.month.isin(months)]
        for feature in FEATURES:
            series=group[feature].dropna()
            if len(series)<cfg['minimum_reference_samples']:
                raise ValueError(f'Insufficient calibration samples: month={month}, feature={feature}, n={len(series)}')
            row={'month':month,'feature':feature,'sample_count':len(series),'reference_months':','.join(map(str,months)),
             'reference_start':cfg['reference_start'],'reference_end':cfg['reference_end']}
            row.update({f'q{int(q*100):02d}':float(series.quantile(q,interpolation='linear')) for q in QUANTILES})
            result.append(row)
    return pd.DataFrame(result)

def screen(frame,thresholds,cfg):
    lookup=thresholds.set_index(['month','feature']).to_dict('index');records=[]
    def q(m,f,p):return float(lookup[m,f][p])
    for r in frame.to_dict('records'):
        day=r['date'];m=day.month
        # Strictly increasing severity must remain relative screening, never probability.
        settings={
          'rainfall':(['chirps_rain_3d_mm'],{'rain_3d_q90_mm':q(m,'chirps_rain_3d_mm','q90'),'rain_3d_q95_mm':q(m,'chirps_rain_3d_mm','q95')}),
          'water_supply':(['chirps_rain_14d_mm','chirps_rain_30d_mm','soil_7d_m3_m3'],{
            'rain_14d_q10_mm':q(m,'chirps_rain_14d_mm','q10'),'rain_30d_q10_mm':q(m,'chirps_rain_30d_mm','q10'),
            'soil_q20_m3_m3':q(m,'soil_7d_m3_m3','q20'),'soil_q10_m3_m3':q(m,'soil_7d_m3_m3','q10')}),
          'wet_environment':(['rh_7d_pct','nasa_wet_days_7d'],{'rh_q90_pct':q(m,'rh_7d_pct','q90'),
            'rh_q95_pct':q(m,'rh_7d_pct','q95'),'min_wet_days_7d':cfg['wet_environment_min_wet_days_7d']}),
          'warm_anomaly':(['tmax_3d_c'],{'tmax_q90_c':q(m,'tmax_3d_c','q90'),'tmax_q95_c':q(m,'tmax_3d_c','q95')}),
          'cold_anomaly':(['tmin_3d_c'],{'tmin_q10_c':q(m,'tmin_3d_c','q10'),'tmin_q05_c':q(m,'tmin_3d_c','q05')})}
        for typ,(required,limits) in settings.items():
            missing=[f for f in required if pd.isna(r[f])]
            level=None;support=None
            if not missing:
                level=0
                if typ=='rainfall':
                    v=r['chirps_rain_3d_mm']
                    if v>0 and v>=limits['rain_3d_q90_mm']:level=1
                    if v>0 and v>=limits['rain_3d_q95_mm']:level=2
                    if pd.notna(r['nasa_rain_3d_mm']):
                        support=bool(r['nasa_rain_3d_mm']>0 and r['nasa_rain_3d_mm']>=q(m,'nasa_rain_3d_mm','q90'))
                elif typ=='water_supply':
                    if r['chirps_rain_14d_mm']<=limits['rain_14d_q10_mm'] and r['soil_7d_m3_m3']<=limits['soil_q20_m3_m3']:
                        level=1
                        if r['chirps_rain_30d_mm']<=limits['rain_30d_q10_mm'] and r['soil_7d_m3_m3']<=limits['soil_q10_m3_m3']:level=2
                elif typ=='wet_environment':
                    if r['nasa_wet_days_7d']>=limits['min_wet_days_7d']:
                        if r['rh_7d_pct']>=limits['rh_q90_pct']:level=1
                        if r['rh_7d_pct']>=limits['rh_q95_pct']:level=2
                elif typ=='warm_anomaly':
                    if r['tmax_3d_c']>=limits['tmax_q90_c']:level=1
                    if r['tmax_3d_c']>=limits['tmax_q95_c']:level=2
                elif typ=='cold_anomaly':
                    if r['tmin_3d_c']<=limits['tmin_q10_c']:level=1
                    if r['tmin_3d_c']<=limits['tmin_q05_c']:level=2
            phase='reference_retrospective' if day<=pd.Timestamp(cfg['reference_end']) else ('validation_frozen' if day.year==2024 else 'held_out_frozen')
            records.append({'date':day.strftime('%Y-%m-%d'),'site_id':cfg['site_id'],'event_type':typ,
             'event_name':NAMES[typ],'screen_level':level,
             'status':'insufficient_data' if missing else ('candidate' if level else 'not_triggered'),
             'indicator_values':{f:None if pd.isna(r[f]) else float(r[f]) for f in required},'thresholds':limits,
             'missing_inputs':missing,'nasa_rainfall_same_date_support':support,
             'rule_version':cfg['rule_version'],'phase':phase,'spatial_support':'regional_background',
             'evidence_status':'weather_screen_only','inspection_topic':TOPICS[typ],
             'is_diagnosis':False,'is_forecast':False})
    return records

def group_events(records):
    events=[]
    for typ in NAMES:
        group=[r for r in records if r['event_type']==typ]
        current=[]
        def finish():
            if not current:return
            peak=max(current,key=lambda r:r['screen_level'])
            events.append({'event_id':f"JB-{typ}-{current[0]['date']}",'event_type':typ,'event_name':NAMES[typ],
              'start_date':current[0]['date'],'end_date':current[-1]['date'],'duration_days':len(current),
              'max_screen_level':peak['screen_level'],'first_max_level_date':peak['date'],
              'inspection_topic':TOPICS[typ],'spatial_support':'regional_background',
              'interpretation':'Retrospective run of consecutive triggered days, not a confirmed disaster episode',
              'left_censored':group.index(current[0])==0 or group[group.index(current[0])-1]['status']=='insufficient_data',
              'right_censored':group.index(current[-1])==len(group)-1 or group[group.index(current[-1])+1]['status']=='insufficient_data'})
        for r in group:
            if r['status']=='candidate':current.append(r)
            else:finish();current=[]
        finish()
    return sorted(events,key=lambda e:(e['start_date'],e['event_type']))

def tabular(records):
    rows=[]
    for r in records:
        row=r.copy()
        for k in ['indicator_values','thresholds','missing_inputs']:row[k]=json.dumps(row[k],ensure_ascii=False,allow_nan=False)
        rows.append(row)
    frame=pd.DataFrame(rows)
    frame['screen_level']=pd.array(frame['screen_level'],dtype='Int64')
    return frame
