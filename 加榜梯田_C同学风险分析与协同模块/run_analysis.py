"""Rebuild cleaned inputs, reference thresholds, daily screens and summary.
Run from this folder: python run_analysis.py [--pdf]
"""
from pathlib import Path
import argparse,hashlib,json,platform,importlib.metadata
import numpy as np
import pandas as pd
from agri_risk.weather import load_clean,indicators,write_csv,write_json,json_safe
from agri_risk.rules import fit_thresholds,screen,group_events,tabular,NAMES
from agri_risk.integration import fuse,adapt_a_response

ROOT=Path(__file__).resolve().parent

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pdf',action='store_true')
    args=parser.parse_args()
    cfg=json.loads((ROOT/'config.json').read_text(encoding='utf8'))
    out=ROOT/'results';out.mkdir(exist_ok=True)
    clean,quality=load_clean(ROOT/'inputs/weather_daily.csv')
    features=indicators(clean,cfg);thresholds=fit_thresholds(features,cfg)
    records=screen(features,thresholds,cfg);events=group_events(records)
    write_csv(clean,out/'weather_clean.csv');write_csv(features,out/'weather_features.csv')
    write_csv(thresholds,out/'seasonal_thresholds_2021_2023.csv')
    write_csv(tabular(records),out/'risk_daily.csv');write_csv(pd.DataFrame(events),out/'candidate_events.csv')
    write_json([{**r,'date':r['date'].strftime('%Y-%m-%d')} for r in clean.to_dict('records')],out/'weather_daily.json')
    write_json(records,out/'risk_daily.json');write_json(events,out/'candidate_events.json');write_json(quality,out/'data_quality.json')
    # Do not turn unavailable source days into a zero monthly/annual precipitation sum.
    monthly=[]
    for (year,month),group in clean.groupby([clean.date.dt.year,clean.date.dt.month]):
        expected=pd.Period(f'{year}-{month:02}').days_in_month
        row={'month':f'{year}-{month:02}','expected_days':expected}
        for key in ['nasa_precip_mm_day','chirps_selected_precip_mm_day']:
            row[key.replace('_mm_day','_mm_sum')]=group[key].sum(min_count=expected)
        monthly.append(row)
    monthly=pd.DataFrame(monthly);write_csv(monthly,out/'monthly_precipitation.csv')
    annual=[]
    for year,group in clean.groupby(clean.date.dt.year):
        n=366 if pd.Timestamp(year,12,31).dayofyear==366 else 365
        annual.append({'year':int(year),'days':len(group),
          'chirps_mm':group.chirps_selected_precip_mm_day.sum(min_count=n),
          'nasa_mm':group.nasa_precip_mm_day.sum(min_count=n)})
    counts=[]
    for year in range(2021,2026):
        for typ,name in NAMES.items():
            rows=[r for r in records if r['date'].startswith(str(year)) and r['event_type']==typ]
            valid=sum(r['status']!='insufficient_data' for r in rows)
            active=sum(r['status']=='candidate' for r in rows)
            counts.append({'year':year,'event_type':typ,'event_name':name,'valid_days':valid,
             'candidate_days':active,'level2_days':sum(r['screen_level']==2 for r in rows),
             'candidate_day_percent':100*active/valid if valid else None,
             'episodes_starting_in_year':sum(e['event_type']==typ and e['start_date'].startswith(str(year)) for e in events),
             'interpretation':'Screened days may overlap across types; not disease incidence or model accuracy'})
    write_csv(pd.DataFrame(counts),out/'annual_screening_counts.csv')
    # All dates are retrospective; reference-year statistics are not held-out tests.
    monthly_correlation=monthly['nasa_precip_mm_sum'].corr(monthly['chirps_selected_precip_mm_sum'])
    rain_days=[r for r in records if r['event_type']=='rainfall' and r['status']=='candidate' and r['date']>='2024-01-01']
    valid_support=[r for r in rain_days if r['nasa_rainfall_same_date_support'] is not None]
    peak=features.loc[features[features.date>='2024-01-01'].chirps_rain_3d_mm.idxmax()]
    peak_day=peak.date.strftime('%Y-%m-%d')
    day_records=[r for r in records if r['date']==peak_day]
    summary={'site_id':cfg['site_id'],'period':[quality['first_date'],quality['last_date']],
      'reference_period':[cfg['reference_start'],cfg['reference_end']],
      'thresholds_frozen_for':'2024-2025; no outcome labels and no accuracy evaluation',
      'annual_precipitation':annual,'annual_screening':counts,
      'daily_precip_correlation':clean.nasa_precip_mm_day.corr(clean.chirps_selected_precip_mm_day),
      'monthly_precip_correlation':monthly_correlation,
      'monthly_precip_pairs':len(monthly.dropna()),
      'rainfall_candidate_days_2024_2025':len(rain_days),
      'nasa_supported_rainfall_days_2024_2025':sum(bool(r['nasa_rainfall_same_date_support']) for r in valid_support),
      'rainfall_support_evaluable_days':len(valid_support),
      'case_peak_rain_3d':{'date':peak_day,'chirps_3d_mm':peak.chirps_rain_3d_mm,
        'nasa_3d_mm':peak.nasa_rain_3d_mm,'candidates':[r['event_name'] for r in day_records if r['status']=='candidate']},
      'real_image_observations_available':0,'real_verified_field_boundaries_available':0,
      'real_field_inspection_requests':0,'accuracy_metrics':None,
      'scope':'Regional historical screening, not flood/disease diagnosis, not a forecast',
      'time_alignment':'Source dates not harmonized; support is an approximate cross-product comparison, not ground-truth validation'}
    write_json(summary,out/'summary.json')
    # Actual deliverable stays region-level until real dated field evidence arrives.
    regional=fuse({'as_of_date':peak_day},records,cfg)
    write_json(regional,out/'regional_case_without_field_evidence.json')
    write_json([],out/'field_inspection_requests_real.json')
    # A tiny explicitly synthetic contract example tests the A-C-B wiring only.
    template=json.loads((ROOT/'examples/fusion_synthetic_template.json').read_text(encoding='utf8'))
    template['as_of_date']=peak_day
    template['a_observations'][0]['metadata']['captured_at']=peak_day+'T10:00:00+08:00'
    template['plots']['features'][0]['properties']['drainage_problem']['observed_at']=peak_day+'T11:00:00+08:00'
    write_json(template,ROOT/'examples/fusion_synthetic_request.json')
    write_json(fuse(template,records,cfg),ROOT/'examples/fusion_synthetic_result.json')
    # Auditing A's saved LoveDA examples does not make them Jiabang field observations.
    old=json.loads((ROOT/'reference/a_saved_analysis_summary.json').read_text(encoding='utf8'))
    audit=[]
    for row in old:
        stats=row.get('index_statistics',{}).get('VARI',{})
        audit.append({'name':row['name'],'VARI_mean':stats.get('mean'),'VARI_max':stats.get('max'),
          'within_current_code_clipped_range':all(-1<=stats[k]<=1 for k in ['mean','min','max']),
          'eligible_for_jiabang_fusion':False,
          'reasons':['Saved LoveDA example, not confirmed Jiabang imagery','No capture date or verified field association',
                     'Saved VARI statistics inconsistent with current clipped implementation; regenerate with current code']})
    write_json(audit,out/'A_reference_compatibility_audit.json')
    versions={'python':platform.python_version(),**{p:importlib.metadata.version(p) for p in ['pandas','numpy','matplotlib','fastapi','shapely']}}
    write_json(versions,out/'runtime_versions.json')
    from agri_risk.report import make_report
    make_report(ROOT,summary,cfg,clean,features,records,thresholds,args.pdf)
    print(json.dumps(json_safe(summary),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
