from pathlib import Path
import copy,json
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from agri_risk.weather import load_clean,indicators
from agri_risk.rules import fit_thresholds,screen,group_events
from agri_risk.integration import adapt_a_response,fuse
from agri_risk.api import app

ROOT=Path(__file__).resolve().parents[1]
CFG=json.loads((ROOT/'config.json').read_text(encoding='utf8'))

@pytest.fixture(scope='module')
def base():return load_clean(ROOT/'inputs/weather_daily.csv')[0]
@pytest.fixture(scope='module')
def feat(base):return indicators(base,CFG)
@pytest.fixture(scope='module')
def thresholds(feat):return fit_thresholds(feat,CFG)
@pytest.fixture(scope='module')
def records():return json.loads((ROOT/'results/risk_daily.json').read_text(encoding='utf8'))
@pytest.fixture
def demo():return json.loads((ROOT/'examples/fusion_synthetic_request.json').read_text(encoding='utf8'))

def test_full_dates_and_quality(base):
    assert len(base)==1826
    assert base.date.iloc[0]==pd.Timestamp('2021-01-01')
    assert base.date.iloc[-1]==pd.Timestamp('2025-12-31')
    assert not base.isna().any().any()

def test_duplicate_dates_fail(base,tmp_path):
    path=tmp_path/'duplicate.csv';pd.concat([base.iloc[:3],base.iloc[:1]]).to_csv(path,index=False)
    with pytest.raises(ValueError,match='Duplicate'):load_clean(path)

def test_missing_calendar_day_not_hidden(base,tmp_path):
    subset=base.iloc[:40].drop(index=8);path=tmp_path/'gap.csv';subset.to_csv(path,index=False)
    cleaned,q=load_clean(path);result=indicators(cleaned,CFG)
    assert q['inserted_missing_dates']==['2021-01-09']
    assert pd.isna(result.loc[14,'chirps_rain_7d_mm'])
    assert pd.notna(result.loc[15,'chirps_rain_7d_mm'])

def test_invalid_fraction_not_zero(base,tmp_path):
    sample=base.iloc[:40].copy();sample.loc[9,'nasa_rh_pct']=101
    path=tmp_path/'bad.csv';sample.to_csv(path,index=False)
    cleaned,q=load_clean(path)
    assert pd.isna(cleaned.loc[9,'nasa_rh_pct'])
    assert q['invalid_values'][0]['reason']=='outside_physical_range'

def test_window_and_zero_rain(base):
    sample=base.iloc[:35].copy();sample['chirps_selected_precip_mm_day']=np.arange(1,36)
    r=indicators(sample,CFG)
    assert r.loc[6,'chirps_rain_7d_mm']==28
    assert r.loc[29,'chirps_rain_30d_mm']==465
    assert r.loc[:28,'chirps_rain_30d_mm'].isna().all()
    sample['chirps_selected_precip_mm_day']=0
    assert indicators(sample,CFG).loc[29,'chirps_rain_30d_mm']==0

def test_calibration_excludes_later_years(feat,thresholds):
    modified=feat.copy();modified.loc[modified.date>='2024-01-01','chirps_rain_3d_mm']=1e6
    pd.testing.assert_frame_equal(fit_thresholds(modified,CFG),thresholds)

def test_future_values_do_not_change_past(base,thresholds):
    edited=base.copy();edited.loc[edited.date>'2025-06-24','nasa_rh_pct']=1
    a=screen(indicators(base,CFG),thresholds,CFG)
    b=screen(indicators(edited,CFG),thresholds,CFG)
    assert [r for r in a if r['date']<='2025-06-24']==[r for r in b if r['date']<='2025-06-24']

def test_unknown_not_safe(feat,thresholds):
    sample=feat.iloc[:1].copy();rows=screen(sample,thresholds,CFG)
    assert all(r['status']=='insufficient_data' and r['screen_level'] is None for r in rows)

def test_zero_rain_never_high_even_zero_threshold(feat,thresholds):
    sample=feat.iloc[100:101].copy();sample['chirps_rain_3d_mm']=0
    t=thresholds.copy();t.loc[t.feature=='chirps_rain_3d_mm',['q90','q95']]=0
    assert next(r for r in screen(sample,t,CFG) if r['event_type']=='rainfall')['screen_level']==0

def test_event_gaps_and_censoring():
    rows=[]
    for i,status in enumerate(['insufficient_data','candidate','candidate','insufficient_data','candidate','not_triggered']):
        rows.append({'date':f'2025-01-0{i+1}','event_type':'rainfall','status':status,'screen_level':1 if status=='candidate' else None})
    events=group_events(rows)
    assert len(events)==2 and events[0]['duration_days']==2
    assert events[0]['left_censored'] and events[0]['right_censored']
    assert events[1]['left_censored'] and not events[1]['right_censored']

def test_no_fields_no_fabricated_tasks(records):
    result=fuse({'as_of_date':'2025-06-24'},records,CFG)
    assert result['regional_candidates'] and not result['inspection_requests']
    assert result['missing_inputs']

def test_synthetic_contract_and_non_executable(records,demo):
    result=fuse(demo,records,CFG)
    assert result['mode']=='synthetic_contract_demo'
    assert len(result['inspection_requests'])==1
    task=result['inspection_requests'][0]
    assert task['flight_ready'] is False and task['chemical_or_fertilizer_dose'] is None

def test_synthetic_blocked_by_default(records,demo):
    demo['demo_mode']=False
    with pytest.raises(ValueError,match='Synthetic'):fuse(demo,records,CFG)

@pytest.mark.parametrize('field,value',[
 ('captured_at',None),('captured_at','2025-06-24T10:00:00'),
 ('plot_association_verified',False),('image_scope','whole_scene'),('site_id','other_site')])
def test_a_metadata_contract(demo,field,value):
    item=demo['a_observations'][0];item['metadata'][field]=value
    with pytest.raises(ValueError):adapt_a_response(item['response'],item['metadata'],CFG,True)

def test_old_vari_rejected(demo):
    item=demo['a_observations'][0];item['response']['data']['vegetation']['index_statistics']['VARI']['mean']=98.4
    with pytest.raises(ValueError,match='VARI'):adapt_a_response(item['response'],item['metadata'],CFG,True)

@pytest.mark.parametrize('value',[-.1,1.1,float('nan'),True])
def test_fraction_validation(demo,value):
    item=demo['a_observations'][0];item['response']['data']['anomaly']['low_vigor_fraction']=value
    with pytest.raises(ValueError):adapt_a_response(item['response'],item['metadata'],CFG,True)

@pytest.mark.parametrize('stamp',['2025-06-25T01:00:00+08:00','2025-06-01T01:00:00+08:00'])
def test_future_or_stale_images_quarantined(records,demo,stamp):
    demo['a_observations'][0]['metadata']['captured_at']=stamp
    result=fuse(demo,records,CFG)
    assert not result['accepted_image_observations'] and result['quarantined_observations']

def test_latest_normal_does_not_use_old_image_anomaly(records,demo):
    # Pick a date with no regional candidate; remove local simulated drainage evidence.
    normal_date=next(d for d in sorted({r['date'] for r in records if r['date']>='2024-01-01'})
      if not any(r['status']=='candidate' for r in records if r['date']==d))
    demo['as_of_date']=normal_date
    demo['plots']['features'][0]['properties'].pop('drainage_problem')
    first=demo['a_observations'][0];first['metadata']['captured_at']=normal_date+'T08:00:00+08:00'
    new=copy.deepcopy(first);new['metadata']['observation_id']='SYNTHETIC_NEW';new['metadata']['captured_at']=normal_date+'T12:00:00+08:00'
    new['response']['data']['anomaly']['low_vigor_fraction']=0
    demo['a_observations'].append(new)
    assert fuse(demo,records,CFG)['inspection_requests']==[]

def test_unknown_weather_date_not_replaced(records):
    with pytest.raises(ValueError,match='No weather'):fuse({'as_of_date':'2026-11-10'},records,CFG)

def test_unverified_crs_rejected(records,demo):
    demo['plots']['features'][0]['properties']['crs_verified']=False
    with pytest.raises(ValueError,match='verified'):fuse(demo,records,CFG)

def test_api_contracts(demo):
    client=TestClient(app)
    assert client.get('/health').json()['success'] is True
    response=client.get('/risk/daily',params={'date':'2025-06-24'})
    assert response.status_code==200 and len(response.json()['data'])==5
    response=client.get('/risk/daily',params={'date':'2026-11-10'})
    assert response.status_code==404 and response.json()['detail']['success'] is False
    assert client.get('/weather/daily',params={'start':'bad'}).status_code==400
    assert client.post('/integration/fuse',json=demo).json()['data']['inspection_requests']
    assert client.post('/integration/fuse',json={'as_of_date':'2026-11-10'}).status_code==400
