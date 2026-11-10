"""Build Jiabang tables and QC from saved provider responses.
Dependencies: pandas, numpy, netCDF4. Run with --output PACKAGE_DIRECTORY.
"""
from pathlib import Path
import argparse, calendar, datetime, json, sys

local_deps = Path(__file__).resolve().parent.parent / 'weather_deps'
if local_deps.exists():
    sys.path.insert(0, str(local_deps))
import numpy as np
import pandas as pd
import netCDF4

FIELDS = {'T2M':'tmean_c','T2M_MAX':'tmax_c','T2M_MIN':'tmin_c','RH2M':'rh_pct',
          'PRECTOTCORR':'precip_mm_day','WS2M':'wind2m_ms','ALLSKY_SFC_SW_DWN':'solar_mj_m2_day'}
ERA_FIELDS = {'soil_moisture_0_to_7cm':'sm_0_7_m3_m3',
              'soil_moisture_7_to_28cm':'sm_7_28_m3_m3','temperature_2m':'temperature_c'}

def write_csv(frame, path, hourly=False):
    frame.to_csv(path, index=False, encoding='utf-8-sig',
                 date_format='%Y-%m-%dT%H:%M' if hourly else '%Y-%m-%d', float_format='%.8f')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent.parent)
    parser.add_argument('--payload',type=Path)
    args=parser.parse_args(); out=args.output
    raw=out/'01_原始数据'; tab=out/'02_整理后表格'; geo=out/'03_空间范围'; meta=out/'05_来源与核验'
    for d in [tab,geo,meta,out/'04_复现脚本']: d.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((out/'study_area.json').read_text(encoding='utf8'))
    obj=json.loads((raw/'nasa_power_J01_2021-2025.json').read_text(encoding='utf8'))
    nasa=pd.DataFrame(obj['properties']['parameter']).rename(columns=FIELDS).replace(obj['header']['fill_value'],np.nan)
    nasa.index=pd.to_datetime(nasa.index,format='%Y%m%d'); nasa.index.name='date'
    nasa=nasa.reset_index().sort_values('date').reset_index(drop=True)
    nasa.insert(0,'point_id','J01'); nasa['day_basis']='LST'
    nasa['requested_longitude']=cfg['point']['longitude']; nasa['requested_latitude']=cfg['point']['latitude']
    era_parts=[]; grid_records=[]
    for year in range(2021,2026):
        ob=json.loads((raw/f'era5_land_J01_{year}.json').read_text(encoding='utf8'))
        e=pd.DataFrame(ob['hourly']).rename(columns={'time':'time_local',**ERA_FIELDS})
        e['time_local']=pd.to_datetime(e['time_local']); e.insert(0,'point_id','J01')
        e['grid_latitude']=ob['latitude']; e['grid_longitude']=ob['longitude']; e['timezone']=ob['timezone']
        era_parts.append(e)
        grid_records.append({'year':year,'point_id':'J01',**cfg['point'],
          'era5_grid_latitude':ob['latitude'],'era5_grid_longitude':ob['longitude'],
          'era5_grid_elevation_m':ob['elevation'],'requested_crs_verified':False,
          'elevation_downscaling':False})
    era=pd.concat(era_parts,ignore_index=True).sort_values('time_local').reset_index(drop=True)
    values=list(ERA_FIELDS.values()); grouped=era.groupby(era.time_local.dt.normalize())
    daily=grouped[values].mean(); counts=grouped[values].count()
    for c in values:
        daily.loc[counts[c]!=24,c]=np.nan
        daily[c+'_valid_hours']=counts[c]
    daily.index.name='date'
    daily=daily.rename(columns={c:c+'_daily_mean' for c in values}).reset_index()
    daily.insert(0,'point_id','J01'); daily['day_basis']='Asia/Shanghai'
    chirps=json.loads((raw/'chirps_v2_region_daily_2021-2025.json').read_text(encoding='utf8'))
    ch=pd.DataFrame([{'date':datetime.date(r['year'],r['month'],r['day']),
       'precip_selected_cell_mean_mm_day':r['value']['avg']} for r in chirps['data']])
    ch['date']=pd.to_datetime(ch.date); ch=ch.sort_values('date').reset_index(drop=True)
    ch.loc[ch.precip_selected_cell_mean_mm_day<0,'precip_selected_cell_mean_mm_day']=np.nan
    ch['day_basis']='CHIRPS_product_date'; ch['area_weighted']=False

    ds=netCDF4.Dataset('chirps.nc',memory=(raw/'CHIRPS_v2_加榜初始框_2021-2025.nc').read_bytes())
    lat=np.asarray(ds['latitude'][:]); lon=np.asarray(ds['longitude'][:])
    rainfall=np.ma.filled(ds['precipitation_amount'][:],np.nan)
    times=netCDF4.num2date(ds['time'][:],ds['time'].units,calendar=ds['time'].calendar)
    dates=pd.to_datetime([t.strftime('%Y-%m-%d') for t in times])
    active=np.isfinite(rainfall[0]); fixed_mask=bool(np.all(np.isnan(rainfall)==np.isnan(rainfall[0])))
    assert fixed_mask, 'Time-varying missing CHIRPS cells need investigation'
    diff=float(np.max(np.abs(np.nanmean(rainfall,axis=(1,2))-ch.precip_selected_cell_mean_mm_day.to_numpy())))
    assert diff<1e-5, diff
    cell_rows=[]; cell_mapping=[]
    for n,(i,j) in enumerate(zip(*np.where(active)),start=1):
        cid=f'CH{n:02d}'
        m={'cell_id':cid,'longitude':float(lon[j]),'latitude':float(lat[i]),'nominal_resolution_degrees':0.05}
        cell_mapping.append(m)
        for date,value in zip(dates,rainfall[:,i,j]):
            cell_rows.append({**m,'date':date,'precip_mm_day':float(value),'day_basis':'CHIRPS_product_date'})
    grid_qc={'shape':list(rainfall.shape),'valid_cells_each_day':int(active.sum()),
      'fixed_masked_cells_each_day':int((~active).sum()),'fixed_mask_all_dates':fixed_mask,
      'valid_cell_coordinates':cell_mapping,'max_abs_grid_mean_vs_API_mm':diff,
      'temporal_missing_values_within_selected_cells':int(np.isnan(rainfall[:,active]).sum()),
      'value_range_mm':[float(np.nanmin(rainfall)),float(np.nanmax(rainfall))],
      'time_bounds_variable_missing':ds['time'].getncattr('bounds') not in ds.variables,
      'interpretation':'ClimateSERV-selected cell arithmetic mean; not exact polygon area-weighted precipitation. With one valid cell this equals that cell value. Fixed outer masked cells must not be filled with zero.'}
    ds.close()
    ch['valid_cells']=int(active.sum())
    base=nasa[['date']+list(FIELDS.values())].rename(columns={c:'nasa_'+c for c in FIELDS.values()})
    base=base.merge(daily[['date']+[c+'_daily_mean' for c in values]].rename(columns={c+'_daily_mean':'era5_'+c+'_daily_mean' for c in values}),on='date',validate='one_to_one')
    base=base.merge(ch[['date','precip_selected_cell_mean_mm_day']].rename(columns={'precip_selected_cell_mean_mm_day':'chirps_selected_precip_mm_day'}),on='date',validate='one_to_one')
    write_csv(nasa,tab/'NASA_POWER_J01_区域背景日数据.csv')
    write_csv(era,tab/'ERA5_Land_J01_区域背景逐小时.csv',hourly=True)
    write_csv(daily,tab/'ERA5_Land_J01_区域背景日均.csv')
    write_csv(ch,tab/'CHIRPS_v2_初始框选中格点日降水.csv')
    write_csv(pd.DataFrame(cell_rows),tab/'CHIRPS_v2_有效格点日降水.csv')
    write_csv(base,tab/'J01_入门合并日表.csv')
    write_csv(pd.DataFrame(grid_records),geo/'J01请求坐标与ERA5实际网格.csv')
    write_csv(pd.DataFrame(cell_mapping),geo/'CHIRPS有效格点坐标.csv')
    monthly=[]
    for (year,month),g in base.groupby([base.date.dt.year,base.date.dt.month]):
        exp=calendar.monthrange(year,month)[1]
        r={'month':f'{year}-{month:02}','expected_days':exp}
        for c in base.columns[1:]:
            additive=c in ['nasa_precip_mm_day','nasa_solar_mj_m2_day','chirps_selected_precip_mm_day']
            r[c+('_sum' if additive else '_mean')]=g[c].sum(min_count=exp) if additive else (g[c].mean() if g[c].count()==exp else np.nan)
        monthly.append(r)
    write_csv(pd.DataFrame(monthly),tab/'J01_月汇总_不同日界线.csv')

    w,s,e,n=cfg['source_bbox_wsen']; flags={'crs_verified':False,'coordinate_transform_applied':False,
      'working_crs_assumption':'Supplied coordinates provisionally interpreted as WGS84','flight_ready':False}
    features=[{'type':'Feature','properties':{'id':'jiabang_initial_aoi','role':'initial_extraction_rectangle','not_field_or_administrative_boundary':True,**flags},
      'geometry':{'type':'Polygon','coordinates':[[[w,s],[e,s],[e,n],[w,n],[w,s]]]}},
      {'type':'Feature','properties':{'id':'J01','role':'regional_background_request',**flags},
      'geometry':{'type':'Point','coordinates':cfg['source_center_lon_lat']}}]
    (geo/'加榜初始框与背景点_坐标系待核准.geojson').write_text(json.dumps({'type':'FeatureCollection','features':features},ensure_ascii=False,indent=2),encoding='utf8')
    native_features=[{'type':'Feature','properties':{'id':'ERA5_J01','role':'provider_returned_grid','source':'Open-Meteo ERA5-Land'},'geometry':{'type':'Point','coordinates':[grid_records[0]['era5_grid_longitude'],grid_records[0]['era5_grid_latitude']]}}]
    native_features += [{'type':'Feature','properties':{'id':m['cell_id'],'role':'provider_selected_grid','source':'ClimateSERV CHIRPS v2'},'geometry':{'type':'Point','coordinates':[m['longitude'],m['latitude']]}} for m in cell_mapping]
    (geo/'数据服务实际网格中心_WGS84.geojson').write_text(json.dumps({'type':'FeatureCollection','features':native_features},ensure_ascii=False,indent=2),encoding='utf8')

    expected=pd.date_range('2021-01-01','2025-12-31')
    expected_hours=pd.date_range('2021-01-01','2025-12-31 23:00',freq='h')
    for frame in [nasa,daily,ch,base]:
        assert frame.date.tolist()==expected.tolist()
        assert not frame.date.duplicated().any()
    assert dates.tolist()==expected.tolist()
    assert era.time_local.tolist()==expected_hours.tolist()
    assert len(era)==43824 and len(base)==1826
    assert not base.isna().any().any() and not era[values].isna().any().any()
    assert (counts==24).all().all()
    assert nasa.rh_pct.between(0,100).all()
    assert ((nasa.tmin_c<=nasa.tmean_c)&(nasa.tmean_c<=nasa.tmax_c)).all()
    assert (nasa[['precip_mm_day','wind2m_ms','solar_mj_m2_day']]>=0).all().all()
    assert era.sm_0_7_m3_m3.between(0,1).all() and era.sm_7_28_m3_m3.between(0,1).all()
    assert np.nanmin(rainfall)>=0
    quality={'period':['2021-01-01','2025-12-31'],'background_request_points':1,
      'rows':{'nasa_daily':len(nasa),'era5_hourly':len(era),'era5_daily':len(daily),'chirps_daily':len(ch),'merged_daily':len(base)},
      'missing_merged_values':int(base.isna().sum().sum()),'missing_era5_hourly_values':int(era[values].isna().sum().sum()),
      'date_coverage_check':'passed','duplicate_time_check':'passed','24_valid_hours_each_era5_day':True,
      'numeric_ranges':{c:[float(base[c].min()),float(base[c].max())] for c in base.columns[1:]},
      'era5_grid':grid_records[0],'chirps_grid':grid_qc,
      'day_alignment':'Date labels only; NASA LST, ERA5 Asia/Shanghai, CHIRPS product date are not normalized to a common day boundary.'}
    for name,value in [('数据质量核验.json',quality),('CHIRPS网格核验.json',grid_qc)]:
        (meta/name).write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf8')
    if args.payload:
        payload={'rows':json.loads(base.to_json(orient='records',date_format='iso')),'quality':quality,'config':cfg}
        args.payload.write_text(json.dumps(payload,ensure_ascii=False),encoding='utf8')
    print(json.dumps(quality,ensure_ascii=True,indent=2))

if __name__=='__main__': main()
