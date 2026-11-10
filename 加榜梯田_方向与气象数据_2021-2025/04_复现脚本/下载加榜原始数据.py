"""Jiabang 2021-2025 historical background data. Standard library only."""
from pathlib import Path
import argparse,datetime,json,hashlib,time,urllib.request,urllib.parse,concurrent.futures,zipfile,math

def fetch(url,post=False):
    for attempt in range(3):
        try:
            req=urllib.request.Request(url,data=b'' if post else None,headers={'User-Agent':'Research-data-retrieval/1.0'})
            with urllib.request.urlopen(req,timeout=45) as r:return r.read()
        except Exception:
            if attempt==2:raise
            time.sleep(2*(attempt+1))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parent.parent)
    out=parser.parse_args().output;raw=out/'01_原始数据';meta=out/'05_来源与核验'
    raw.mkdir(parents=True,exist_ok=True);meta.mkdir(exist_ok=True)
    config={'project':'加榜梯田山地农业风险监测与无人机协同巡检','study_area':'加车村附近梯田群初始实验框','period':['2021-01-01','2025-12-31'],
      'source':'User-provided screenshot on 2026-11-10','source_center_lon_lat':[108.586886,25.602516],
      'source_bbox_wsen':[108.56,25.58,108.62,25.625],
      'source_crs':'unspecified; screenshot references Amap','working_crs':'EPSG:4326 interpretation of supplied numeric coordinates, PROVISIONAL',
      'coordinate_transform_applied':False,'crs_verified':False,'flight_ready':False,
      'point':{'point_id':'J01','longitude':108.586886,'latitude':25.602516},
      'approximate_width_km':round(.06*111.32*math.cos(math.radians(25.6025)),3),
      'approximate_height_km':round(.045*111.2,3),
      'notes':['This rectangle is an initial data extraction AOI, not terrace field boundaries.','Only one background request point is used because native weather grids are coarse.','Amap coordinates may be GCJ-02; original CRS is unconfirmed. No unverified conversion has been applied.']}
    (out/'study_area.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf8')
    jobs=[];params=['T2M','T2M_MAX','T2M_MIN','RH2M','PRECTOTCORR','WS2M','ALLSKY_SFC_SW_DWN']
    q={'parameters':','.join(params),'community':'AG','longitude':108.586886,'latitude':25.602516,'start':'20210101','end':'20251231','format':'JSON','time-standard':'LST'}
    jobs.append(('NASA_POWER','nasa_power_J01_2021-2025.json','https://power.larc.nasa.gov/api/temporal/daily/point?'+urllib.parse.urlencode(q)))
    for year in range(2021,2026):
        q={'latitude':25.602516,'longitude':108.586886,'start_date':f'{year}-01-01','end_date':f'{year}-12-31','hourly':'soil_moisture_0_to_7cm,soil_moisture_7_to_28cm,temperature_2m','models':'era5_land','timezone':'Asia/Shanghai','elevation':'nan','cell_selection':'land'}
        jobs.append(('ERA5_Land_via_OpenMeteo',f'era5_land_J01_{year}.json','https://archive-api.open-meteo.com/v1/archive?'+urllib.parse.urlencode(q)))
    results=[]
    old_manifest=meta/'base_download_manifest.json'
    prior={r['file']:r for r in json.loads(old_manifest.read_text(encoding='utf8'))} if old_manifest.exists() else {}
    def get_one(job):
        provider,filename,url=job;p=raw/filename;cached=p.exists();b=p.read_bytes() if cached else fetch(url)
        obj=json.loads(b)
        if obj.get('error'):raise ValueError(obj)
        if provider=='NASA_POWER':assert len(obj['properties']['parameter']['T2M'])==1826
        else:assert len(obj['hourly']['time']) in (8760,8784)
        p.write_bytes(b)
        digest=hashlib.sha256(b).hexdigest()
        previous=prior.get(filename,{})
        retrieved=previous.get('retrieved_at_utc') if cached and previous.get('sha256')==digest else None
        if not cached:retrieved=datetime.datetime.now(datetime.timezone.utc).isoformat()
        return {'provider':provider,'file':filename,'url':url,'bytes':len(b),'sha256':digest,
                'retrieved_at_utc':retrieved,'used_cached_file':cached,
                'checked_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
        for f in concurrent.futures.as_completed([ex.submit(get_one,j) for j in jobs]):
            r=f.result();results.append(r);print('Downloaded',r['file'],r['bytes'],flush=True)
    (meta/'base_download_manifest.json').write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding='utf8')
    base='https://climateserv.servirglobal.net/api/'
    geom={'type':'Polygon','coordinates':[[[108.56,25.58],[108.62,25.58],[108.62,25.625],[108.56,25.625],[108.56,25.58]]],'properties':{}}
    for op in [5,7]:
        statefile=meta/f'chirps_job_{op}.json'
        saved_name='chirps_v2_region_daily_2021-2025.json' if op==5 else 'CHIRPS_v2_加榜初始框_2021-2025.zip'
        saved_path=raw/saved_name
        if statefile.exists() and saved_path.exists():
            saved_state=json.loads(statefile.read_text(encoding='utf8'))
            if saved_state.get('sha256')==hashlib.sha256(saved_path.read_bytes()).hexdigest():
                if op==7 and not (raw/'CHIRPS_v2_加榜初始框_2021-2025.nc').exists():
                    with zipfile.ZipFile(saved_path) as z:
                        matches=[x for x in z.infolist() if x.filename.endswith(('.nc','.nc4'))];assert len(matches)==1
                        (raw/'CHIRPS_v2_加榜初始框_2021-2025.nc').write_bytes(z.read(matches[0]))
                print('Using verified cached CHIRPS',op,flush=True)
                continue
        if statefile.exists():state=json.loads(statefile.read_text(encoding='utf8'))
        else:
            q={'datatype':0,'seasonal_ensemble':'','seasonal_variable':'','begintime':'01/01/2021','endtime':'12/31/2025','intervaltype':0,'operationtype':op,'dateType_Category':'default','isZip_CurrentDataType':'false','geometry':json.dumps(geom,separators=(',',':'))}
            u=base+'submitDataRequest/?'+urllib.parse.urlencode(q)
            state={'params':q,'job_id':json.loads(fetch(u,post=True))[0],'submit_url':u}
            statefile.write_text(json.dumps(state,indent=2),encoding='utf8')
        id=state['job_id'];print('CHIRPS job',op,id,flush=True);deadline=time.monotonic()+600
        while True:
            progress=json.loads(fetch(base+'getDataRequestProgress/?id='+id))[0]
            if progress==100:break
            if progress<0 or time.monotonic()>deadline:raise RuntimeError(f'CHIRPS incomplete: {progress}, job {id}')
            print('CHIRPS progress',op,progress,flush=True);time.sleep(8)
        if op==5:
            u=base+'getDataFromRequest/?id='+id;b=fetch(u);obj=json.loads(b);assert len(obj['data'])==1826
            filename='chirps_v2_region_daily_2021-2025.json';(raw/filename).write_bytes(b)
        else:
            u=base+'getFileForJobID/?id='+id;b=fetch(u);filename='CHIRPS_v2_加榜初始框_2021-2025.zip';p=raw/filename;p.write_bytes(b)
            with zipfile.ZipFile(p) as z:
                matches=[x for x in z.infolist() if x.filename.endswith(('.nc','.nc4'))];assert len(matches)==1
                (raw/'CHIRPS_v2_加榜初始框_2021-2025.nc').write_bytes(z.read(matches[0]))
        state.update({'download_url':u,'file':filename,'bytes':len(b),'sha256':hashlib.sha256(b).hexdigest(),'retrieved_at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
        statefile.write_text(json.dumps(state,ensure_ascii=False,indent=2),encoding='utf8');print('Downloaded CHIRPS',op,len(b),flush=True)
    print('DOWNLOAD COMPLETE. Raw data only; reprocess CSV/XLSX separately.',flush=True)

if __name__=='__main__':main()
