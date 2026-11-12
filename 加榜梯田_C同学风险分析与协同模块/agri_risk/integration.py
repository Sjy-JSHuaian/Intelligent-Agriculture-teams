"""Adapt A's current JSON contract; propose inspection requests for B.
No pixel coordinates are interpreted as geographic coordinates.
"""
from datetime import datetime,date,timedelta,timezone
import math
from shapely.geometry import shape,mapping

CST=timezone(timedelta(hours=8))

def finite_number(value,name,low=None,high=None):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value):
        raise ValueError(f'{name} must be a finite number')
    if low is not None and value<low:raise ValueError(f'{name} < {low}')
    if high is not None and value>high:raise ValueError(f'{name} > {high}')
    return float(value)

def parse_capture(value):
    if not isinstance(value,str):raise ValueError('captured_at is required; API timestamp is processing time')
    stamp=datetime.fromisoformat(value.replace('Z','+00:00'))
    if stamp.tzinfo is None:raise ValueError('captured_at needs an explicit timezone')
    return stamp.astimezone(CST)

def adapt_a_response(response,metadata,cfg,allow_synthetic=False):
    if not isinstance(response,dict) or response.get('success') is not True:
        raise ValueError('A response success must be true')
    data=response.get('data')
    if not isinstance(data,dict):raise ValueError('A response data must be an object')
    anomaly=data.get('anomaly',data)
    if not isinstance(anomaly,dict) or 'low_vigor_fraction' not in anomaly:
        raise ValueError('Use A /analyze or /anomaly output; low_vigor_fraction missing')
    low=finite_number(anomaly['low_vigor_fraction'],'low_vigor_fraction',0,1)
    bare=finite_number(anomaly['bare_land_fraction'],'bare_land_fraction',0,1) if 'bare_land_fraction' in anomaly else None
    veg=data.get('vegetation',{})
    if not isinstance(veg,dict):raise ValueError('vegetation must be an object')
    for key in ['vegetation_fraction_gli','vegetation_fraction_vari']:
        if key in veg:finite_number(veg[key],key,0,1)
    if 'vegetation_fraction_gli' in veg and low>veg['vegetation_fraction_gli']+.0002:
        raise ValueError('low_vigor_fraction exceeds GLI vegetation fraction; inconsistent outputs')
    stats=veg.get('index_statistics',data.get('index_statistics',{}))
    for index,lo,hi in [('VARI',-1,1),('GLI',-1,1),('NDI',-1,1),('ExG',-2,2)]:
        if index in stats:
            entry=stats[index]
            for k in ['min','mean','max']:
                if k in entry:finite_number(entry[k],f'{index}.{k}',lo-1e-5,hi+1e-5)
            if all(k in entry for k in ['min','mean','max']) and not entry['min']<=entry['mean']<=entry['max']:
                raise ValueError(f'{index}: min/mean/max inconsistent')
    distribution=data.get('segmentation',{}).get('class_distribution')
    if distribution is not None:
        if set(distribution)!={'background','building','road','water','barren','forest','agriculture'}:
            raise ValueError('Unknown segmentation class mapping')
        for k,v in distribution.items():finite_number(v,f'class_distribution.{k}',0,1)
        if abs(sum(distribution.values())-1)>.002:raise ValueError('Class fractions do not sum to one')
    if metadata.get('is_synthetic',False) and not allow_synthetic:raise ValueError('Synthetic image is blocked outside explicit demo mode')
    for k in ['observation_id','plot_id','source_reference']:
        if not isinstance(metadata.get(k),str) or not metadata[k].strip():raise ValueError(f'{k} required')
    if metadata.get('site_id')!=cfg['site_id']:raise ValueError('Image site_id does not match Jiabang site')
    if metadata.get('location_verified') is not True or metadata.get('plot_association_verified') is not True:
        raise ValueError('Image-to-plot association must be verified; no guessed location')
    if metadata.get('image_scope')!='single_plot_crop':
        raise ValueError('Whole-image statistics require a verified single-plot crop before plot-level use')
    captured=parse_capture(metadata.get('captured_at'))
    return {'observation_id':metadata['observation_id'],'plot_id':metadata['plot_id'],
      'captured_at':captured.isoformat(),'capture_date':captured.date().isoformat(),
      'low_vigor_fraction':low,'bare_land_fraction':bare,'fraction_denominator':'all_pixels_in_verified_plot_crop',
      'mask_paths':{k:anomaly.get(k) for k in ['low_vigor_mask','bare_land_mask']},
      'segmentation_overlay':data.get('segmentation',{}).get('overlay'),
      'evidence_status':'rgb_model_candidate_not_diagnosis','is_synthetic':metadata.get('is_synthetic',False),
      'source_reference':metadata['source_reference']}

def validate_plots(collection,cfg,allow_synthetic=False):
    if collection.get('type')!='FeatureCollection':raise ValueError('B plots must be a GeoJSON FeatureCollection')
    plots={}
    for feature in collection.get('features',[]):
        prop=feature.get('properties',{});pid=prop.get('plot_id')
        if not isinstance(pid,str) or not pid.strip() or pid in plots:raise ValueError('Missing/duplicate plot_id')
        if prop.get('site_id')!=cfg['site_id']:raise ValueError(f'{pid}: wrong site_id')
        if prop.get('crs')!='EPSG:4326' or prop.get('crs_verified') is not True or prop.get('boundary_verified') is not True:
            raise ValueError(f'{pid}: boundary and WGS84 must be verified')
        if not prop.get('boundary_source'):raise ValueError(f'{pid}: boundary_source required')
        if prop.get('is_synthetic',False) and not allow_synthetic:raise ValueError('Synthetic plot blocked outside demo mode')
        geom=shape(feature.get('geometry'))
        if geom.geom_type not in ['Polygon','MultiPolygon'] or geom.is_empty or not geom.is_valid or geom.has_z:
            raise ValueError(f'{pid}: invalid polygon')
        west,south,east,north=geom.bounds
        if not (108.56<=west<east<=108.62 and 25.58<=south<north<=25.625):
            raise ValueError(f'{pid}: outside the current initial AOI; confirm study boundary before extension')
        plots[pid]={'properties':prop,'geometry':mapping(geom)}
    return plots

def _local_evidence(prop,key,as_of,max_age=30):
    evidence=prop.get(key,{})
    if not isinstance(evidence,dict) or evidence.get('verified') is not True or not evidence.get('source_reference'):
        return False
    observed=parse_capture(evidence.get('observed_at')).date()
    age=(as_of-observed).days
    return 0<=age<=max_age and evidence.get('present') is True

def fuse(payload,records,cfg):
    as_of=date.fromisoformat(payload['as_of_date'])
    demo=payload.get('demo_mode',False)
    if not isinstance(demo,bool):raise ValueError('demo_mode must be boolean')
    dates={r['date'] for r in records}
    if as_of.isoformat() not in dates:
        raise ValueError('No weather for this date; older weather is not substituted for a live decision')
    day=[r for r in records if r['date']==as_of.isoformat()]
    active=[r for r in day if r['status']=='candidate']
    plots=validate_plots(payload.get('plots',{'type':'FeatureCollection','features':[]}),cfg,demo)
    accepted=[];quarantine=[]
    seen=set()
    for item in payload.get('a_observations',[]):
        try:
            observation=adapt_a_response(item['response'],item['metadata'],cfg,demo)
            if observation['observation_id'] in seen:raise ValueError('Duplicate observation_id')
            seen.add(observation['observation_id'])
            age=(as_of-date.fromisoformat(observation['capture_date'])).days
            if age<0:raise ValueError('Image captured after the review date')
            if age>cfg['max_image_age_days']:raise ValueError('Image too old for the configured evidence window')
            if observation['plot_id'] not in plots:raise ValueError('Image plot_id not found in verified B polygons')
            accepted.append(observation)
        except (ValueError,KeyError,TypeError) as ex:
            quarantine.append({'observation_id':item.get('metadata',{}).get('observation_id'),'reason':str(ex)})
    requests=[];not_localized=[]
    for pid,plot in plots.items():
        prop=plot['properties'];images=[r for r in accepted if r['plot_id']==pid]
        # Only the latest accepted capture is used; an old anomaly must not override a newer image.
        images=sorted(images,key=lambda x:(x['captured_at'],x['observation_id']))
        image=images[-1] if images else None
        low=image is not None and image['low_vigor_fraction']>0
        drainage=_local_evidence(prop,'drainage_problem',as_of)
        irrigation=_local_evidence(prop,'irrigation_problem',as_of)
        selected=[]
        for r in active:
            local=(r['event_type']=='rainfall' and drainage) or (r['event_type']=='water_supply' and irrigation) or low
            if local:selected.append(r)
        if not selected and not low:
            not_localized.append(pid);continue
        # This is a transparent inspection ordering rule, not a disease probability.
        priority='P1' if selected and (drainage or irrigation or low) and any(r['screen_level']==2 for r in selected) else 'P2'
        why=[]
        if low:why.append('A latest verified plot crop contains low-greenness candidate pixels; cause unknown')
        if drainage:why.append('Recent verified local drainage problem')
        if irrigation:why.append('Recent verified local irrigation-supply problem')
        why += [r['event_name']+': '+r['inspection_topic'] for r in selected]
        requests.append({'request_id':f'JB-{as_of}-{pid}','plot_id':pid,'as_of_date':str(as_of),
          'priority':priority,'priority_meaning':'Inspection ordering only, not probability or agronomic severity',
          'task_type':'inspection_only','geometry':plot['geometry'],'geometry_crs':'EPSG:4326',
          'weather_event_types':[r['event_type'] for r in selected],
          'observation_id':image['observation_id'] if image else None,'reasons':why,
          'evidence_status':'weather_and_local_candidate' if selected else 'image_candidate_only',
          'unknowns':['crop_growth_stage','cause_of_image_anomaly','agronomic_validation'],
          'execution_status':'awaiting_B_route_and_site_checks','flight_ready':False,
          'chemical_or_fertilizer_dose':None,'is_synthetic_demo':demo})
    return {'mode':'synthetic_contract_demo' if demo else 'historical_evidence_review',
      'as_of_date':str(as_of),'regional_candidates':active,'accepted_image_observations':accepted,
      'quarantined_observations':quarantine,'inspection_requests':requests,
      'plots_without_local_evidence':not_localized,
      'missing_inputs':['verified_plot_boundaries','dated_geolocated_image_or_local_evidence'] if not plots else [],
      'is_forecast':False,'live_data_availability_verified':False,
      'note':'Historical daily review; coarse weather is shared background. No executable flight or chemical prescription.'}
