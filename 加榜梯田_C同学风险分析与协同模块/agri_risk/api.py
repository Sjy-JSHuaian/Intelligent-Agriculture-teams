from datetime import datetime,timezone
from pathlib import Path
import json
from fastapi import FastAPI,HTTPException,Query
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from .integration import fuse

ROOT=Path(__file__).resolve().parents[1]
app=FastAPI(title='加榜梯田 C 气象风险筛查与协同接口',version='1.0.0')

def ok(data,message='success'):
    return {'success':True,'status':'success','message':message,'data':data,'timestamp':datetime.now(timezone.utc).isoformat()}

def fail(message,code=400):
    return HTTPException(status_code=code,detail={'success':False,'status':'error','message':message,
      'data':None,'timestamp':datetime.now(timezone.utc).isoformat()})

@app.exception_handler(RequestValidationError)
async def validation_error(request,exc):
    return JSONResponse(status_code=422,content={'detail':{'success':False,'status':'error',
      'message':'Invalid parameters or request body','data':None,'timestamp':datetime.now(timezone.utc).isoformat()}})

def read(name):
    path=ROOT/'results'/name
    if not path.exists():raise fail('Run python run_analysis.py first',503)
    return json.loads(path.read_text(encoding='utf8'))

@app.get('/health')
def health():
    return ok({'module':'C','mode':'historical_replay','period':['2021-01-01','2025-12-31'],
      'results_available':(ROOT/'results/risk_daily.json').exists(),'is_forecast':False})

@app.get('/risk/summary')
def summary():return ok(read('summary.json'))

@app.get('/weather/daily')
def weather(start:str='2025-01-01',end:str='2025-12-31',limit:int=Query(366,ge=1,le=1826)):
    try:
        from datetime import date
        a,b=date.fromisoformat(start),date.fromisoformat(end)
        if a>b:raise ValueError()
    except ValueError:raise fail('Invalid start/end date')
    rows=[r for r in read('weather_daily.json') if start<=r['date']<=end]
    return ok({'total':len(rows),'items':rows[:limit]})

@app.get('/risk/daily')
def daily(date:str):
    rows=[r for r in read('risk_daily.json') if r['date']==date]
    if not rows:raise fail('Date not covered; historical data ends 2025-12-31',404)
    return ok(rows)

@app.get('/risk/events')
def events(start:str='2024-01-01',end:str='2025-12-31'):
    try:
        from datetime import date
        if date.fromisoformat(start)>date.fromisoformat(end):raise ValueError()
    except ValueError:raise fail('Invalid start/end date')
    return ok([r for r in read('candidate_events.json') if r['end_date']>=start and r['start_date']<=end])

@app.post('/integration/fuse')
def integration(payload:dict):
    cfg=json.loads((ROOT/'config.json').read_text(encoding='utf8'))
    try:return ok(fuse(payload,read('risk_daily.json'),cfg))
    except (ValueError,TypeError,KeyError,AttributeError) as ex:raise fail(str(ex))
