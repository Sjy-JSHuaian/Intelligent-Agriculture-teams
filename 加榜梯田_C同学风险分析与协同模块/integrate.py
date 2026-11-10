from pathlib import Path
import argparse,json
from agri_risk.integration import fuse
from agri_risk.weather import write_json
ROOT=Path(__file__).resolve().parent
parser=argparse.ArgumentParser()
parser.add_argument('--request',required=True,type=Path)
parser.add_argument('--output',required=True,type=Path)
args=parser.parse_args()
cfg=json.loads((ROOT/'config.json').read_text(encoding='utf8'))
records=json.loads((ROOT/'results/risk_daily.json').read_text(encoding='utf8'))
payload=json.loads(args.request.read_text(encoding='utf8-sig'))
result=fuse(payload,records,cfg)
args.output.parent.mkdir(parents=True,exist_ok=True)
write_json(result,args.output)
print('Saved',args.output,'mode=',result['mode'],'inspection_requests=',len(result['inspection_requests']))
