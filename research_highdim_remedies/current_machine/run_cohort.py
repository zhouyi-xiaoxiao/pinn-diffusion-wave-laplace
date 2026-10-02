"""Independent current-machine completion cohort; never appends to historical files.
Two single-thread processes: B1 repeats all seeds/arms before same-machine equal-CPU calibration;
B4 fills the SinLinD PINN block. Results are retrospective, not new preregistered evidence.
"""
from pathlib import Path
import sys,json,time,hashlib,platform,argparse
candidate=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('block', choices=('B1','B4'))
parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent)
args=parser.parse_args()
root=args.output_dir.resolve();(root/'checkpoints').mkdir(parents=True,exist_ok=True)
sys.path.insert(0,str(candidate/'research_highdim_remedies/check'))
import reimpl2 as M
import torch,numpy
block=args.block
M.R.OUT=str(root/f'{block}-runs.jsonl')
original_net=M.R.Net
class CapturedNet(original_net):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs)
  global last_net
  last_net=self
M.R.Net=CapturedNet
original_train=M.R.train
def train(p,method,d,seed,arm,w,iters=4000):
 result=original_train(p,method,d,seed,arm,w,iters)
 ck=root/'checkpoints'/f'{block}-{p}-{method}-d{d}-seed{seed}-{arm}-{iters}.pt'
 torch.save({'state_dict':last_net.state_dict(),'config':dict(p=p,method=method,d=d,seed=seed,arm=arm,w=w,iters=iters),'metrics':result},ck)
 result['checkpoint']=str(ck.relative_to(root));result['checkpoint_sha256']=hashlib.sha256(ck.read_bytes()).hexdigest()
 return result
M.R.train=train
meta={'block':block,'started':time.strftime('%Y-%m-%dT%H:%M:%S%z'),'platform':platform.platform(),'python':sys.version,'torch':torch.__version__,'numpy':numpy.__version__,'threads':torch.get_num_threads(),'seeds':list(M.SEEDS),'status':'running','scope':'retrospective current-machine cohort, no historical file mutations','source_sha256':{p:hashlib.sha256((candidate/'research_highdim_remedies/check'/p).read_bytes()).hexdigest() for p in ('reimpl.py','reimpl2.py')}}
(root/f'{block}-meta.json').write_text(json.dumps(meta,indent=2))
for seed in M.SEEDS:
 for arm in (('plain','presolve','lift3c') if block=='B1' else ('plain','presolve','lift3c','lift1')):
  M.run('main','P1' if block=='B1' else 'P6','pinn',20,seed,arm,1000.)
if block=='B1':
 n=M.neq('P1','pinn'); print('same-machine equal-CPU calibration iterations',n,flush=True)
 for seed in M.SEEDS:M.run('eqcpu','P1','pinn',20,seed,'lift3c',1000.,n)
meta['status']='completed';meta['finished']=time.strftime('%Y-%m-%dT%H:%M:%S%z');(root/f'{block}-meta.json').write_text(json.dumps(meta,indent=2))
