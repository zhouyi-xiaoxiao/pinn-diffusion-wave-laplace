"""Read-only validation of the completed current-machine cohort and saved checkpoints.
Usage: python analyze_cohort.py [--output PATH]
"""
from pathlib import Path
import argparse,json,hashlib,sys,statistics,math
candidate=Path(__file__).resolve().parents[2]
parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path);parser.add_argument('--table-output',type=Path);parser.add_argument('--input-dir',type=Path,default=Path(__file__).resolve().parent);args=parser.parse_args()
root=args.input_dir.resolve()
sys.path.insert(0,str(candidate/'research_highdim_remedies/check'))
import reimpl2 as M
import torch,numpy as np

records=[]
for block in ('B1','B4'):
 records.extend(json.loads(l) for l in (root/f'{block}-runs.jsonl').read_text().splitlines())
expected={(p,arm,seed,'main') for p,arms in [('P1',['plain','presolve','lift3c']),('P6',['plain','presolve','lift3c','lift1'])] for arm in arms for seed in (40,41,42)}|{('P1','lift3c',seed,'eqcpu') for seed in (40,41,42)}
actual={(r['p'],r['arm'],r['seed'],r['block']) for r in records}
assert actual==expected and len(records)==len(expected)==24, f'expected 24 unique runs; got {len(records)}, missing {expected-actual}'
checks=[]
for r in records:
 ck=root/r['checkpoint'];assert hashlib.sha256(ck.read_bytes()).hexdigest()==r['checkpoint_sha256']
 saved=torch.load(ck,map_location='cpu',weights_only=True);cfg=saved['config']
 for key in ('p','method','d','seed','arm','w','iters'):assert cfg[key]==r[key]
 net=M.R.Net(r['d'],r['arm'],r['shift']);net.load_state_dict(saved['state_dict']);net.eval()
 # Generate the specified held-out set directly, then compute its norm ratio without R.rel_err.
 x=torch.rand(50_000,r['d'],generator=torch.Generator().manual_seed(777000+r['d']),dtype=torch.float32)
 with torch.no_grad():
  y=net(x).double();truth=M.u_star(r['p'],x).double();num=torch.sum((y-truth)**2).item();den=torch.sum(truth**2).item()
 error=math.sqrt(num/den);assert abs(error-r['rel_l2'])<1e-10,(r,error)
 checks.append({'p':r['p'],'arm':r['arm'],'seed':r['seed'],'block':r['block'],'checkpoint_sha256':r['checkpoint_sha256'],'recomputed_rel_l2':error,'absolute_difference':abs(error-r['rel_l2'])})
by={(r['p'],r['arm'],r['seed'],r['block']):r for r in records};rows=[]
for p,arm,block in [('P1','presolve','main'),('P1','lift3c','main'),('P1','lift3c','eqcpu'),('P6','presolve','main'),('P6','lift3c','main'),('P6','lift1','main')]:
 a=[by[p,arm,s,block] for s in (40,41,42)];b=[by[p,'plain',s,'main'] for s in (40,41,42)];rho=[y['rel_l2']/x['rel_l2'] for x,y in zip(a,b)];med=statistics.median(rho)
 rows.append({'problem':p,'arm':arm,'block':block,'iterations':sorted({r['iters'] for r in a}),'error_range':[min(r['rel_l2'] for r in a),max(r['rel_l2'] for r in a)],'ratios':rho,'median_ratio':med,'wins':sum(x>1 for x in rho),'verdict':'helps' if all(x>1 for x in rho) and med>=1.5 else 'hurts' if med<=1/1.1 else 'neutral','median_cpu_ratio':statistics.median(r['cpu_train'] for r in a)/statistics.median(r['cpu_train'] for r in b)})
report={'status':'passed','records':len(records),'checkpoint_checks':checks,'comparisons':rows,'baseline_errors':{p:[by[p,'plain',s,'main']['rel_l2'] for s in (40,41,42)] for p in ('P1','P6')},'note':'Retrospective current-machine cohort; historical runs remain separate. CPU-calibrated iterations use current-machine medians, not historical machine timing.'}
s=json.dumps(report,indent=2);print(s)
if args.output: args.output.write_text(s+'\n')

if args.table_output:
 lines=[r'\begin{tabular}{@{}llrrrr@{}}',r'\toprule',r'Problem & Arm & Iterations & Median $\rho$ & Wins & CPU ratio \\',r'\midrule']
 for x in rows:
  prob=r'\Pone' if x['problem']=='P1' else r'\Psinlin';arm=x['arm']+(' (calibrated)' if x['block']=='eqcpu' else '')
  lines.append(f"{prob} & {arm} & {x['iterations'][0]} & {x['median_ratio']:.2f} & {x['wins']}/3 & {x['median_cpu_ratio']:.2f} "+r'\\')
 lines += [r'\bottomrule',r'\end{tabular}']
 args.table_output.write_text('\n'.join(lines)+'\n')
