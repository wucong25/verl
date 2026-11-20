from pathlib import Path
import torch,json
root=Path('/home/w00848261/verl/.zero_diff/runs/proxy8-gdn-009')
leaf=Path('/home/w00848261/verl/.zero_diff/runs/gdn-stage-replay-001')
def cmp(a,b):
 if isinstance(a,torch.Tensor):return {'shape':list(a.shape),'mismatch':int((a!=b).sum()),'max_abs':float((a.float()-b.float()).abs().max())}
 if isinstance(a,(list,tuple)):return [cmp(x,y) for x,y in zip(a,b)]
 return (a,b)
rows=[]
for p in sorted((root/'trace/rank0').glob('*.000.pt')):
 name=p.name[:-7];d=torch.load(p,weights_only=True);other=p.with_name(f'{name}.003.pt');rec={'name':name}
 if other.exists():
  v=torch.load(other,weights_only=True)
  try:rec['repeat']=cmp(d['output'],v['output'])
  except RuntimeError:pass
 l=leaf/f'00-{name}.pt'
 if l.exists():rec['leaf']=cmp(d['output'],torch.load(l,weights_only=True)['output'])
 rows.append(rec)
(root/'stage-comparison.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
print('env',torch.load(root/'trace/rank0/environment.pt',weights_only=True))
