from pathlib import Path
import torch,json
root=Path('/home/w00848261/verl/.zero_diff/runs/proxy8-gdn-008')
rows=[]
def compare(a,b,path=''):
 out={}
 if isinstance(a,torch.Tensor):
  if a.ndim!=b.ndim:return {'shape':(list(a.shape),list(b.shape))}
  sizes=a.shape
  if a.shape!=b.shape:
   for dim,(x,y) in enumerate(zip(a.shape,b.shape)):
    if x!=y:
     if (x,y) not in ((1024,1040),(4096,4160)):return {'shape':(list(a.shape),list(b.shape))}
     b=b.narrow(dim,0,x)
  bad=a!=b
  out[path]={'mismatch':int(bad.sum()),'max_abs':float((a.float()-b.float()).abs().max()) if a.numel() else 0}
 elif isinstance(a,(list,tuple)):
  for i,(x,y) in enumerate(zip(a,b)):out.update(compare(x,y,path+f'/{i}'))
 elif isinstance(a,dict):
  for k in a.keys()&b.keys():out.update(compare(a[k],b[k],path+'/'+k))
 return out
for rank in range(8):
 t=root/f'trace/rank{rank}';r={'rank':rank}
 for p in sorted(t.glob('*.000.pt')):
  name=p.name[:-7];second='001' if name=='causal_conv1d_fn' else '002'
  b=t/f'{name}.{second}.pt'
  if b.exists():r[name]=compare(torch.load(p,weights_only=True),torch.load(b,weights_only=True))
 rows.append(r)
(root/'boundary-comparison.json').write_text(json.dumps(rows,indent=2))
for r in rows:
 print('RANK',r['rank'])
 for n,vals in r.items():
  if n!='rank':print(n,{k:v for k,v in vals.items() if v.get('mismatch',0) or 'shape' in v})
