from pathlib import Path
import torch,json
r=Path('.zero_diff/runs/proxy-vllm-008/trace')
report=[]
def tensors(x,path=''):
 if isinstance(x,torch.Tensor):yield path,x
 elif isinstance(x,(tuple,list)):
  for i,v in enumerate(x):yield from tensors(v,path+f'/{i}')
for f in sorted(r.glob('*.016.pt')):
 name=f.name[:-7]
 ref=torch.load(f,weights_only=True)
 row={'module':name}
 for key in ('inputs','output'):
  pre=dict(tensors(torch.load(r/(name+'.000.pt'),weights_only=True)[key]))
  dec=[dict(tensors(torch.load(r/(name+f'.{i:03d}.pt'),weights_only=True)[key])) for i in range(1,16)]
  for p,v in tensors(ref[key]):
   if not v.ndim or v.shape[0] not in (144,576):continue
   factor=v.shape[0]//144
   a=pre[p]; b=v[:128*factor]
   if a.shape!=b.shape:continue
   ds=torch.cat([d[p] for d in dec]); target=v[128*factor:143*factor]
   if ds.shape!=target.shape:continue
   stat=lambda x,y:{'neq':int((x!=y).sum()),'numel':x.numel(),'max':float((x.float()-y.float()).abs().max())}
   row[key+p]={'prefix':stat(a,b),'decode':stat(ds,target)}
 report.append(row)
print(json.dumps(report,indent=2))
(r.parent/'trace_comparison.json').write_text(json.dumps(report,indent=2))
