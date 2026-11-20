from pathlib import Path
import json,torch
root=Path('/home/w00848261/verl/.zero_diff/runs/proxy8-prefix-007')
files=sorted((root/'trace').glob('*layers.0.*'))
shapes={p.name.split('.')[-2]:torch.load(p,weights_only=True)['output'][0].shape[0] for p in files}
print(shapes)
batch=next(k for k,v in shapes.items() if v>1024 and v!=1040)
single=next(k for k,v in shapes.items() if v==1024)
history=next(k for k,v in shapes.items() if v==1040)
rows=[]
for layer in range(8):
 name=f'language_model.model.layers.{layer}'
 a=torch.load(root/'trace'/f'{name}.{batch}.pt',weights_only=True)
 b=torch.load(root/'trace'/f'{name}.{single}.pt',weights_only=True)
 c=torch.load(root/'trace'/f'{name}.{history}.pt',weights_only=True)
 record={'layer':layer}
 for j,key in enumerate(('hidden','residual')):
  x=a['output'][j][-1024:];y=b['output'][j];z=c['output'][j][:1024]
  for tag,v in (('batch_single',x),('history_single',z)):
   bad=(v!=y);r=bad.any(-1).nonzero().flatten().tolist()
   record[f'{tag}_{key}']={'mismatch':int(bad.sum()),'max_abs':float((v.float()-y.float()).abs().max()),'bad_rows':r}
 rows.append(record)
(root/'layer-comparison.json').write_text(json.dumps(rows,indent=2))
for r in rows:print({k:({a:b for a,b in v.items() if a!='bad_rows'} | {'row_range':v['bad_rows'][:1]+v['bad_rows'][-1:]}) if isinstance(v,dict) else v for k,v in r.items()})
