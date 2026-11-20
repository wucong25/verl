from pathlib import Path
import json,torch
root=Path('/home/w00848261/verl/.zero_diff/runs/full-inference-002')
trace=root/'trace/rank0'
rows=[]
for layer in range(40):
    name=f'language_model.model.layers.{layer}'
    a=torch.load(trace/f'{name}.002.pt',weights_only=True)['output']
    b=torch.load(trace/f'{name}.022.pt',weights_only=True)['output']
    result={'layer':layer}
    for index,key in enumerate(('hidden','residual')):
        x,y=a[index][3],b[index][1024]
        result[key]={'mismatch':int((x!=y).sum()),'max_abs':float((x.float()-y.float()).abs().max())}
    rows.append(result)
print(json.dumps(rows,indent=2))
(root/'first-divergence.json').write_text(json.dumps(rows,indent=2))
