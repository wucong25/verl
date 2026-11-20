from pathlib import Path
import json,torch
root=Path('/home/w00848261/verl/.zero_diff/runs/proxy8-prefix-005');trace=root/'trace'
frames=[]
for p in sorted(trace.glob('language_model.model.layers.0.*.pt')):
    d=torch.load(p,weights_only=True);pos=d['kwargs']['positions'][0]
    starts=(pos==0).nonzero().flatten().tolist();ends=starts[1:]+[pos.numel()]
    for start,end in zip(starts,ends):
        if end-start==1024:frames.append((p.name.split('.')[-2],start,end))
assert len(frames)==2,frames
batch,single=frames
result={'batch_frame':batch,'single_frame':single,'layers':[]}
def cmp(a,b):
    assert a.shape==b.shape,(a.shape,b.shape)
    rows=(a!=b).flatten(1).any(1).nonzero().flatten().tolist() if a.ndim>1 else (a!=b).nonzero().flatten().tolist()
    return {'count':a.numel(),'mismatch':int((a!=b).sum()),'max_abs':float((a.float()-b.float()).abs().max()),'bad_rows':rows}
for name in ['language_model.model.embed_tokens']+[f'language_model.model.layers.{i}' for i in range(8)]:
    a=torch.load(trace/f'{name}.{batch[0]}.pt',weights_only=True);b=torch.load(trace/f'{name}.{single[0]}.pt',weights_only=True)
    if name.endswith('embed_tokens'):
        result['embedding']=cmp(a['output'][batch[1]:batch[2]],b['output'][single[1]:single[2]])
        result['input_ids']=cmp(a['args'][0][batch[1]:batch[2]],b['args'][0][single[1]:single[2]])
    else:
        record={'layer':int(name.split('.')[-1])}
        for i,label in enumerate(('hidden','residual')):record[label]=cmp(a['output'][i][batch[1]:batch[2]],b['output'][i][single[1]:single[2]])
        result['layers'].append(record)
(root/'boundary-comparison.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
