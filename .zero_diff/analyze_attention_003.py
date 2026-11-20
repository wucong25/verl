from pathlib import Path
import torch,json
root=Path('/home/w00848261/verl/.zero_diff/runs/proxy8-attention-003')
def cmp(a,b):
    assert a.shape==b.shape,(a.shape,b.shape)
    return {'mismatch':int((a!=b).sum()),'max_abs':float((a.float()-b.float()).abs().max())}
records=[]
for rank in range(8):
    trace=root/f'trace/rank{rank}'
    d=torch.load(trace/'fa.001.pt',weights_only=False,map_location='cpu')
    p=torch.load(trace/'fa.002.pt',weights_only=False,map_location='cpu')
    record={'rank':rank,'q':cmp(d['query'][0],p['query'][1024]),'k':cmp(d['cache_kv'][0][0],p['key'][:1025]),'v':cmp(d['cache_kv'][0][1],p['value'][:1025]),'out':cmp(d['output'][0],p['output'][1024]),'decode_q_shape':list(d['query'].shape),'prefill_q_shape':list(p['query'].shape)}
    records.append(record)
print(json.dumps(records,indent=2));(root/'fa-input-output-comparison.json').write_text(json.dumps(records,indent=2))
