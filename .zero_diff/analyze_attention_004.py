from pathlib import Path
import torch,json
root=Path('/home/w00848261/verl/.zero_diff/runs/proxy8-attention-004')
def cmp(a,b):
 assert a.shape==b.shape,(a.shape,b.shape)
 return {'mismatch':int((a!=b).sum()),'max_abs':float((a.float()-b.float()).abs().max()),'bad_rows':(a!=b).flatten(1).any(1).nonzero().flatten().tolist() if a.ndim>1 else []}
def select(records,kvlen,qlen):
 for p,x in records:
  start=0
  for request,(end,length) in enumerate(zip(x['q_ends'],x['kv_lengths'])):
   if length==kvlen and end-start==qlen:return p,x,request,start,end
   start=end
 raise ValueError((kvlen,qlen))
records=[]
for rank in range(8):
 trace=root/f'trace/rank{rank}'
 values=[(p.name,torch.load(p,weights_only=False,map_location='cpu')) for p in sorted(trace.glob('fa.*.pt'))]
 dn,d,di,ds,de=select(values,1025,1);pn,p,pi,ps,pe=select(values,1040,1040)
 record={'rank':rank,'decode_file':dn,'prefill_file':pn,'decode_request':di,'prefill_request':pi,'q':cmp(d['query'][ds:de],p['query'][ps+1024:ps+1025]),'k':cmp(d['cache_kv'][di][0],p['key'][ps:ps+1025]),'v':cmp(d['cache_kv'][di][1],p['value'][ps:ps+1025]),'out':cmp(d['output'][ds:de],p['output'][ps+1024:ps+1025])}
 records.append(record)
print(json.dumps(records,indent=2));(root/'fa-input-output-comparison.json').write_text(json.dumps(records,indent=2))
