from pathlib import Path
import torch,json
base=Path('/home/w00848261/verl/.zero_diff/runs')
def select(root,kvlen,qlen):
 for path in sorted((root/'trace/rank0').glob('fa.*.pt')):
  d=torch.load(path,weights_only=False);start=0
  for end,length in zip(d['q_ends'],d['kv_lengths']):
   if length==kvlen and end-start==qlen:return {k:d[k][start:end] for k in ('query','key','value','output')}
   start=end
 raise ValueError((root,kvlen,qlen))
def cmp(a,b):
 return {'mismatch':int((a!=b).sum()),'max_abs':float((a.float()-b.float()).abs().max()),'bad_rows':(a!=b).flatten(1).any(1).nonzero().flatten().tolist()}
a=select(base/'proxy8-attention-003',1024,1024);b=select(base/'proxy8-attention-004',1024,1024)
c=select(base/'proxy8-attention-003',1040,1040);d=select(base/'proxy8-attention-004',1040,1040)
result={'original_prefix_003_vs_004':{k:cmp(a[k],b[k]) for k in a},'forced_history_003_vs_004':{k:cmp(c[k],d[k]) for k in c}}
(base/'proxy8-attention-004/schedule-comparison.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
