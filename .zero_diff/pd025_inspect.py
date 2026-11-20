from pathlib import Path
import json,torch
root=Path('.zero_diff/runs/proxy-vllm-025');r=json.loads((root/'result.json').read_text());bad=[i for i,(a,b) in enumerate(zip(r['decode_logprobs'],r['prefill_logprobs'])) if a!=b];print('bad',[(i,r['decode_logprobs'][i],r['prefill_logprobs'][i]) for i in bad])
for i in bad:
 for rank in range(8):
  trace=root/'trace'/f'rank{rank}'
  for path in sorted(trace.glob('*.016.pt')):
   name=path.name[:-7]
   if any(x in name for x in ('apply_','routing')):continue
   a=torch.load(trace/f'{name}.{i:03d}.pt',weights_only=False)['output'];b=torch.load(path,weights_only=False)['output']
   if isinstance(a,tuple):a=a[0];b=b[0]
   if not isinstance(a,torch.Tensor):continue
   if a.shape[0] not in (1,128) or b.shape[0]!=144:continue
   a=a[-1:];b=b[127+i:128+i]
   if a.shape!=b.shape:continue
   count=int((a!=b).sum())
   if count:print(rank,name,count,float((a.float()-b.float()).abs().max()))
