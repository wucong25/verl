from pathlib import Path
import torch,json
T=Path('.zero_diff/runs/proxy-megatron-011/trace');I=Path('.zero_diff/runs/proxy-vllm-021/trace');p='chunk0.module.module.language_model.decoder.layers.1.self_attention';q='language_model.model.layers.3.self_attn'
def load(root,name):return torch.load(root/name,weights_only=False)
def cmp(a,b):
 assert a.shape==b.shape,(a.shape,b.shape)
 return {'mismatch':int((a!=b).sum()),'max':float((a.float()-b.float()).abs().max()),'n':a.numel()}
ns=[load(T/f'rank{r}',p+'.core_attention.000.pt') for r in (8,9)]
is_=[load(I/f'rank{r}',q+'.attn.017.pt') for r in range(8)]
for k,name in enumerate(('q','k','v')):
 a=torch.cat([n['inputs'][k] for n in ns],dim=1)
 take=range(8) if k==0 else (0,4)
 b=torch.cat([is_[r]['inputs'][k].reshape(144,-1,256) for r in take],dim=1)
 print(name,cmp(a,b),flush=True)
print('attn_output',cmp(torch.cat([n['output'] for n in ns],1),torch.cat([i['output'].reshape(144,-1,256) for i in is_],1)))
for r in (8,9):
 for part in ('q_layernorm','k_layernorm'):
  a=load(T/f'rank{r}',p+f'.{part}.000.pt');print('native',r,part,'input',a['inputs'][0].shape,'out',a['output'].shape)
