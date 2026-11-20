from pathlib import Path
import torch,json
T=Path('.zero_diff/runs/proxy-megatron-011/trace');I=Path('.zero_diff/runs/proxy-vllm-020/trace')
def load(r,n):return torch.load(r/n,weights_only=False)
def unwrap(x):return x[0] if isinstance(x,(tuple,list)) else x
def cmp(a,b):
 a,b=a.squeeze(1),b.squeeze(1)
 assert a.shape==b.shape,(a.shape,b.shape)
 return {'mismatch':int((a!=b).sum()),'max':float((a.float()-b.float()).abs().max()),'n':a.numel()}
for l in range(4):
 ranks=(0,1) if l<2 else (8,9)
 p=f'chunk0.module.module.language_model.decoder.layers.{l%2}.';q=f'language_model.model.layers.{l}.'
 for nm,im in [('self_attention.out_proj' if l<3 else 'self_attention.linear_proj','linear_attn.out_proj' if l<3 else 'self_attn.o_proj'),('pre_mlp_layernorm','post_attention_layernorm'),('mlp','mlp')]:
  n=torch.cat([unwrap(load(T/f'rank{r}',p+nm+'.000.pt')['output']) for r in ranks],0)
  i=unwrap(load(I/'rank0',q+im+'.017.pt')['output']);print(l,nm,cmp(n,i),flush=True)
ref=load(Path('.zero_diff/runs/proxy-megatron-011'),'recompute.pt')['native_recompute'];rs=json.load(open('.zero_diff/runs/proxy-vllm-020/result.json'));fixed=torch.tensor(rs['fixed_reference_logprobs']).unsqueeze(0).expand_as(ref)
print('final',cmp(ref,fixed),flush=True)
