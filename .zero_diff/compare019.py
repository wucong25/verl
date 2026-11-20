from pathlib import Path
import torch,json
T=Path('.zero_diff/runs/proxy-megatron-011/trace');I=Path('.zero_diff/runs/proxy-vllm-019/trace');p='chunk0.module.module.language_model.decoder.layers.0.';q='language_model.model.layers.0.'
def load(r,n):return torch.load(r/n,weights_only=False)
def unwrap(x):return x[0] if isinstance(x,(tuple,list)) else x
def cmp(a,b):
 a,b=a.squeeze(1),b.squeeze(1);return {'shapes':[list(a.shape),list(b.shape)],'mismatch':int((a!=b).sum()),'max':float((a.float()-b.float()).abs().max())}
for nm,im in [('mlp','mlp'),('mlp.shared_experts.linear_fc2','mlp.shared_expert.down_proj')]:
 n=torch.cat([unwrap(load(T/f'rank{r}',p+nm+'.000.pt')['output']) for r in (0,1)],0)
 i=unwrap(load(I/'rank0',q+im+'.017.pt')['output']);print(nm,cmp(n,i))
a=json.load(open('.zero_diff/runs/proxy-megatron-011/result.json'));b=json.load(open('.zero_diff/runs/proxy-vllm-019/result.json'));print('native keys',a.keys());print('fixed logprobs',b['fixed_reference_logprobs'])
