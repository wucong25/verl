from pathlib import Path
import torch
T=Path('.zero_diff/runs/proxy-megatron-011/trace/rank0');I=Path('.zero_diff/runs/proxy-vllm-016/trace/rank0')
p='chunk0.module.module.language_model.decoder.layers.0.mlp.';q='language_model.model.layers.0.mlp.experts.routed_experts.'
def load(root,name):return torch.load(root/name,weights_only=False)
def describe(x):
 if isinstance(x,torch.Tensor):return (tuple(x.shape),str(x.dtype),x.tolist() if x.numel()<80 else None)
 if isinstance(x,(list,tuple)):return [describe(a) for a in x]
 return x
for part in ('experts','experts.linear_fc1','experts.linear_fc2'):
 d=load(T,p+part+'.000.pt');print(part,{k:describe(v) for k,v in d.items()})
a=load(T,p+'experts.linear_fc1.000.pt');b=load(I,q+'apply_gmm1.017.pt')
x=a['inputs'][0]; y=b['hidden_states'];print('FC1 row shapes',x.shape,y.shape)
counts=a['inputs'][1];print('native_counts',describe(counts))
if isinstance(counts,torch.Tensor):counts=counts.tolist()
ends=torch.tensor(counts).cumsum(0).tolist();ies=b['group_list'];ies=ies.cumsum(0).tolist() if b['group_list_type']==1 else ies.tolist()
total=bad=maximum=0
for e in range(32):
 l=0 if e==0 else ends[e-1];il=0 if e==0 else ies[e-1]
 for j in range(il,ies[e]):
  match=(x[l:ends[e]]==y[j]).all(-1).nonzero().flatten()
  if not len(match): print('missing',e,j);continue
  v=a['output'][0][l+int(match[0])];w=b['output'][j];total+=v.numel();bad+=int((v!=w).sum());maximum=max(maximum,float((v.float()-w.float()).abs().max()))
print('FC1 result',total,bad,maximum)
