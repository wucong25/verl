from pathlib import Path
import torch,json
root=Path('.zero_diff/runs/proxy-vllm-016/trace')
prefix='language_model.model.layers.0.mlp.experts.routed_experts'
def load(r,method,s):return torch.load(root/f'rank{r}'/f'{prefix}.{method}.{s:03d}.pt',weights_only=False)
def desc(x):
 if isinstance(x,torch.Tensor):return [list(x.shape),str(x.dtype)]
 if isinstance(x,(tuple,list)):return [desc(v) for v in x]
 return x
def diff(a,b):return {'count':int((a!=b).sum()),'max':float((a.float()-b.float()).abs().max()) if a.numel() else 0,'n':a.numel()}
for r in range(8):
 d=load(r,'apply_gmm1',1);p=load(r,'apply_gmm1',16)
 if r==0:print('shapes', {k:desc(v) for k,v in p.items()})
 print('rank',r,'decode_group',d['group_list'].tolist(),'pref_group',p['group_list'].tolist())
 gd=d['group_list'].tolist();gp=p['group_list'].tolist()
 if d['group_list_type']==1:gd=torch.tensor(gd).cumsum(0).tolist()
 if p['group_list_type']==1:gp=torch.tensor(gp).cumsum(0).tolist()
 da=load(r,'apply_act_quant',1);pa=load(r,'apply_act_quant',16);d2=load(r,'apply_gmm2',1);p2=load(r,'apply_gmm2',16)
 for e in range(len(gd)):
  dl=gd[e-1] if e else 0;pl=gp[e-1] if e else 0
  for i in range(dl,gd[e]):
   matches=(p['hidden_states'][pl:gp[e]]==d['hidden_states'][i]).all(-1).nonzero().flatten()
   if len(matches)!=1: print('map',r,e,i,'matches',len(matches));continue
   j=pl+int(matches[0]);print('row',r,e,i,j,'gmm1',diff(d['output'][i],p['output'][j]),'act',diff(da['args'][0][i],pa['args'][0][j]),'gmm2',diff(d2['output'][i],p2['output'][j]),'scales',desc(d['topk_scales']))
