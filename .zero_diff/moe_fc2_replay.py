exec(open('.zero_diff/moe_native_compare.py').read().split('total=bad=maximum=0')[0])
import torch_npu,json
torch.npu.set_device(9)
n2=load(T,p+'experts.linear_fc2.000.pt');probs=load(T,p+'experts.000.pt')['inputs'][2]
act=torch.empty((y.shape[0],512),dtype=torch.bfloat16);expected=torch.empty_like(y);nativeprob=torch.empty(y.shape[0],dtype=torch.bfloat16)
for e in range(32):
 l=0 if e==0 else ends[e-1];il=0 if e==0 else ies[e-1]
 for j in range(il,ies[e]):
  match=(x[l:ends[e]]==y[j]).all(-1).nonzero().flatten();assert len(match)>0
  ni=l+int(match[0]);act[j]=n2['inputs'][0][ni];expected[j]=n2['output'][0][ni];nativeprob[j]=probs[ni]
w1,w2=load(I,q+'weights.pt')
w2=w2.npu();z=torch_npu.npu_grouped_matmul(x=[act.npu()],weight=[w2],split_item=2,group_list_type=b['group_list_type'],group_type=0,group_list=b['group_list'].npu())[0].cpu()
actreplay=(torch_npu.npu_swiglu(b['output'].npu())*nativeprob.npu().unsqueeze(-1)).cpu()
def cmp(a,b):return {'n':a.numel(),'mismatch':int((a!=b).sum()),'max':float((a.float()-b.float()).abs().max())}
print(json.dumps({'fc2':cmp(z,expected),'weighted_activation':cmp(actreplay,act)}),flush=True)
