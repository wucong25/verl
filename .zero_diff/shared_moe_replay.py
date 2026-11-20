from pathlib import Path
import torch,torch_npu,json
from safetensors import safe_open
from vllm_ascend.ops.triton.batch_invariant.matmul import linear_persistent
torch.npu.set_device(10)
T=Path('.zero_diff/runs/proxy-megatron-011/trace');I=Path('.zero_diff/runs/proxy-vllm-016/trace');p='chunk0.module.module.language_model.decoder.layers.0.mlp.shared_experts';q='language_model.model.layers.0.mlp.shared_expert'
def load(root,name):return torch.load(root/name,weights_only=False)
def cmp(a,b):
 a,b=a.cpu(),b.cpu();return {'shape':[list(a.shape),list(b.shape)],'mismatch':int((a!=b).sum()),'max':float((a.float()-b.float()).abs().max())}
v=[load(I/f'rank{r}',q+'.gate_up_proj.017.pt')['output'][0] for r in range(8)]
gates,ups=zip(*(x.chunk(2,-1) for x in v));full=torch.cat([torch.cat(gates,-1),torch.cat(ups,-1)],-1).npu()
nt=[load(T/f'rank{r}',p+'.linear_fc1.000.pt')['output'][0].squeeze(1) for r in (0,1)]
ng,nu=zip(*(x.chunk(2,-1) for x in nt));nativefc1=torch.cat([torch.cat(ng,-1),torch.cat(nu,-1)],-1)
act=torch.nn.functional.silu(full.chunk(2,-1)[0])*full.chunk(2,-1)[1]
na=torch.cat([load(T/f'rank{r}',p+'.linear_fc2.000.pt')['inputs'][0].squeeze(1) for r in (0,1)],-1)
with safe_open('.zero_diff/proxy-4layer/model-00009-of-00009.safetensors',framework='pt') as f:
 w=f.get_tensor('model.language_model.layers.0.mlp.shared_expert.down_proj.weight').npu()
 wg=f.get_tensor('model.language_model.layers.0.mlp.shared_expert_gate.weight').npu()
left,right=act.chunk(2,-1);wl,wr=w.chunk(2,-1);out=linear_persistent(left,wl)+linear_persistent(right,wr)
nout=torch.cat([load(T/f'rank{r}',p+'.linear_fc2.000.pt')['output'][0].squeeze(1) for r in (0,1)],0)
print('nativeFC2shape',nout.shape)
source=load(I/'rank0','language_model.model.layers.0.mlp.gate.017.pt')['inputs'][0].npu()
gated=out*torch.sigmoid(linear_persistent(source,wg))
expected=torch.cat([load(T/f'rank{r}',p+'.000.pt')['output'].squeeze(1) for r in (0,1)],0)
print(json.dumps({'fc1':cmp(full,nativefc1),'act':cmp(act,na),'down':cmp(out,nout),'gated':cmp(gated,expected)}),flush=True)
