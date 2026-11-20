from pathlib import Path
import torch,torch_npu,time
from vllm_ascend.ops.triton.batch_invariant.matmul import linear_persistent
p=Path('.zero_diff/runs/proxy-vllm-026/trace/rank0')
torch.npu.set_device(0)
x=torch.load(p/'logprobs.016.pt',weights_only=False)['logits'].npu().float();x=x-x.max(-1,keepdim=True).values
def fixed_sum(y):
 width=y.shape[-1]; tile=1024; chunks=(width+tile-1)//tile
 padded=torch.nn.functional.pad(y,(0,chunks*tile-width))
 partial=linear_persistent(padded.reshape(-1,tile),torch.ones((1,tile),device=y.device,dtype=y.dtype))
 return linear_persistent(partial.reshape(-1,chunks),torch.ones((1,chunks),device=y.device,dtype=y.dtype)).squeeze(-1)
for half in (0,1):
 y=x.chunk(2,-1)[half].exp();w=torch.ones((1,y.shape[-1]),device=y.device,dtype=y.dtype)
 t=time.time();a=fixed_sum(y);b=fixed_sum(y[-1:]);torch.npu.synchronize()
 ref=y.cpu().double().sum(-1)
 print('half',half,'single',b.item(),'batch',a[-1].item(),'relerr',((a.cpu().double()-ref).abs()/ref).max().item(),'time',time.time()-t,flush=True)
