from pathlib import Path
import torch,torch_npu,time
from vllm_ascend.ops.triton.batch_invariant.matmul import linear_persistent
p=Path('.zero_diff/runs/proxy-vllm-026/trace/rank0')
torch.npu.set_device(0)
x=torch.load(p/'logprobs.016.pt',weights_only=False)['logits'].npu().float();x=x-x.max(-1,keepdim=True).values
for half in (0,1):
 y=x.chunk(2,-1)[half].exp();w=torch.ones((1,y.shape[-1]),device=y.device,dtype=y.dtype)
 t=time.time();a=linear_persistent(y,w).squeeze(-1);b=linear_persistent(y[-1:],w).squeeze(-1);torch.npu.synchronize()
 ref=y.cpu().double().sum(-1)
 print('half',half,'single',b.item(),'batch',a[-1].item(),'relerr',((a.cpu().double()-ref).abs()/ref).max().item(),'time',time.time()-t,flush=True)
