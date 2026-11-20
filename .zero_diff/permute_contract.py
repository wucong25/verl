import torch,torch_npu
from pathlib import Path
torch.npu.set_device(9)
x=torch.arange(4*32,dtype=torch.float32).reshape(4,32).to(torch.bfloat16).npu();ids=torch.tensor([[3,1],[0,2],[2,1],[3,0]],dtype=torch.int32,device='npu');p=torch.tensor([[.1,.9],[.3,.7],[.2,.8],[.4,.6]],device='npu')
y,ix=torch_npu.npu_moe_token_permute(x,ids)
print('ids',ids.cpu(),'index',ix.cpu(),'rows',y[:,0].cpu(),flush=True)
print('flat_probs_gather',p.flatten()[ix.long()].cpu(),flush=True)
