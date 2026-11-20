from pathlib import Path
import torch,torch_npu
p=Path('.zero_diff/runs/proxy-vllm-026/trace/rank0')
torch.npu.set_device(0)
x=torch.load(p/'logprobs.016.pt',weights_only=False)['logits'].npu().float();x=x-x.max(-1,keepdim=True).values
single=x[142:143].clone(); batch=x
for half in (0,1):
 a=single.chunk(2,-1)[half];b=batch.chunk(2,-1)[half]
 ea,eb=a.exp(),b.exp()
 print('half',half,'exp mismatch',int((ea!=eb[142:143]).sum()),flush=True)
 for name,y in [('single',ea),('batch',eb),('batch_last',eb[142:143]),('single_repeat',ea.repeat(143,1)),('batch_3d',eb[:,None,:])]:
  z=y.sum(-1).flatten();print(name,z[-1].item(),flush=True)
 print('exp copied repeat', int((ea.repeat(143,1)[-1]!=eb[-1]).sum()),flush=True)
