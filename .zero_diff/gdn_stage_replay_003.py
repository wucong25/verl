import torch,torch_npu,json,importlib
from pathlib import Path
torch.npu.set_device(8)
from vllm_ascend.ops.gdn_batch_invariant import chunk_gdn_training_math
from vllm_ascend.batch_invariant import enable_batch_invariant_mode
enable_batch_invariant_mode()
torch.use_deterministic_algorithms(True,warn_only=True)
torch.npu.set_compile_mode(jit_compile=False)
root=Path('/home/w00848261/verl/.zero_diff/runs/gdn-stage-replay-003');root.mkdir(parents=True,exist_ok=True)
step=[0];records=[]
def tree(v):
 if isinstance(v,torch.Tensor):return v.detach().cpu().clone()
 if isinstance(v,(tuple,list)):return tuple(tree(x) for x in v)
 if isinstance(v,dict):return {k:tree(x) for k,x in v.items()}
 return v
source=Path('/home/w00848261/verl/.zero_diff/runs/proxy8-gdn-008/trace/rank0')
a=torch.load(source/'chunk_gdn_training_math.000.pt',weights_only=True)
b=torch.load(source/'chunk_gdn_training_math.002.pt',weights_only=True)
with torch.inference_mode():
 for i,record in enumerate((a,a,b,a,a)):
  step[0]=i
  out,state=chunk_gdn_training_math(*(t.npu() for t in record['args']),**record['kwargs'])
  out=out.cpu();ref=a['output'][0] if i!=2 else b['output'][0]
  r={'step':i,'shape':list(out.shape),'vs_captured':int((out!=ref).sum()),'vs_history':int((out[:,:1024]!=b['output'][0][:,:1024]).sum())};records.append(r);print(r,flush=True)
(root/'result.json').write_text(json.dumps(records,indent=2))
