import torch,torch_npu,json,importlib
from pathlib import Path
torch.npu.set_device(0)
from vllm_ascend.ops.gdn_batch_invariant import chunk_gdn_training_math
root=Path('/home/w00848261/verl/.zero_diff/runs/gdn-stage-replay-001');root.mkdir(parents=True,exist_ok=True)
step=[0];records=[]
def tree(v):
 if isinstance(v,torch.Tensor):return v.detach().cpu().clone()
 if isinstance(v,(tuple,list)):return tuple(tree(x) for x in v)
 if isinstance(v,dict):return {k:tree(x) for k,x in v.items()}
 return v
for modname,fn in [('mindspeed_ops.api.triton.cumsum','chunk_local_cumsum'),('mindspeed_ops.api.triton.chunk_scaled_dot_kkt','chunk_scaled_dot_kkt_fwd'),('mindspeed_ops.api.triton.solve_tril','solve_tril'),('fla_npu.ops.ascendc','npu_recompute_w_u_fwd'),('fla_npu.ops.ascendc','npu_chunk_gated_delta_rule_fwd_h'),('fla_npu.ops.ascendc','npu_chunk_fwd_o')]:
 mod=importlib.import_module(modname);orig=getattr(mod,fn)
 def wrap(orig=orig,fn=fn):
  def call(*args,**kwargs):
   result=orig(*args,**kwargs)
   torch.save({'output':tree(result)},root/f'{step[0]:02d}-{fn}.pt')
   return result
  return call
 setattr(mod,fn,wrap())
source=Path('/home/w00848261/verl/.zero_diff/runs/proxy8-gdn-008/trace/rank0')
a=torch.load(source/'chunk_gdn_training_math.000.pt',weights_only=True)
b=torch.load(source/'chunk_gdn_training_math.002.pt',weights_only=True)
with torch.no_grad():
 for i,record in enumerate((a,a,b,a,a)):
  step[0]=i
  out,state=chunk_gdn_training_math(*(t.npu() for t in record['args']),**record['kwargs'])
  out=out.cpu();ref=a['output'][0] if i!=2 else b['output'][0]
  r={'step':i,'shape':list(out.shape),'vs_captured':int((out!=ref).sum()),'vs_history':int((out[:,:1024]!=b['output'][0][:,:1024]).sum())};records.append(r);print(r,flush=True)
(root/'result.json').write_text(json.dumps(records,indent=2))
