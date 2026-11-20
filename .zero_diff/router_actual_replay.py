import json,traceback
from pathlib import Path
import torch,torch_npu
from safetensors import safe_open
from vllm_ascend.ops.triton.batch_invariant.matmul import linear_persistent
from vllm_ascend.ops.triton.batch_invariant.softmax import softmax_batch_invariant
root=Path('/home/w00848261/verl/.zero_diff/runs/router-actual-001');root.mkdir(exist_ok=True)
report={}
def cmp(a,b):
 a,b=a.detach().cpu(),b.detach().cpu();assert a.shape==b.shape,(a.shape,b.shape)
 return {'count':a.numel(),'mismatch':int((a!=b).sum()),'max_abs':float((a.float()-b.float()).abs().max())}
try:
 torch.npu.set_device(8)
 tr=Path('/home/w00848261/verl/.zero_diff/runs/proxy-megatron-010/trace');inf=Path('/home/w00848261/verl/.zero_diff/runs/proxy-vllm-013/trace')
 key='chunk0.module.module.language_model.decoder.layers.0.mlp.router.000.pt'
 records=[torch.load(tr/f'rank{i}'/key,weights_only=True) for i in (0,1)]
 x=torch.cat([r['inputs'][0].reshape(-1,2048) for r in records],0).npu()
 native=torch.cat([r['output'][0] for r in records],0)
 rec=torch.load(inf/'language_model.model.layers.0.mlp.gate.016.pt',weights_only=True)
 report['input']=cmp(x,rec['inputs'][0])
 ckpt=Path('/mnt/weight/Qwen3.5-35B-A3B');idx=json.loads((ckpt/'model.safetensors.index.json').read_text())['weight_map'];key='model.language_model.layers.0.mlp.gate.weight'
 with safe_open(str(ckpt/idx[key]),framework='pt',device='cpu') as f:w=f.get_tensor(key).npu()
 topk=int((native[0]!=0).sum());report['topk']=topk
 with torch.no_grad():
  for name,dtype in [('fp32',torch.float32),('bf16',torch.bfloat16)]:
   logits=linear_persistent(x.to(dtype),w.to(dtype))
   if dtype==torch.bfloat16:report['bf16_logits_vs_infer']=cmp(logits,rec['output'][0])
   scores,ids=torch.topk(logits.float(),topk,dim=-1,sorted=False)
   for mode,fn in [('native_softmax',lambda t:torch.softmax(t,dim=-1)),('bi_softmax',lambda t:softmax_batch_invariant(t,dim=-1))]:
    probs=fn(scores)
    dense=torch.zeros_like(logits,dtype=torch.float32).scatter(1,ids,probs)
    report[name+'_'+mode]=cmp(dense,native)
    report[name+'_routing_mismatch']=int(((dense.cpu()!=0)!=(native!=0)).sum())
   whole=torch.softmax(logits.float(),dim=-1).gather(1,ids);whole=whole/whole.sum(-1,keepdim=True)
   report[name+'_pre_softmax_renorm']=cmp(torch.zeros_like(logits,dtype=torch.float32).scatter(1,ids,whole),native)
except Exception:report['error']=traceback.format_exc()
finally:
 (root/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
