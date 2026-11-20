import json,traceback
from pathlib import Path
import torch,torch_npu
from safetensors import safe_open
from vllm_ascend.ops.triton.batch_invariant.matmul import linear_persistent
root=Path('/home/w00848261/verl/.zero_diff/runs/row-projection-001');root.mkdir(exist_ok=True)
report={}
def cmp(a,b):
 a,b=a.detach().cpu(),b.detach().cpu();assert a.shape==b.shape,(a.shape,b.shape)
 return {'count':a.numel(),'mismatch':int((a!=b).sum()),'max_abs':float((a.float()-b.float()).abs().max())}
try:
 torch.npu.set_device(8)
 tr=Path('/home/w00848261/verl/.zero_diff/runs/proxy-megatron-009/trace'); inf=Path('/home/w00848261/verl/.zero_diff/runs/proxy-vllm-011/trace')
 name='chunk0.module.module.language_model.decoder.layers.0.self_attention.out_proj.000.pt'
 native=[torch.load(tr/f'rank{i}'/name,weights_only=True) for i in (0,1)]
 x=torch.cat([r['inputs'][0].squeeze(1) for r in native],-1).npu()
 expected=torch.cat([r['output'][0].squeeze(1) for r in native],0)
 iv=torch.load(inf/'language_model.model.layers.0.linear_attn.out_proj.016.pt',weights_only=True)
 report['input_native_vs_infer_rank0']=cmp(x[:,:512],iv['inputs'][0])
 ckpt=Path('/mnt/weight/Qwen3.5-35B-A3B');idx=json.loads((ckpt/'model.safetensors.index.json').read_text())['weight_map'];key='model.language_model.layers.0.linear_attn.out_proj.weight'
 with safe_open(str(ckpt/idx[key]),framework='pt',device='cpu') as f:w=f.get_tensor(key).npu()
 with torch.no_grad():
  p2=[linear_persistent(a,b) for a,b in zip(x.chunk(2,-1),w.chunk(2,-1))]
  canonical=p2[0]+p2[1]
  p8=[linear_persistent(a,b) for a,b in zip(x.chunk(8,-1),w.chunk(8,-1))]
  serial=p8[0]
  for p in p8[1:]:serial=serial+p
  report['canonical_tp2_vs_native']=cmp(canonical,expected)
  report['serial_tp8_vs_native']=cmp(serial,expected)
  report['canonical_tp2_vs_infer']=cmp(canonical,iv['output'][0])
  report['full_gemm_vs_native']=cmp(linear_persistent(x,w),expected)
except Exception:report['error']=traceback.format_exc()
finally:
 (root/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
