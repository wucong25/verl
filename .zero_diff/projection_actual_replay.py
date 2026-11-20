import json, traceback
from pathlib import Path
import megatron_adaptor
import torch,torch_npu
from safetensors import safe_open
from vllm_ascend.ops.triton.batch_invariant.matmul import matmul_persistent,linear_persistent
root=Path('/home/w00848261/verl/.zero_diff/runs/projection-actual-001');root.mkdir(exist_ok=True)
report={}
def cmp(a,b):
 a,b=a.detach().cpu(),b.detach().cpu();assert a.shape==b.shape,(a.shape,b.shape)
 return {'count':a.numel(),'mismatch':int((a!=b).sum()),'max_abs':float((a.float()-b.float()).abs().max())}
try:
 torch.npu.set_device(8)
 tr=Path('/home/w00848261/verl/.zero_diff/runs/proxy-megatron-005/trace/rank0'); inf=Path('/home/w00848261/verl/.zero_diff/runs/proxy-vllm-010/trace')
 nat=torch.load(tr/'chunk0.module.module.language_model.decoder.layers.0.self_attention.in_proj.000.pt',weights_only=True)['output'][0].squeeze(1)
 rec=torch.load(inf/'language_model.model.layers.0.linear_attn.in_proj_qkvz.016.pt',weights_only=True)
 x=rec['inputs'][0].npu(); ref=rec['output'][0]
 ckpt=Path('/mnt/weight/Qwen3.5-35B-A3B');idx=json.loads((ckpt/'model.safetensors.index.json').read_text())['weight_map']
 def weight(s):
  name='model.language_model.layers.0.'+s
  with safe_open(str(ckpt/idx[name]),framework='pt',device='cpu') as f:return f.get_tensor(name).npu()
 w=weight('linear_attn.in_proj_qkv.weight');z=weight('linear_attn.in_proj_z.weight')
 w2=torch.cat([w[:1024],w[2048:3072],w[4096:6144],z[:2048],weight('linear_attn.in_proj_b.weight')[:16],weight('linear_attn.in_proj_a.weight')[:16]])
 w8=torch.cat([w[:256],w[2048:2304],w[4096:4608],z[:512]])
 emb=torch.load(inf/'language_model.model.layers.0.input_layernorm.016.pt',weights_only=True)['inputs'][0].npu()
 gamma=weight('input_layernorm.weight')
 xn=torch_npu.npu_rms_norm(emb,1+gamma.float(),1e-6)[0]
 report['norm_native_formula_vs_infer']=cmp(xn,x)
 with torch.no_grad():
  for label,func in [('torch',lambda a,b:torch.matmul(a,b.t())),('bi_mm',lambda a,b:matmul_persistent(a,b.t())),('bi_linear',linear_persistent)]:
   o2=func(xn,w2);o8=func(xn,w8)
   gathered=torch.cat([o2[:,:256],o2[:,1024:1280],o2[:,2048:2560],o2[:,4096:4608]],-1)
   report[label]={'vs_native_tp2':cmp(o2,nat),'vs_infer_tp8':cmp(o8,ref),'tp2_vs_tp8':cmp(gathered,o8),'batch_split':cmp(o8,torch.cat([func(t,w8) for t in xn.split([1,62,81])]))}
except Exception:report['error']=traceback.format_exc()
finally:
 (root/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
