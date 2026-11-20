import json,traceback
from pathlib import Path
import megatron_adaptor
import torch
from safetensors import safe_open
from verl.utils.kernel.npu.fla_compat import register_fla_legacy_ops
from mindspeed_bridge.features.megatron.common.fla_npu_causal_conv1d import fla_npu_causal_conv1d
from mindspeed_bridge.models.qwen_vl.modelling_qwen3_vl.flash_gated_delta_rule import flash_gated_delta_rule
from fla.modules.l2norm import l2norm
from megatron.core.ssm.gated_delta_net import GatedDeltaNet
root=Path('/home/w00848261/verl/.zero_diff/runs/gdn-actual-006');root.mkdir(exist_ok=True)
report={}
def cmp(a,b):
 a,b=a.detach().cpu(),b.detach().cpu();assert a.shape==b.shape,(a.shape,b.shape)
 return {'count':a.numel(),'mismatch':int((a!=b).sum()),'max_abs':float((a.float()-b.float()).abs().max())}
try:
 torch.npu.set_device(8);register_fla_legacy_ops()
 train=Path('/home/w00848261/verl/.zero_diff/runs/proxy-megatron-005/trace/rank0')
 infer=Path('/home/w00848261/verl/.zero_diff/runs/proxy-vllm-008/trace')
 prefix='chunk0.module.module.language_model.decoder.layers.0.self_attention.'
 projection=torch.load(train/(prefix+'in_proj.000.pt'),weights_only=True)['output'][0].transpose(0,1).npu()
 qkv,z,b,a=projection.split([4096,2048,16,16],dim=-1)
 ckpt=Path('/mnt/weight/Qwen3.5-35B-A3B');index=json.loads((ckpt/'model.safetensors.index.json').read_text())['weight_map']
 def weight(s):
  name='model.language_model.layers.0.linear_attn.'+s
  with safe_open(str(ckpt/index[name]),framework='pt',device='cpu') as f:return f.get_tensor(name).npu()
 w=weight('conv1d.weight').squeeze(1)
 w=torch.cat([w[:1024],w[2048:3072],w[4096:6144]]).contiguous()
 A_log=weight('A_log')[:16];dt=weight('dt_bias')[:16]
 with torch.no_grad():
  conv,_=fla_npu_causal_conv1d(qkv.contiguous(),w,activation='silu')
  ref=torch.load(train/(prefix+'out_norm.000.pt'),weights_only=True)['inputs'][0].reshape(1,144,16,128)
  inf=torch.load(infer/'language_model.model.layers.0.linear_attn.norm.016.pt',weights_only=True)['inputs'][0].reshape(1,144,4,128)
  for combined in (False,True):
   q,k,v=conv.split([1024,1024,2048],dim=-1)
   if combined:
    qk=l2norm(torch.cat([q,k],dim=-1).reshape(1,144,16,128).contiguous())
    q,k=qk.split(8,dim=2)
   else:
    q=l2norm(q.reshape(1,144,8,128).contiguous())
    k=l2norm(k.reshape(1,144,8,128).contiguous())
   q=q.repeat_interleave(2,dim=2).contiguous();k=k.repeat_interleave(2,dim=2).contiguous()
   v=v.reshape(1,144,16,128).contiguous()
   g,beta=GatedDeltaNet._compute_g_and_beta(None,A_log,dt,a,b)
   for packed in (False,True):
    cu=torch.tensor([0,144],dtype=torch.int32,device='npu') if packed else None
    out,_=flash_gated_delta_rule(q,k,v,g,beta,scale=128**-0.5,use_qk_l2norm_in_kernel=False,cu_seqlens=cu)
    report[f'combined{combined}_packed{packed}']={'vs_native':cmp(out,ref),'vs_infer_first4':cmp(out[:,:,:4],inf)}
  actual=torch.load(train/'chunk0.module.module.language_model.decoder.layers.0.self_attention.kernel.000.pt',weights_only=True)
  report['kernel_input_shapes']={'args':[list(t.shape) for t in actual['inputs']],'kwargs':{k:(list(v.shape) if isinstance(v,torch.Tensor) else v) for k,v in actual['kwargs'].items()}}
  for name,got,want in zip(('q','k','v'),(q,k,v),actual['inputs']):report['input_'+name]=cmp(got,want)
  for name,got in [('g',g),('beta',beta)]:report['input_'+name]=cmp(got,actual['kwargs'][name])
  kwargs={k:(v.npu() if isinstance(v,torch.Tensor) else v) for k,v in actual['kwargs'].items()}
  again,_=flash_gated_delta_rule(*(v.npu() for v in actual['inputs']),**kwargs)
  report['identical_inputs_kernel_replay']=cmp(again,actual['output'][0])
  from types import SimpleNamespace
  owner=SimpleNamespace(qk_dim_local_tp=1024,v_dim_local_tp=2048,cp_size=1,key_head_dim=128,value_head_dim=128,use_qk_l2norm=True,num_value_heads=32,num_key_heads=16)
  prepared=GatedDeltaNet._prepare_qkv_for_gated_delta_rule(owner,conv,z.reshape(1,144,16,128),b,a,1,144)
  for name,got,want in zip(('q','k','v'),prepared[:3],actual['inputs']):report['compiled_prepare_'+name]=cmp(got,want)

except Exception:report['error']=traceback.format_exc()
finally:
 (root/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
