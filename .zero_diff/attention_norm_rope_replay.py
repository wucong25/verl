from pathlib import Path
import torch,torch_npu,json
T=Path('.zero_diff/runs/proxy-megatron-011/trace');I=Path('.zero_diff/runs/proxy-vllm-020/trace');p='chunk0.module.module.language_model.decoder.layers.1.self_attention';q='language_model.model.layers.3.self_attn'
torch.npu.set_device(9)
def load(root,name):return torch.load(root/name,weights_only=False)
def cmp(a,b):
 a,b=a.cpu(),b.cpu();assert a.shape==b.shape,(a.shape,b.shape)
 return {'n':a.numel(),'mismatch':int((a!=b).sum()),'max':float((a.float()-b.float()).abs().max())}
report={}
inv=1/(10000000**(torch.arange(0,64,2,dtype=torch.float32,device='npu')/64))
freq=torch.arange(144,dtype=torch.float32,device='npu')[:,None]*inv[None,:]
cos=freq.cos().bfloat16().repeat(1,2)[:,None,:];sin=freq.sin().bfloat16().repeat(1,2)[:,None,:]
for nr in (8,9):
 core=load(T/f'rank{nr}',p+'.core_attention.000.pt')
 for j,(part,name) in enumerate((('q_layernorm','q'),('k_layernorm','k'))):
  nc=load(T/f'rank{nr}',p+'.'+part+'.000.pt');x=nc['inputs'][0].squeeze(1).npu();gamma=load(T/f'rank{nr}',p+'.'+part+'.weight.pt').npu().float()+1
  norm=torch_npu.npu_rms_norm(x,gamma,epsilon=1e-6)[0];report[f'{nr}_{name}_norm']=cmp(norm,nc['output'].squeeze(1))
  rot=norm[...,:64];half=torch.cat([-rot[...,32:],rot[...,:32]],dim=-1)
  out=torch.cat([rot*cos+half*sin,norm[...,64:]],dim=-1)
  report[f'{nr}_{name}_rope']=cmp(out,core['inputs'][j])
 qn,kn,vn=[x.npu().contiguous() for x in core['inputs'][:3]]
 mask=torch.ones(2048,2048,dtype=torch.bool,device='npu').triu(1)
 attn=torch_npu.npu_fused_infer_attention_score(query=qn,key=kn,value=vn,num_heads=8,num_key_value_heads=1,input_layout='TND',scale=256**-0.5,sparse_mode=3,atten_mask=mask,actual_seq_lengths=[144],actual_seq_lengths_kv=[144])[0]
 report[f'{nr}_FIA_nativeQKV']=cmp(attn,core['output'])
print(json.dumps(report,indent=2),flush=True)
