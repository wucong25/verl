"""Temporary eager forward hooks for first-divergence localization."""
def install_trace(model):
    from pathlib import Path
    import torch
    import torch.distributed as dist
    if dist.get_rank()!=0:
        return []
    root=Path('/home/w00848261/verl/.zero_diff/runs/proxy-vllm-011/trace')
    root.mkdir(parents=True,exist_ok=True)
    selected={'embed_tokens','input_layernorm','post_attention_layernorm','linear_attn','self_attn','mlp','in_proj_qkvz','in_proj_qkv','in_proj_z','in_proj_ba','out_proj','norm','gate','experts','qkv_proj','o_proj'}
    names=[]; handles=[]
    def cpu_tree(value):
        if isinstance(value,torch.Tensor):
            return value.detach().cpu().clone()
        if isinstance(value,(tuple,list)):
            return tuple(cpu_tree(t) for t in value)
        if value is None or isinstance(value,(int,float,str,bool)):
            return value
        return str(type(value))
    def make_hook(name):
        counter=[0]
        def hook(module,args,output):
            step=counter[0]; counter[0]+=1
            if step == 0:
                for n,p in module.named_parameters(recurse=False):
                    if p.numel()<=4096:torch.save(cpu_tree(p),root/f'{name}.{n}.pt')
            torch.save({'inputs':cpu_tree(args),'output':cpu_tree(output)},root/f'{name}.{step:03d}.pt')
        return hook
    for name,module in model.named_modules():
        if 'visual' not in name and name.rsplit('.',1)[-1] in selected:
            names.append(name)
            handles.append(module.register_forward_hook(make_hook(name)))
    model._zero_diff_trace_handles=handles
    return names
