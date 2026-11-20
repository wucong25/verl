def install_trace(model):
    from pathlib import Path
    import re,torch
    import torch.distributed as dist
    if dist.get_rank()!=0:return []
    root=Path('/home/w00848261/verl/.zero_diff/runs/proxy8-solve-012/trace');root.mkdir(parents=True,exist_ok=True)
    def tree(value):
        if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
        if isinstance(value,(list,tuple)):return tuple(tree(v) for v in value)
        if isinstance(value,dict):return {k:tree(v) for k,v in value.items()}
        return value
    def make_hook(name):
        counter=[0]
        def hook(module,args,kwargs,output):
            step=counter[0];counter[0]+=1
            torch.save({'args':tree(args),'kwargs':tree(kwargs),'output':tree(output)},root/f'{name}.{step:03d}.pt')
        return hook
    handles=[];names=[]
    for name,module in model.named_modules():
        if name.endswith('.embed_tokens') or re.search(r'\.layers\.\d+$',name):
            handles.append(module.register_forward_hook(make_hook(name),with_kwargs=True));names.append(name)
    model._zero_diff_trace_handles=handles
    return names
