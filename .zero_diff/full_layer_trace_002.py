"""Rank-zero layer-boundary trace for the first full-model PD divergence."""
def install_trace(model):
    from pathlib import Path
    import re
    import torch
    import torch.distributed as dist
    if dist.get_rank() != 0:
        return []
    root=Path('/home/w00848261/verl/.zero_diff/runs/full-inference-002/trace/rank0')
    root.mkdir(parents=True,exist_ok=True)
    def tree(value):
        if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
        if isinstance(value,(tuple,list)):return tuple(tree(v) for v in value)
        return None
    def make_hook(name):
        counter=[0]
        def hook(module,args,output):
            step=counter[0];counter[0]+=1
            torch.save({'inputs':tree(args),'output':tree(output)},root/f'{name}.{step:03d}.pt')
        return hook
    names=[];handles=[]
    for name,module in model.named_modules():
        if re.search(r'\.layers\.\d+$',name):
            handles.append(module.register_forward_hook(make_hook(name)))
            names.append(name)
    model._zero_diff_trace_handles=handles
    return names
