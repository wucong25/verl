def install_trace(model):
    from pathlib import Path
    import torch,importlib
    import torch.distributed as dist
    from prefix_trace_012 import install_trace as boundaries
    names=boundaries(model)
    mod=importlib.import_module('mindspeed_ops.api.triton.solve_tril');original=mod.solve_tril
    root=Path('/home/w00848261/verl/.zero_diff/runs/proxy8-solve-012/trace')/f'rank{dist.get_rank()}'
    root.mkdir(parents=True,exist_ok=True);count=[0]
    def solve(A,cu_seqlens=None,output_dtype=torch.float32):
        if cu_seqlens is not None:return original(A,cu_seqlens=cu_seqlens,output_dtype=output_dtype)
        length=A.shape[1]
        guard=A.new_zeros((A.shape[0],64,A.shape[2],A.shape[3]))
        padded=torch.cat((A,guard),dim=1)
        result=original(padded,cu_seqlens=None,output_dtype=output_dtype)[:,:length].contiguous()
        if count[0]==0:torch.save({'A':A.cpu(),'output':result.cpu()},root/'first-solve.pt')
        count[0]+=1
        return result
    mod.solve_tril=solve
    return names
