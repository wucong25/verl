def install_trace(model):
    from pathlib import Path
    import torch,torch.distributed as dist,importlib,json
    from prefix_trace_007 import install_trace as boundaries
    # Reuse boundary hook mechanics with a local root in this diagnostic module.
    root=Path('/home/w00848261/verl/.zero_diff/runs/proxy8-solve-010/trace')/f'rank{dist.get_rank()}'
    root.mkdir(parents=True,exist_ok=True)
    mod=importlib.import_module('mindspeed_ops.api.triton.solve_tril');original=mod.solve_tril
    counts=[0]
    def solve(A,cu_seqlens=None,output_dtype=torch.float32):
        out=original(A,cu_seqlens=cu_seqlens,output_dtype=output_dtype)
        if counts[0]<1:
            step=counts[0];counts[0]+=1
            again=original(A,cu_seqlens=cu_seqlens,output_dtype=output_dtype)
            packed=original(A,cu_seqlens=torch.tensor([0,A.shape[1]],device=A.device,dtype=torch.int32),output_dtype=output_dtype)
            torch.save({'A':A.cpu(),'out':out.cpu(),'again':again.cpu(),'packed':packed.cpu()},root/'solve.pt')
        return out
    mod.solve_tril=solve
    return ['solve first call repeat and packed diagnostic; returns original']
