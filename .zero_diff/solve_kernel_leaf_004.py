def install_trace(model):
    import torch,importlib
    import torch.distributed as dist
    from pathlib import Path
    root=Path('/home/w00848261/verl/.zero_diff/runs/gdn-solve-leaf-004/trace')/'rank0'
    root.mkdir(parents=True,exist_ok=True)
    mod=importlib.import_module('mindspeed_ops.arch32.triton.solve_tril')
    class Trace:
        def __init__(self,kernel,name):self.kernel=kernel;self.name=name;self.count=0
        def __getitem__(self,grid):
            runner=self.kernel[grid]
            def run(*args,**kwargs):
                out=runner(*args,**kwargs)
                if self.count<1:
                    output=args[1] if args else kwargs['Ai']
                    torch.save({'output':output.cpu().clone(),'grid':grid,'kwargs':{k:v for k,v in kwargs.items() if not isinstance(v,torch.Tensor)}},root/f'{self.name}.pt')
                self.count+=1
                return out
            return run
    for name in ('solve_tril_16x16_loop_kernel_paral_v3','merge_16x16_to_32x32_loop_inverse_kernel','merge_32x32_to_64x64_loop_inverse_kernel'):
        setattr(mod,name,Trace(getattr(mod,name),name))
    return ['capture first solve intermediate buffers']
