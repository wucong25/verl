import torch,torch_npu
from pathlib import Path
torch.npu.set_device(8)
from solve_kernel_leaf_005 import install_trace
install_trace(None)
from mindspeed_ops.api.triton.solve_tril import solve_tril
A=torch.load('/home/w00848261/verl/.zero_diff/runs/gdn-stage-replay-001/00-chunk_scaled_dot_kkt_fwd.pt',weights_only=True)['output'].npu()
y=solve_tril(A,output_dtype=torch.bfloat16)
print('done',y.shape)
