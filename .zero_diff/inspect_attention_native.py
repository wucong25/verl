import inspect
import megatron_adaptor
from megatron.core.extensions.transformer_engine import TEDotProductAttention
print('forward_source',inspect.getsourcefile(TEDotProductAttention.forward))
print(inspect.getsource(TEDotProductAttention.forward))
from megatron.bridge.models.qwen_vl.modelling_qwen3_vl.attention import Qwen3VLSelfAttention
print('qwen_source',inspect.getsourcefile(Qwen3VLSelfAttention))
