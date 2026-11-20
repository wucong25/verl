import json,traceback
from pathlib import Path

def main():
    from vllm import LLM,SamplingParams
    root=Path('/home/w00848261/verl/.zero_diff/runs/proxy8-prefix-007');root.mkdir(exist_ok=True,parents=True)
    original=json.loads(Path('/home/w00848261/verl/.zero_diff/runs/full-inference-002/result.json').read_text())['cases']
    report={'scope':'8-layer real-weight first-divergence replay','prefix_length':1024}
    try:
        llm=LLM(model='/home/w00848261/verl/.zero_diff/proxy-8layer',trust_remote_code=True,tensor_parallel_size=8,enable_expert_parallel=True,dtype='bfloat16',enforce_eager=True,max_model_len=1152,max_num_seqs=4,max_num_batched_tokens=2304,gpu_memory_utilization=0.3,enable_prefix_caching=False,enable_chunked_prefill=False,limit_mm_per_prompt={'image':0,'video':0},additional_config={'rl_config':{'enabled':True,'training_tensor_parallel_size':2,'training_router_math':True,'training_expert_math':True,'training_attention_math':True,'training_logprob_math':True}},seed=42)
        from prefix_trace_007 import install_trace
        report['modules']=llm.apply_model(install_trace)
        params=SamplingParams(temperature=0,max_tokens=2,ignore_eos=True,logprobs=0)
        single=llm.generate([{'prompt_token_ids':original[3]['tokens']}],params,use_tqdm=False)[0]
        batched=llm.generate([{'prompt_token_ids':c['tokens']} for c in original],params,use_tqdm=False)
        report['batch_next_token']=list(batched[3].outputs[0].token_ids)
        report['single_next_token']=list(single.outputs[0].token_ids)
        llm.generate([{'prompt_token_ids':original[3]['tokens']+original[3]['response']}],SamplingParams(temperature=0,max_tokens=1,ignore_eos=True,prompt_logprobs=0),use_tqdm=False)
        report['completed']=True
    except Exception:report['error']=traceback.format_exc()
    finally:
        (root/'result.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
