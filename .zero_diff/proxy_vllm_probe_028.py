"""Compare rollout token logprobs against same-token prefill on the proxy."""
import json,traceback,time
from pathlib import Path

def main():
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    from vllm import LLM,SamplingParams
    root=Path('/home/w00848261/verl/.zero_diff/runs/proxy-vllm-028')
    root.mkdir(parents=True,exist_ok=True)
    report={'scope':'4-layer inference PD diagnostic only','tp':8,'batch_invariant':True,'source_data':'dapo-math-17k','prompt_prefix_tokens':128,'response_tokens':16}
    (root/'config.json').write_text(json.dumps(report,indent=2))
    try:
        model='/home/w00848261/verl/.zero_diff/proxy-4layer'
        tokenizer=AutoTokenizer.from_pretrained(model,trust_remote_code=True,local_files_only=True)
        row=pq.read_table('/mnt/share/datasets/dapo-math-17k/dapo-math-17k.parquet',columns=['prompt']).slice(0,1).to_pylist()[0]
        rendered=tokenizer.apply_chat_template(row['prompt'],tokenize=False,add_generation_prompt=True)
        tokens=tokenizer.encode(rendered,add_special_tokens=False)[:128]
        llm=LLM(model=model,trust_remote_code=True,tensor_parallel_size=8,enable_expert_parallel=True,dtype='bfloat16',enforce_eager=True,max_model_len=256,max_num_seqs=2,max_num_batched_tokens=256,gpu_memory_utilization=0.3,enable_prefix_caching=False,enable_chunked_prefill=False,limit_mm_per_prompt={'image':0,'video':0},additional_config={'rl_config':{'enabled':True,'training_tensor_parallel_size':2,'training_router_math':True,'training_expert_math':True,'training_attention_math':True,'training_logprob_math':True}},seed=42)
        from trace_hooks_028 import install_trace
        report["trace_modules"]=llm.apply_model(install_trace)
        params=SamplingParams(temperature=1.0,top_p=1.0,top_k=-1,max_tokens=16,ignore_eos=True,logprobs=0,seed=42)
        started=time.time()
        first=llm.generate([{'prompt_token_ids':tokens}],params,use_tqdm=False)[0]
        sample=first.outputs[0]
        response=list(sample.token_ids)
        old=[entry[token].logprob for entry,token in zip(sample.logprobs,response)]
        second=llm.generate([{'prompt_token_ids':tokens+response}],SamplingParams(temperature=0,max_tokens=1,prompt_logprobs=0),use_tqdm=False)[0]
        ref=[second.prompt_logprobs[len(tokens)+i][token].logprob for i,token in enumerate(response)]
        fixed=json.loads(Path('/home/w00848261/verl/.zero_diff/runs/proxy-vllm-013/result.json').read_text())
        forced=llm.generate([{'prompt_token_ids':fixed['tokens']+fixed['response']}],SamplingParams(temperature=0,max_tokens=1,prompt_logprobs=0),use_tqdm=False)[0]
        report['fixed_reference_logprobs']=[forced.prompt_logprobs[len(fixed['tokens'])+i][token].logprob for i,token in enumerate(fixed['response'])]
        report['expert_parallel']=True
        diffs=[abs(a-b) for a,b in zip(old,ref)]
        report.update(tokens=tokens,response=response,decode_logprobs=old,prefill_logprobs=ref,mismatch=sum(a!=b for a,b in zip(old,ref)),max_abs=max(diffs),count=len(diffs),elapsed=time.time()-started)
    except Exception:
        report['error']=traceback.format_exc()
    finally:
        (root/'result.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':
    main()
