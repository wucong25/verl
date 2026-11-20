"""Model-level PD and mixed-length batch invariance on four real DAPO rows."""
import json, traceback, time, math
from pathlib import Path


def main():
    from vllm import LLM, SamplingParams
    root=Path('/home/w00848261/verl/.zero_diff/runs/full-inference-002');root.mkdir(parents=True,exist_ok=True)
    cases=json.loads(Path('/home/w00848261/verl/.zero_diff/case_manifest_full_dataset.json').read_text())['cases']
    report={'scope':'40-layer diagnostic; actual DAPO prompt prefixes; mixed-batch PD','cases':[]}
    try:
        llm=LLM(model='/mnt/weight/Qwen3.5-35B-A3B',trust_remote_code=True,tensor_parallel_size=8,enable_expert_parallel=True,dtype='bfloat16',enforce_eager=True,max_model_len=1152,max_num_seqs=4,max_num_batched_tokens=2304,gpu_memory_utilization=0.3,enable_prefix_caching=False,enable_chunked_prefill=False,limit_mm_per_prompt={'image':0,'video':0},additional_config={'rl_config':{'enabled':True,'training_tensor_parallel_size':2,'training_router_math':True,'training_expert_math':True,'training_attention_math':True,'training_logprob_math':True}},seed=42)
        from full_layer_trace_002 import install_trace
        report['trace_modules']=llm.apply_model(install_trace)
        params=SamplingParams(temperature=1.0,top_p=1.0,top_k=-1,max_tokens=16,ignore_eos=True,logprobs=0,seed=42)
        started=time.time();outputs=llm.generate([{'prompt_token_ids':c['tokens']} for c in cases],params,use_tqdm=False)
        forced=SamplingParams(temperature=0,max_tokens=1,ignore_eos=True,prompt_logprobs=0)
        histories=[]
        for c,out in zip(cases,outputs):
            response=list(out.outputs[0].token_ids)
            record=dict(c,response=response,decode_logprobs=[p[t].logprob for p,t in zip(out.outputs[0].logprobs,response)])
            report['cases'].append(record);histories.append({'prompt_token_ids':c['tokens']+response})
        batched=llm.generate(histories,forced,use_tqdm=False)
        for c,history,prefill in zip(report['cases'],histories,batched):
            single=llm.generate([history],forced,use_tqdm=False)[0]
            def lp(out):return [out.prompt_logprobs[len(c['tokens'])+i][t].logprob for i,t in enumerate(c['response'])]
            c['batch_prefill_logprobs']=lp(prefill);c['single_prefill_logprobs']=lp(single)
            c['finite']=all(math.isfinite(x) for name in ('decode_logprobs','batch_prefill_logprobs','single_prefill_logprobs') for x in c[name])
            c['pd_mismatch']=sum(a!=b for a,b in zip(c['decode_logprobs'],c['batch_prefill_logprobs']))
            c['batch_mismatch']=sum(a!=b for a,b in zip(c['batch_prefill_logprobs'],c['single_prefill_logprobs']))
            c['pd_max_abs']=max(abs(a-b) for a,b in zip(c['decode_logprobs'],c['batch_prefill_logprobs']))
            c['batch_max_abs']=max(abs(a-b) for a,b in zip(c['batch_prefill_logprobs'],c['single_prefill_logprobs']))
        report['elapsed']=time.time()-started
        report['passed']=len(report['cases'])==4 and all(c['finite'] and len(c['decode_logprobs'])==16 and c['pd_mismatch']==c['batch_mismatch']==0 for c in report['cases'])
    except Exception:
        report['error']=traceback.format_exc()
    finally:
        (root/'result.json').write_text(json.dumps(report,indent=2))
        print(json.dumps({k:v for k,v in report.items() if k!='cases'},indent=2),flush=True)
        print([{k:v for k,v in c.items() if k not in ('tokens','response','decode_logprobs','batch_prefill_logprobs','single_prefill_logprobs')} for c in report['cases']],flush=True)

if __name__=='__main__':main()
