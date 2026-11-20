from pathlib import Path
import hashlib,json
import pyarrow.parquet as pq
from transformers import AutoTokenizer
model='/mnt/weight/Qwen3.5-35B-A3B/'
tokenizer=AutoTokenizer.from_pretrained(model,trust_remote_code=True,local_files_only=True)
rows=pq.read_table('/mnt/share/datasets/dapo-math-17k/dapo-math-17k.parquet',columns=['prompt']).to_pylist()
remaining=[65,257,513,1024];cases=[]
for index,row in enumerate(rows):
 text=tokenizer.apply_chat_template(row['prompt'],tokenize=False,add_generation_prompt=True)
 tokens=tokenizer.encode(text,add_special_tokens=False)
 for length in list(remaining):
  if len(tokens)>=length:
   cases.append({'dataset_row':index,'original_length':len(tokens),'prefix_length':length,'prompt_sha256':hashlib.sha256(text.encode()).hexdigest(),'tokens':tokens[:length]});remaining.remove(length)
 if not remaining:break
root=Path('/home/w00848261/verl/.zero_diff');(root/'case_manifest_full_dataset.json').write_text(json.dumps({'cases':cases,'unavailable_lengths':remaining},indent=2))
print([{k:v for k,v in c.items() if k!='tokens'} for c in cases]);print('unavailable',remaining)
