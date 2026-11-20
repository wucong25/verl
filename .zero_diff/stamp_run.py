import datetime,hashlib,json,subprocess,sys
from pathlib import Path
run=Path('/home/w00848261/verl/.zero_diff/runs')/sys.argv[1]
run.mkdir(parents=True,exist_ok=True)
files=[Path(s) for s in sys.argv[2:]] + [Path(__file__)]
repos={str(p):{'commit':subprocess.check_output(['git','-C',str(p),'rev-parse','HEAD'],text=True).strip(),'status':subprocess.check_output(['git','-C',str(p),'status','--short'],text=True)} for p in map(Path,['/home/w00848261/verl','/vllm','/vllm-ascend'])}
for repo, paths in [('/home/w00848261/verl', ['verl', 'tests', 'examples', 'scripts']), ('/vllm', ['vllm', 'tests']), ('/vllm-ascend', ['vllm_ascend', 'tests'])]:
    tracked = subprocess.check_output(['git', '-C', repo, 'diff', '--name-only', '--diff-filter=ACMRT'], text=True).splitlines()
    untracked = subprocess.check_output(['git', '-C', repo, 'ls-files', '--others', '--exclude-standard', '--', *paths], text=True).splitlines()
    for name in tracked + untracked:
        path = Path(repo) / name
        if path.is_file() and path.suffix in ('.py', '.sh', '.yaml', '.yml', '.toml'):
            files.append(path)
data={'recorded_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'image':'sha256:bdc20cc234a54139fe8b66c871e7b7c625bf0288b783c77a01ec14b00a269c6c','repos':repos,'files':{str(p.resolve()):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},'note':'Source snapshot at record time; consult launch log for execution timestamps.'}
(run/'source.json').write_text(json.dumps(data,indent=2))

# Preserve the exact launch-time contents as well as their hashes.
roots = [(Path('/home/w00848261/verl'), 'verl'), (Path('/vllm'), 'vllm'), (Path('/vllm-ascend'), 'vllm-ascend')]
for file in set(files):
    resolved = file.resolve()
    for root, name in roots:
        if resolved.is_relative_to(root):
            destination = run / 'source-tree' / name / resolved.relative_to(root)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(resolved.read_bytes())
            break
    else:
        raise ValueError(f'Unexpected source path: {resolved}')
