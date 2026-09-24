"""Archive-generation record. Run only in a new staging location; never overwrite a sealed bundle."""
import gzip,hashlib,json,subprocess,tarfile
from pathlib import Path
root=Path('/private/tmp/stable-envs-semantic-recovery');commit='b6a8242f5ff5a9cae5e9e883a15611d779c2c99c';bundle=Path('/private/tmp/stable-envs-craftax-'+commit[:8]+'-gpu');bundle.mkdir()
def archive(repo,sha,dest):
    with dest.open('wb') as out:
        with gzip.GzipFile(filename='',mode='wb',fileobj=out,mtime=0) as zipfile:
            result=subprocess.Popen(['git','archive','--format=tar',sha],cwd=repo,stdout=subprocess.PIPE)
            while chunk:=result.stdout.read(1048576):zipfile.write(chunk)
            assert result.wait()==0
archive(root,commit,bundle/'source.tar.gz');oracle='176d54532b3c0a251a4f509b74f2fdacc143a63d';archive('/Users/samib/stable-worldmodel',oracle,bundle/'oracle.tar.gz')
runner=root/'scripts/recovery/oscar_craftax_gate.sbatch';(bundle/'gate.sbatch').write_bytes(runner.read_bytes())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
metadata={'status':'PREPARED_NOT_UPLOADED_NOT_SUBMITTED','source_commit':commit,'production_commit':'2953735e5eabae96b796843a376abd7150781de4','oracle_commit':oracle,'bundle_local_path':str(bundle),'remote_target':'/oscar/scratch/sboughan/se-semantic-recovery/'+bundle.name,'archive_sha256':{n:sha(bundle/n) for n in ['source.tar.gz','oracle.tar.gz']},'runner_sha256':sha(bundle/'gate.sbatch'),'source_file_hashes':{str(p.relative_to(root)):sha(p) for p in [runner,root/'scripts/recovery/craftax_gate.py',root/'scripts/recovery/profiles/craftax-gpu.in',root/'scripts/recovery/profiles/craftax-gpu.txt',root/'tests/unit/test_craftax_semantics.py',root/'tests/unit/test_craftax_rendering.py',root/'tests/helpers/reference/craftax_source.py']},'expected_registration_test_counts':{'CraftaxClassicPixels':28,'CraftaxClassicSymbolic':17,'CraftaxPixels':28,'CraftaxSymbolic':19},'expected_per_registration_arrays':['reset-0.npz','reset-17.npz','reset-42.npz','mine_tree.npz','craft.npz','combat.npz','damage.npz','night.npz','terminal.npz','near_limit.npz','stochastic_grass.npz'],'expected_per_registration_other':['profile.json','schema.json','result.json','SHA256SUMS'],'expected_external_files':['{name}.xml','{name}-tests.log','{name}-arrays.log','freeze.txt','install.log','nvidia-smi.txt','bundle.json','SHA256SUMS','GPU-SHA256SUMS'],'registration_terminal_marker':'CRAFTAX_REGISTRATION_ACCEPTANCE_PASS {name} {job_id}','array_terminal_marker':'CRAFTAX_{NAME_UPPER}_GPU_GATE_PASS','family_terminal_marker':'CRAFTAX_FAMILY_GPU_GATE_PASS {job_id}','required_gpu_backend':'JAX gpu and torch CUDA both available','python':'3.12, actual patch recorded on allocated node','dependencies':'fully hashed craftax-gpu.txt, uv pip sync --require-hashes --torch-backend cu126','cross_device_equality':'NOT_CLAIMED'}
with tarfile.open(bundle/'source.tar.gz') as archive:
    files=archive.getnames();assert all(p in files for p in metadata['source_file_hashes'])
    for name in metadata['expected_registration_test_counts']:
        for case in ['mine_tree','craft','combat','damage','night','terminal','near_limit','stochastic_grass']:
            assert 'docs/research/semantic_recovery_evidence/craftax_source_audit/'+name+'Wrapper-'+case+'.npz' in files
metadata['source_archive_members']=len(files)
(bundle/'bundle.json').write_text(json.dumps(metadata,indent=2)+'\n')
(bundle/'SHA256SUMS').write_text(''.join(sha(p)+'  '+p.name+'\n' for p in sorted(bundle.iterdir()) if p.is_file()))
evidence=root/'docs/research/semantic_recovery_evidence/craftax_gpu_prepared'
for name in ['bundle.json','SHA256SUMS','gate.sbatch']:
    target=evidence/name;assert not target.exists();target.write_bytes((bundle/name).read_bytes())
print(json.dumps({'bundle':str(bundle),'sha256':metadata['archive_sha256'],'source_commit':commit,'members':len(files)},indent=2))
