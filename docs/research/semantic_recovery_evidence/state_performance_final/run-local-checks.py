import hashlib,json,os,pathlib,subprocess,time
root=pathlib.Path('/private/tmp/stable-envs-semantic-recovery')
os.chdir(root)
out=root/'docs/research/semantic_recovery_evidence/state_performance_final'
cold=root/'docs/research/semantic_recovery_evidence/state_performance_cold_cpu'
while len(list(cold.glob('*/result.json')))!=8:
 time.sleep(10)
assert all(json.loads(p.read_text())['status']=='COMPLETE' for p in cold.glob('*/result.json'))
env=dict(os.environ,PYTHONPATH='.:/Users/samib/stable-worldmodel',SWM_ROOT='/Users/samib/stable-worldmodel',MUJOCO_GL='disable',SDL_VIDEODRIVER='dummy',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
commands=[
 ('dmc-integration',['.venv/bin/python','scripts/benchmarks/native_dmc_integration.py','--cpu','docs/research/semantic_recovery_evidence/state_performance_cpu','--output','docs/research/semantic_recovery_evidence/state_performance_dmc_native']),
 ('shared-tests',['.venv/bin/python','-m','pytest','-q','tests/unit/test_full_state_contract.py','tests/unit/test_semantic_harness.py','tests/unit/test_replacement_readiness.py','tests/unit/test_swm_matrix.py','tests/unit/test_repo_contracts.py','tests/unit/test_variation_maps_newcells.py','tests/unit/test_ogbench_manipulation_acceptance.py','tests/unit/test_state_ops_benchmark.py']),
 ('ruff-check',['.venv/bin/ruff','check','scripts/benchmarks','scripts/research_native','tests/unit/test_state_ops_benchmark.py']),
 ('ruff-format',['.venv/bin/ruff','format','--check','scripts/benchmarks','scripts/research_native','tests/unit/test_state_ops_benchmark.py']),
 ('preservation',['/private/tmp/craftax-recovery-312/bin/python','scripts/recovery/verify_rocket_audit.py','--output',str(out/'accepted-preservation.json')]),
]
records=[]
for name,command in commands:
 path=out/(name+'.log')
 start=time.time()
 with path.open('xb') as stream:
  result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,env=env)
 records.append(dict(name=name,command=command,working_directory=str(root),exit_code=result.returncode,elapsed_seconds=time.time()-start,log=str(path.relative_to(root)),log_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),stdout_tail=path.read_text()[-1600:]))
 (out/'commands.json').write_text(json.dumps(records,indent=2)+'\n')
 print(name,result.returncode,flush=True)
 assert result.returncode==0,(name,path)
start=json.loads((root/'docs/research/semantic_recovery_evidence/state_performance_start/start.json').read_text())
for kind in ['artifact_hashes','production_hashes']:
 for path,expected in start[kind].items():
  assert hashlib.sha256((root/path).read_bytes()).hexdigest()==expected,(kind,path)
for name in ['semantic_recovery_matrix.json','semantic_recovery_implementation_map.json']:
 path='docs/research/'+name
 assert (root/path).read_bytes()==subprocess.check_output(['git','show',start['starting_HEAD']+':'+path])
(out/'starting-checkpoint-preserved.json').write_text(json.dumps(dict(accepted_artifact_hashes=len(start['artifact_hashes']),production_hashes=len(start['production_hashes']),matrix_and_map_byte_identical_to_start=True,counts={'VERIFIED':39,'INCOMPLETE':3},production_changes=[],shared_contract_changes=[],push_performed=False),indent=2)+'\n')
print('ALL_LOCAL_FINAL_CHECKS_PASS',flush=True)
