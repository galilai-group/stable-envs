"""Historical record builder; do not rerun over sealed outputs."""
import collections, hashlib, json, subprocess, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
root=Path('/private/tmp/stable-envs-semantic-recovery');e=root/'docs/research/semantic_recovery_evidence/craftax_cpu';g=root/'docs/research/semantic_recovery_evidence/craftax_gpu_prepared'
names=['CraftaxClassicPixels','CraftaxClassicSymbolic','CraftaxPixels','CraftaxSymbolic'];expected=dict(zip(names,[28,17,28,19]));counts=collections.Counter()
cases=ET.parse(e/'final-acceptance.xml').findall('.//testcase');assert len(cases)==92
for case in cases:
    assert not any(case.find(t) is not None for t in ['failure','error','skipped'])
    test=case.attrib['name'];name=next((n for n in names if n in test),'CraftaxSymbolic' if 'full_symbolic_preserves' in test else None);assert name,test;counts[name]+=1
assert dict(counts)==expected,counts
bundle=json.loads((g/'bundle.json').read_text())
result={}
for name in names:
    path=e/'final'/name;r=json.loads((path/'result.json').read_text());assert r['status']=='PASS' and len(r['cases'])==11
    for line in (path/'SHA256SUMS').read_text().splitlines():
        sha,rel=line.split(maxsplit=1);assert hashlib.sha256((path/rel).read_bytes()).hexdigest()==sha
    for case in r['cases']:
        assert case['state_exact'] and case['required_output_exact'] and case['nonmutation']
        with np.load(path/case['array_file'],allow_pickle=False) as a:
            for key in a.files:
                if key.startswith('source_engine/'):
                    np.testing.assert_array_equal(a[key],a['state/engine/'+key[len('source_engine/'):]][0])
            np.testing.assert_array_equal(a['source_wrapper_key'],a['state/rng/wrapper_key'][0])
            if 'Pixels' in name:np.testing.assert_array_equal(a['source_frame'],a['full_frame']);np.testing.assert_array_equal(a['full_frame'],a['render_frame'])
            else:np.testing.assert_array_equal(a['source_output'],a['candidate_output'])
    profile=r['profile'];schema=json.loads((path/'schema.json').read_text())
    record={'historical_id':'swm/'+name+'-v1','status':'CPU_PASS_GPU_PENDING','cpu_tests':counts[name],'exact_array_cases':11,'actual_arrays_independently_rechecked':True,'gpu_job':None,'gpu_status':'BLOCKED_SSH_AUTHENTICATION_NOT_SUBMITTED','source_commit':bundle['source_commit'],'production_commit':bundle['production_commit'],'source_archive_sha256':bundle['archive_sha256']['source.tar.gz'],'profile':profile,'schema':str((path/'schema.json').relative_to(root)),'artifacts':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(path.iterdir()) if p.is_file()},'test_xml':{'path':str((e/'final-acceptance.xml').relative_to(root)),'sha256':hashlib.sha256((e/'final-acceptance.xml').read_bytes()).hexdigest()},'promotion':False}
    (e/(name+'-cpu-acceptance.json')).write_text(json.dumps(record,indent=2)+'\n');result[name]=record
cross={}
for engine,a,b in [('Classic','CraftaxClassicPixels','CraftaxClassicSymbolic'),('Full','CraftaxPixels','CraftaxSymbolic')]:
    rows=[]
    for path in sorted((e/'final'/a).glob('*.npz')):
        with np.load(path) as x,np.load(e/'final'/b/path.name) as y:
            differences=[k for k in x.files if k.startswith('source_engine/') and not np.array_equal(x[k],y[k])]
            rows.append({'case':path.name,'source_engine_differences':differences})
    cross[engine]=rows
(e/'cross-presentation-engine-comparison.json').write_text(json.dumps(cross,indent=2)+'\n')
for filename in ['semantic_recovery_matrix.json','semantic_recovery_implementation_map.json']:
    p=root/'docs/research'/filename;j=json.loads(p.read_text())
    for row in j['rows']:
        if row['historical_id'] not in [r['historical_id'] for r in result.values()]:continue
        name=row['historical_id'][4:-3];rec=result[name];path=e/'final'/name
        files=[str((e/(name+'-cpu-acceptance.json')).relative_to(root)),str((path/'result.json').relative_to(root)),str((path/'schema.json').relative_to(root)),str((g/'bundle.json').relative_to(root))]
        if filename=='semantic_recovery_matrix.json':
            row.update(candidate_classes=[name],owner='craftax_recovery',state_contract='Exhaustive named numeric upstream engine State, fixed configuration, owned keys/lifecycle; full symbolic also retains exact source output cache',action_parity='PASS_EXACT_CPU_ALL_SOURCE_IDS',reset_parity='PASS_EXACT_CPU_SEEDS_0_17_42_AND_UNSEEDED_REPLAY',step_parity='PASS_EXACT_CPU_SOURCE_GAMEPLAY_AND_BATCH',reward_parity='PASS_EXACT_CPU_SOURCE_REWARDS_AND_ACHIEVEMENTS',terminated_truncated_parity='PASS_EXACT_CPU_DEATH_LIMIT_NO_AUTORESET',rng_lifecycle='PASS_EXACT_CPU_OWNED_KEYS_INDEPENDENT_RESTORE',variation_parity='SOURCE_EXPOSES_NONE_FIXED_REGISTERED_DEFAULTS',execution_mode='DELEGATED_CPU',adapter_status='IMPLEMENTED_CPU_PASS_GPU_PENDING',renderer_regression='PASS_EXACT_CPU_SOURCE_UINT8_FRAME' if 'Pixels' in name else 'PASS_EXACT_CPU_SYMBOLIC_SOURCE_OUTPUT; PIXEL_RENDER_SOURCE_UNSUPPORTED',oracle_corpus='docs/research/semantic_recovery_evidence/craftax_source_audit',reference_cpu_benchmark='SOURCE_EAGER_JIT_CHARACTERIZED_BOUNDED',stable_cpu_benchmark=str((path/'result.json').relative_to(root)),remaining_issues=['Oscar GPU semantics/rendering acceptance pending SSH authentication; registration remains INCOMPLETE','Legacy frozen crosswalk/atlas classes preserved; new candidate class is separately identified','No cross-device exactness, public eager/vmap, arbitrary parameter or MPS claim'],test='tests/unit/test_craftax_semantics.py; tests/unit/test_craftax_rendering.py',audit_report='docs/research/craftax_recovery_checkpoint.md',recovery_status='CPU_PASS_OSCAR_NOT_SUBMITTED',evidence=files,implementation_commit='2953735e5eabae96b796843a376abd7150781de4',oscar_job=None)
        else:
            row['candidate_classes']=[name];row['upstream']['tested_profile_versions']=rec['profile']['packages'];row['upstream']['version_scope']='Exact pinned CPU profile; GPU locked but not yet executed';row['historical_task_default_kwargs']['candidate_task']='registered no-auto-reset defaults; fixed engine and presentation'
            row['state_schema']={'status':'CPU_PROVED_GPU_PENDING','complete_schema':rec['schema'],'numeric_engine_leaves':49 if 'Classic' in name else 84,'source_cache':'float32[8268] last_symbolic' if name=='CraftaxSymbolic' else None,'note':'Every shape has a leading public batch dimension; uint32 keys encoded losslessly as int64 words'}
            row['restoration_order']=['Validate complete batch exact keys/shapes/dtypes/signature/fixed parameters/key ranges/source invariants/lifecycle','Decode every immutable engine lane; allocate retained keys/lifecycle and full-symbolic source cache','Atomically replace all lanes; no reset, step, world generation or history replay']
            row['action_scaling_clipping_repeat']['summary']='Exact upstream discrete IDs, scalar JIT per batch lane; integer validation, no scaling/clipping';row['action_scaling_clipping_repeat']['parity']='PASS_EXACT_CPU'
            row['reset_rng_semantics'].update(reset_parity='PASS_EXACT_CPU',rng_lifecycle='PASS_EXACT_CPU_WRAPPER_AND_ENGINE_KEYS_THREEFRY_LEGACY_WORDS')
            row['reward_success_terminated_truncated'].update(reward='PASS_EXACT_CPU',lifecycle='PASS_EXACT_CPU_TERMINATED_LIMIT_DEATH_TRUNCATED_FALSE_NO_AUTORESET')
            row['accepted_variation_scope']='NONE_REGISTERED_FIXED_DEFAULTS_CPU_VALIDATED_GPU_PENDING';row['execution_mode']='DELEGATED_CPU';row['evidence_manifest']={'oracle':'docs/research/semantic_recovery_evidence/craftax_source_audit/SHA256SUMS','artifacts':files};row['cpu_semantic_result']='PASS_EXACT_CPU';row['gpu_semantic_result']={'status':'BLOCKED_SSH_AUTHENTICATION_NOT_SUBMITTED','job':None,'note':'Candidate executes upstream JAX on GPU; pending independent registration gate, no GPU acceptance claim'};row['frozen_renderer_result']='PASS_EXACT_CPU_SOURCE_UINT8; legacy atlas contract preserved separately' if 'Pixels' in name else 'PASS_EXACT_CPU_SYMBOLIC; source pixel refusal preserved';row['known_limitations']=['GPU unaccepted; CPU/GPU equality unclaimed','Fixed registered configuration; no Variation, eager public mode, MPS, or alternate rendering sizes','Legacy stable_envs_classes is frozen crosswalk; candidate_classes identifies the new implementation'];row['remaining_gates']=['Oscar independent GPU acceptance after authentication','Download and independently verify all remote arrays, hashes, terminal logs before promotion'];row['closure_code_check']={'production_swm_imports':False,'copied_game_logic':False,'shared_contract_change':False}
    if filename=='semantic_recovery_implementation_map.json':j.setdefault('updates',[]).append({'checkpoint':'craftax_cpu_gpu_pending','report':'docs/research/craftax_recovery_checkpoint.md','counts':{'VERIFIED':35,'INCOMPLETE':7},'promotion':False})
    p.write_text(json.dumps(j,indent=2)+'\n')
summary={'status':'FOUR_CPU_PASS_GPU_AUTH_BLOCKED','source_commit':bundle['source_commit'],'production_commit':bundle['production_commit'],'tests':dict(counts),'total_tests':92,'array_cases':44,'all_prior_accepted_preserved':35,'matrix':{'VERIFIED':35,'INCOMPLETE':7},'gpu_submitted':False,'promoted':[],'source_archive_sha256':bundle['archive_sha256']['source.tar.gz'],'oracle_archive_sha256':bundle['archive_sha256']['oracle.tar.gz'],'cross_presentation_engine_arrays_exact':all(not c['source_engine_differences'] for rows in cross.values() for c in rows)}
(e/'acceptance-summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary,indent=2))
