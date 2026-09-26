import pathlib, subprocess, json, collections, hashlib, shutil
root=pathlib.Path('/private/tmp/stable-envs-semantic-recovery'); base='31153d38c800f44e5c4c0ae1acdd0edca4c5f656'; head='3d0a799d329bb00fbecbf708e970be18a14ce611'
def git(*args): return subprocess.check_output(['git',*args],cwd=root)
def tree(rev):
 out={}
 for row in git('ls-tree','-r','-l','-z',rev).split(b'\0'):
  if row:
   meta,path=row.split(b'\t'); out[path.decode()]=int(meta.split()[3])
 return out
a,b=tree(base),tree(head); added={p:n for p,n in b.items() if p not in a}
result=dict(starting_commit=head,containing_commit=git('log','-1','--format=%H','--','docs/research/branch_replay_checkpoint.md').decode().strip(),comparison_base=base,branch=git('branch','--show-current').decode().strip(),worktree=str(root),status_short=git('status','--short').decode(),matrix={'VERIFIED':39,'INCOMPLETE':3},added_file_count=len(added),added_file_bytes=sum(added.values()),modified_file_net_bytes=sum(b[p]-a[p] for p in a.keys()&b.keys()),tree_net_bytes=sum(b.values())-sum(a.values()),git_count_objects=git('count-objects','-vH').decode(),push_performed=False)
assert result['status_short']==''
for name,rev in [('baseline_fresh_clone_pack',base),('checkpoint_fresh_clone_pack',head)]:
 proc=subprocess.Popen(['git','pack-objects','--stdout','--revs'],cwd=root,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 proc.stdin.write((rev+'\n').encode());proc.stdin.close();size=0
 while chunk:=proc.stdout.read(1024*1024):size+=len(chunk)
 errors=proc.stderr.read().decode();assert proc.wait()==0,errors
 result[name+'_bytes']=size
result['estimated_incremental_packed_bytes']=result['checkpoint_fresh_clone_pack_bytes']-result['baseline_fresh_clone_pack_bytes']
result['packing_method']='Read-only git pack-objects --stdout --revs, single branch ancestry, streams counted and discarded. No repack/prune/history rewrite. Excludes other refs/network framing/checkout.'
prior=root/'docs/research/semantic_recovery_evidence/branch_replay_final/REPOSITORY-SHA256SUMS'
count=0
for line in prior.read_text().splitlines():
 digest,path=line.split('  ',1); assert hashlib.sha256((root/path).read_bytes()).hexdigest()==digest,path;count+=1
result['previous_sequence_root_seal_verified']=count
out=root/'docs/research/semantic_recovery_evidence/rocket_production_start';out.mkdir()
(out/'checkpoint-and-growth.json').write_text(json.dumps(result,indent=2)+'\n')
shutil.copy('/private/tmp/rocket-production-start-preservation.json',out/'prior-checkpoint-verification.json');shutil.copy(__file__,out/'measure-growth.py')
(out/'SHA256SUMS').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name+'\n' for p in sorted(out.iterdir())))
print(json.dumps(result,indent=2))
