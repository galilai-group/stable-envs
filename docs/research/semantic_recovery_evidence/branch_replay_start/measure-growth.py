import pathlib,subprocess,json,collections,hashlib,shutil
root=pathlib.Path('/private/tmp/stable-envs-semantic-recovery');start='13e51037cdb4afb9ce31aba3a4837ccf5c95adaf';head='31153d38c800f44e5c4c0ae1acdd0edca4c5f656'
def git(*args):return subprocess.check_output(['git',*args],cwd=root)
def tree(rev):
 out={}
 for row in git('ls-tree','-r','-l','-z',rev).split(b'\0'):
  if row:
   meta,path=row.split(b'\t');fields=meta.split();out[path.decode()]=int(fields[3])
 return out
a,b=tree(start),tree(head);added={p:n for p,n in b.items() if p not in a};modified={p:b[p]-a[p] for p in a.keys()&b.keys() if a[p]!=b[p]};dirs=collections.defaultdict(int)
for p,n in added.items():
 if p.startswith('docs/research/semantic_recovery_evidence/'):dirs[p.split('/')[3]]+=n
common=pathlib.Path(git('rev-parse','--git-common-dir').decode().strip())
result=dict(starting_commit=head,containing_commit=head,comparison_base=start,branch=git('branch','--show-current').decode().strip(),worktree=str(root),status_short=git('status','--short').decode(),matrix={'VERIFIED':39,'INCOMPLETE':3},added_file_count=len(added),added_file_bytes=sum(added.values()),modified_file_net_bytes=sum(modified.values()),tree_net_bytes=sum(b.values())-sum(a.values()),new_evidence_directory_bytes=dict(sorted(dirs.items())),twenty_largest_added_files=sorted([dict(path=p,bytes=n) for p,n in added.items()],key=lambda r:r['bytes'],reverse=True)[:20],git_count_objects=git('count-objects','-vH').decode(),git_common_directory=str(common),git_directory_logical_bytes=sum(p.stat().st_size for p in common.rglob('*') if p.is_file()),git_directory_du_kib=int(subprocess.check_output(['du','-sk',str(common)]).split()[0]),production_paths_changed=git('diff','--name-only',start,head,'--','stable_envs','pyproject.toml','uv.lock').decode().splitlines(),push_performed=False)
for name,rev in [('baseline_fresh_clone_pack',start),('checkpoint_fresh_clone_pack',head)]:
 proc=subprocess.Popen(['git','pack-objects','--stdout','--revs'],cwd=root,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
 proc.stdin.write((rev+'\n').encode());proc.stdin.close();size=0
 while chunk:=proc.stdout.read(1024*1024):size+=len(chunk)
 errors=proc.stderr.read().decode();assert proc.wait()==0,errors
 result[name+'_bytes']=size
result['estimated_incremental_packed_bytes']=result['checkpoint_fresh_clone_pack_bytes']-result['baseline_fresh_clone_pack_bytes']
result['packing_method']='Read-only git pack-objects --stdout --revs, single branch ancestry; streams counted and discarded. No repack, prune, deletion, or history rewrite. Excludes other remote refs, network framing, and checkout bytes.'
result['compaction_advisable']='Yes, before v0: consolidate repeated JSON and native NPZ comparisons into compact archives, retain manifests and representative witnesses, then separately review any history cleanup. No cleanup performed here.'
out=root/'docs/research/semantic_recovery_evidence/branch_replay_start';out.mkdir()
(out/'checkpoint-and-growth.json').write_text(json.dumps(result,indent=2)+'\n');shutil.copy('/private/tmp/branch-replay-start-preservation.json',out/'prior-checkpoint-verification.json');shutil.copy(__file__,out/'measure-growth.py')
print(json.dumps({k:v for k,v in result.items() if k not in ['twenty_largest_added_files']},indent=2))
