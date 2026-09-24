# Craftax full-symbolic cached-output correction

The final CPU array collector found a concrete boundary missing from the initial
candidate State classification. Full Craftax Symbolic's fused source reset returns
0.8999999761581421 for five normalized intrinsics; later standalone JIT recomputation
returns 0.9000000357627869. The five indices are 8239–8243, and maximum difference is
5.960464477539063e-08 for each tested reset seed 0, 17 and 42.

[Measured boundary](semantic_recovery_evidence/craftax_cpu/full-symbolic-boundary/result.json)
retains complete source/candidate/restored arrays and State. The actual source and
candidate **reset return values are exact**, as are all engine leaves and keys.
The mismatch is specifically recomputation from the returned State. SWM stores the
returned vector in `_last_obs`; the initial adapter discarded it. It cannot be
classified as a bitwise-recomputable output at this boundary.

A separate source-only proof preceded production correction:
[cached-output replay](semantic_recovery_evidence/craftax_cpu/full-symbolic-cache-proposal/result.json).
It encodes the source cache as a named numeric float32 `[B,8268]` leaf, proves detached
snapshot storage, restores into an independent source instance with a different
history, checks exact cached reset output, and continues six actions and an unseeded
reset exactly for seeds 0/17/42. No production cache support was used for this proof.

The bounded correction adds only `source_cache.last_symbolic` to **full Symbolic**
State and its compatibility signature. Reset and step retain the unchanged upstream
return value. Atomic set_state validates exact keys, shape, float32 dtype, finiteness
and signature and restores that owned cache with all other State. Like last reward,
the historical cache is independently restorable; it is not replaced by recomputation
or validated with an approximate-equality threshold. Outputs may exceed the wrapper's
incorrect [0,1] Box. The cache does not substitute for any of the 84 engine leaves or
keys. A caller constructing synthetic State must supply its intended source cache,
as the source wrapper itself owns that field independently of the engine dataclass.

This is a numeric encoding of measured wrapper-owned data, within the authorized
State adapter scope. It introduces no public Observation API, no filtering layer,
no shared contract change, and no copied game or observation algorithm. Learner
selection remains test/SWM-side. The other three registration schemas remain unchanged.
All equality acceptance checks remain exact; no tolerance is relaxed.

The original failing collector artifacts remain under
`semantic_recovery_evidence/craftax_cpu/CraftaxSymbolic/` and its sibling log.
The separate eager-terminal versus JIT-terminal difference and failed supplementary
assertion are also retained. Their correction uses the actual historical JIT boundary;
this cached-reset correction addresses a different, genuine source-return mismatch.
Final acceptance results are indexed by the [CPU checkpoint](craftax_recovery_checkpoint.md).
