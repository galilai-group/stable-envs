# Candidate rigid recording samples

Nominal G34/G62 production-export samples at source commit
`824baddcdad087810d35638bf8f273b884f74975`, using Python 3.11, bpy 4.4.0 and
NumPy 1.26.4. These are renderer INPUTS, separate from the independent Blender
geometry oracles in tests/fixtures/semantics/. No runtime loads an oracle.

Regenerate in a NEW directory using the isolated Blender interpreter:

```bash
python -m scripts.export.object_permanence export --source-root <pinned-source> --output <new-directory> --task G34
python -m scripts.export.object_permanence export --source-root <pinned-source> --output <new-directory> --task G62
```

The exporter refuses to overwrite clips. Compare poses, mesh hashes and source
geometry before replacing samples. Licenses are shared within this dataset;
source-derived CC-BY-NC-4.0 data, not MIT. See licenses/ and each clip manifest
for source, identity, array shapes, camera, semantic IDs and integrity hashes.
