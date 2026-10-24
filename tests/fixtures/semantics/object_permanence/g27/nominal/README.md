# Blender G27 nominal geometry oracle

All 120 source frames at 320x180, generated independently from pinned Blender 4.4.
No candidate exporter, recording or renderer is imported. Source, camera dimensions,
IDs, units, array shapes, hashes and versions are in manifest.json.
Depth XOR chunks preserve every float32 bit; shader RGB is not an oracle.

Regenerate in a new directory with `python -m scripts.testing.freeze_object_permanence
--source-root <pinned-checkout> --output <new-directory> --task G27` (one command).
Existing fixtures are never overwritten. CC-BY-NC-4.0 source-derived data;
see SOURCE_LICENSE.txt for full attribution and license terms.
