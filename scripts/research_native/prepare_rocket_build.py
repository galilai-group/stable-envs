"""Apply bounded research instrumentation to an isolated PyBullet 3.2.7 source tree."""

import argparse
import difflib
from pathlib import Path

p = argparse.ArgumentParser()
p.add_argument("--source", type=Path, required=True)
p.add_argument("--patch", type=Path, required=True)
a = p.parse_args()
changes = []
relative = "examples/SharedMemory/PhysicsServerCommandProcessor.cpp"
f = a.source / relative
before = f.read_text()
assert "seRocketCacheHook" not in before
hook = Path(__file__).with_name("rocket_cache_hook.hpp").read_text()
name = "bool PhysicsServerCommandProcessor::processRequestPhysicsSimulationParametersCommand("
idx = before.index(name)
after = before[:idx] + hook + "\n" + before[idx:]
idx = after.index("{", after.index(name)) + 1
after = (
    after[:idx]
    + "\n#ifndef USE_DISCRETE_DYNAMICS_WORLD\n    seRocketCacheHook(m_data->m_dynamicsWorld);\n#endif\n"
    + after[idx:]
)
f.write_text(after)
changes.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True), "a/" + relative, "b/" + relative))
relative = "Extras/Serialize/BulletWorldImporter/btMultiBodyWorldImporter.cpp"
f = a.source / relative
before = f.read_text()
needle = (
    "\tmb->forwardKinematics(scratchQ, scratchM);\n\tmb->updateCollisionObjectWorldTransforms(scratchQ, scratchM);"
)
assert before.count(needle) == 1
# Negative control only: skipping reconstruction does not itself restore the old caches.
after = "#include <cstdlib>\n" + before.replace(
    needle, '\tif (!std::getenv("SE_ROCKET_SKIP_RECONSTRUCTION")) {\n' + needle + "\n\t}"
)
after = after.replace(
    "#include <cstdlib>\n",
    "#include <cstdlib>\nclass btMultiBodyDynamicsWorld;\nextern void seRocketCacheHook(btMultiBodyDynamicsWorld*);\n",
)
marker = "\t\t\tif (bulletFile2->m_contactManifolds.size())"
assert after.count(marker) == 2
after = after.replace(
    marker,
    '\t\t\tif (std::getenv("SE_ROCKET_PRE_CONTACT_CACHE_IMPORT")) seRocketCacheHook(m_data->m_mbDynamicsWorld);\n'
    + '\t\t\tif (bulletFile2->m_contactManifolds.size() || std::getenv("SE_ROCKET_SYNC_EMPTY_CONTACTS"))',
)
after = after.replace(
    "&bulletFile2->m_contactManifolds[0]",
    "(bulletFile2->m_contactManifolds.size() ? &bulletFile2->m_contactManifolds[0] : 0)",
)
f.write_text(after)
changes.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True), "a/" + relative, "b/" + relative))
a.patch.write_text("".join(changes))
