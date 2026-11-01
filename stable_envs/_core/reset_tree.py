"""Seed and sample a Gymnasium reset tree only along the paths that a reset reads.

``Dict.seed(seed)`` seeds the Dict, draws one sub-seed for each child in key order and seeds each
child with its sub-seed. A lane of a reset reads a few leaves, so seeding the nodes on their paths
gives the same generators as seeding the whole tree, with fewer generators to construct.
"""

from __future__ import annotations

import numpy as np
from gymnasium import spaces

_SUBSEED_HIGH = int(np.iinfo(np.int32).max)


class PathSeeder:
    """Seed the nodes on ``paths`` of ``tree`` as ``tree.seed(seed)`` seeds them.

    Nodes that are not on a path keep their generators. A tree with a node other than a
    Gymnasium ``Dict`` on a path, or a negative seed, falls back to ``tree.seed``.
    """

    def __init__(self, tree, paths):
        self.tree = tree
        self.chains = []
        self.exact = True
        for path in paths:
            node, chain = tree, []
            for part in path.split("."):
                if type(node) is not spaces.Dict or part not in node.spaces:
                    self.exact = False
                    break
                keys = list(node.spaces)
                chain.append((node, keys.index(part), len(keys)))
                node = node.spaces[part]
            if type(node) not in (spaces.Box, spaces.Discrete):
                self.exact = False
            self.chains.append((chain, node))

    def seed(self, seed: int | None) -> None:
        if not self.exact or (seed is not None and seed < 0):
            self.tree.seed(seed)
            return
        if seed is None:
            # Dict.seed(None) gives every leaf fresh entropy and leaves the Dict generators unused.
            for _, leaf in self.chains:
                leaf._np_random = np.random.Generator(np.random.PCG64())
            return
        draws: dict[int, np.ndarray] = {}
        for chain, leaf in self.chains:
            value = seed
            for node, index, count in chain:
                subseeds = draws.get(id(node))
                if subseeds is None:
                    rng = np.random.Generator(np.random.PCG64(value))
                    node._np_random = rng
                    subseeds = draws[id(node)] = rng.integers(_SUBSEED_HIGH, size=count)
                value = int(subseeds[index])
            if id(leaf) not in draws:
                leaf._np_random = np.random.Generator(np.random.PCG64(value))
                draws[id(leaf)] = np.empty(0)


def sample(space):
    """Return ``space.sample()``; a bounded float Box draws its uniform values directly.

    ``Box.sample`` draws the bounded coordinates with one ``uniform`` call in C order and casts
    the float64 result to the Box dtype. The direct draw makes the same call.
    """
    if (
        type(space) is spaces.Box
        and space.dtype.kind == "f"
        and len(space.shape) > 0
        and space.bounded_below.all()
        and space.bounded_above.all()
    ):
        return space.np_random.uniform(space.low, space.high, size=space.shape).astype(space.dtype)
    return space.sample()
