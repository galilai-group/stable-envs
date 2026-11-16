"""The upstream ``EnvParams`` fields that Craftax variation leaves set, per game."""

from __future__ import annotations

#: ``(variation path, EnvParams field)`` for Craftax-Classic.
CLASSIC_PARAMS = (
    ("world.day_length", "day_length"),
    ("world.always_diamond", "always_diamond"),
    ("zombie.health", "zombie_health"),
    ("cow.health", "cow_health"),
    ("skeleton.health", "skeleton_health"),
    ("mobs.despawn_distance", "mob_despawn_distance"),
    ("spawn.cow", "spawn_cow_chance"),
    ("spawn.zombie_base", "spawn_zombie_base_chance"),
    ("spawn.zombie_night", "spawn_zombie_night_chance"),
    ("spawn.skeleton", "spawn_skeleton_chance"),
)

#: ``(variation path, EnvParams field)`` for Craftax.
CRAFTAX_PARAMS = (
    ("world.day_length", "day_length"),
    ("world.always_diamond", "always_diamond"),
    ("mobs.despawn_distance", "mob_despawn_distance"),
    ("player.max_attribute", "max_attribute"),
)

#: Allowed values of the integer parameters, both ends included.
INT_RANGES = {
    "world.day_length": (50, 1000),
    "zombie.health": (1, 9),
    "cow.health": (1, 9),
    "skeleton.health": (1, 9),
    "mobs.despawn_distance": (6, 30),
    "player.max_attribute": (1, 10),
}
