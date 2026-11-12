import numpy as np

ENV_FIELDS = "start_pos start_orn step_count termination truncation state action reward throttle_state ignition_state previous_throttle_state previous_ignition_state previous_render_ignition_state ignition_color landing_pad_contact ang_vel ang_pos lin_vel lin_pos ground_lin_vel previous_ang_vel previous_lin_vel previous_lin_pos previous_ground_lin_vel current_goal".split()
AVIARY_FIELDS = "physics_steps aviary_steps elapsed_time contact_array".split()
DRONE_FIELDS = "mode setpoint cmd state aux_state".split()
#: ``ENV_FIELDS`` that only a reset sets.
RESET_FIELDS = frozenset(["start_pos", "start_orn", "ignition_color", "current_goal"])


def objects(env):
    d = env.env.drones[0]
    result = {
        "wrapper": (env, ENV_FIELDS),
        "aviary": (env.env, AVIARY_FIELDS),
        "drone": (d, DRONE_FIELDS),
        "bodies": (d.bodies, ["local_body_velocities"]),
        "boosters": (d.boosters, ["ratio_fuel_remaining", "throttle", "ignition_state"]),
        "gimbal": (d.booster_gimbal, ["gimbal_state", "rotation1", "rotation2"]),
    }
    for i, surface in enumerate(d.lifting_surfaces.surfaces):
        result[f"fin_{i}"] = (surface, ["local_surface_velocity", "actuation"])
    return result


def _fields(env, result, step_only):
    for prefix, (obj, names) in objects(env).items():
        for name in names:
            if not (step_only and prefix == "wrapper" and name in RESET_FIELDS):
                result[prefix + "/" + name] = np.asarray(getattr(obj, name)).copy()
    for name in ["out_of_bounds", "fatal_collision", "env_complete"]:
        result["info/" + name] = np.asarray(env.info[name]).copy()
    rng = env.np_random.bit_generator.state
    assert rng["bit_generator"] == "PCG64"
    for name in ["state", "inc"]:
        value = rng["state"][name]
        result["rng/" + name] = np.array([(value >> (32 * i)) & 0xFFFFFFFF for i in range(4)], np.int64)
    result["rng/cached"] = np.array([rng["has_uint32"], rng["uinteger"]], np.int64)
    assert all(obj.np_random is env.np_random for obj in [env.env, env.env.drones[0], env.env.drones[0].boosters])


def _base(env, result, i, body):
    pos, orn = env.env.getBasePositionAndOrientation(body)
    vel, ang = env.env.getBaseVelocity(body)
    result[f"physics/body_{i}/position"] = np.array(pos)
    result[f"physics/body_{i}/quaternion"] = np.array(orn)
    result[f"physics/body_{i}/linear_velocity"] = np.array(vel)
    result[f"physics/body_{i}/angular_velocity"] = np.array(ang)


def _mass(env, result, i, body, link):
    info = env.env.getDynamicsInfo(body, link)
    result[f"physics/body_{i}/link_{link}/mass_inertia"] = np.array([info[0], *info[2]])


def _cached_links(env, result):
    d = env.env.drones[0]
    links = env.env.getLinkStates(d.Id, list(range(env.env.getNumJoints(d.Id))), computeLinkVelocity=1)
    result["diagnostic/cached_links"] = np.array([np.concatenate([np.asarray(v) for v in link]) for link in links])


def capture(env):
    """Copy named engine, task, contact and RNG fields for replay verification."""
    result = {}
    _fields(env, result, False)
    for i in range(env.env.getNumBodies()):
        body = env.env.getBodyUniqueId(i)
        _base(env, result, i, body)
        for link in range(-1, env.env.getNumJoints(body)):
            _mass(env, result, i, body, link)
            if link >= 0:
                j = env.env.getJointState(body, link)
                result[f"physics/body_{i}/link_{link}/joint_position_velocity"] = np.array(j[:2])
    _cached_links(env, result)
    return result


def step_bodies(env):
    """``(index, body, links)`` of the one body that moves, with the links whose mass a step changes.

    The ground and the pad have fixed bases, and every rocket joint is fixed. Only the fuel tanks
    change mass during a step.
    """
    d = env.env.drones[0]
    index = next(i for i in range(env.env.getNumBodies()) if env.env.getBodyUniqueId(i) == d.Id)
    return index, d.Id, [int(link) for link in d.boosters.fueltank_ids if link is not None]


def capture_step(env, bodies):
    """The ``capture`` entries that a step can change; ``bodies`` is ``step_bodies(env)``."""
    result = {}
    _fields(env, result, True)
    index, body, links = bodies
    _base(env, result, index, body)
    for link in links:
        _mass(env, result, index, body, link)
    _cached_links(env, result)
    return result
