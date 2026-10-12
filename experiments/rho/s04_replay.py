"""04 | Rebuild the environment and replay actions to restore the same handoff state.

Input: adapter, initial state, action prefix and seed; output: action record, restoration error and image consistency.
Called by stage 03 for repeated labelling; keeps the full physics-state and determinism checks of the J8 protocol.
"""
import hashlib
import json


def stable_seed(*parts):
    return int(hashlib.sha256(json.dumps(parts).encode()).hexdigest()[:8], 16)


def physics_vector(env):
    """Joint positions/velocities, movable-object poses/velocities, door/button/switch joints, robot control targets. Ported from b0._physics_state_vector in an earlier project."""
    import numpy as np
    p, cid, robot = env.p, env.cid, env.robot
    values = []
    for joint_id in (*robot.arm_joint_ids, *robot.gripper_joint_ids):
        j = p.getJointState(robot.robot_uid, joint_id, physicsClientId=cid)
        values.extend((float(j[0]), float(j[1])))
    for item in env.scene.movable_objects:
        pose = p.getBasePositionAndOrientation(item.uid, physicsClientId=cid)
        vel = p.getBaseVelocity(item.uid, physicsClientId=cid)
        values.extend(float(v) for part in (*pose, *vel) for v in part)
    for name in ("doors", "buttons", "switches"):
        for item in getattr(env.scene, name):
            j = p.getJointState(item.uid, item.joint_index, physicsClientId=cid)
            values.extend((float(j[0]), float(j[1])))
    values.extend(float(v) for v in np.asarray(robot.target_pos).ravel())
    values.extend(float(v) for v in np.asarray(robot.target_orn).ravel())
    values.extend(float(v) for v in np.asarray(robot.gripper_action).ravel())
    return np.asarray(values, dtype=np.float64)


def image_hashes(env):
    import numpy as np
    obs = env.get_obs()
    return {k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k, v in obs["rgb_obs"].items()}


class Recorder:
    """Wrap env.step and record the actions actually applied (with the binarised gripper)."""
    def __init__(self):
        self.actions = []

    def attach(self, env):
        original = type(env).step

        def step(action, _env=env):
            self.actions.append(action.copy())
            return original(_env, action)
        env.step = step


def fresh_env(adapter, seed, recorder):
    """Close the old environment and create a new one, so contact history from the previous chain does not carry over."""
    from calvin_env.envs.play_table_env import get_env
    from rlinf.envs.calvin import ENV_CFG_DIR
    if adapter.env is not None:
        try:
            adapter.env.close()
        finally:
            adapter.env.cid = -1
    env = get_env(ENV_CFG_DIR, show_gui=False)
    env.seed(seed)
    adapter.env = env
    recorder.actions = []
    recorder.attach(env)
    return env


def restore(adapter, sequence, prefix, target_vector, target_images, seed, recorder):
    """Rebuild env -> reset to initial state -> replay prefix with cameras off -> check physics state and images."""
    import numpy as np
    env = fresh_env(adapter, seed, recorder)
    env.reset(robot_obs=np.asarray(sequence["initial"]["robot"]), scene_obs=np.asarray(sequence["initial"]["scene_obs"]))
    cameras, env.cameras = env.cameras, []
    try:
        for a in prefix:
            env.step(a.copy())
    finally:
        env.cameras = cameras
    recorder.actions = list(prefix)
    err = float(np.max(np.abs(physics_vector(env) - target_vector)))
    same_images = image_hashes(env) == target_images
    return err, same_images


def run_task(adapter, task, max_steps):
    adapter.start_task(task)
    for steps in range(1, max_steps + 1):
        if adapter.step_task():
            return True, steps
    return False, max_steps


def seed_all(seed):
    import random

    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed % (1 << 32))
    torch.manual_seed(seed)
