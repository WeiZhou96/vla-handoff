"""02 | Execution interface for the CALVIN-D environment and the frozen pi0.5 policy.

Input: resource paths and execution settings; output: observations, action execution and CALVIN oracle success checks.
Call order: sequences -> open -> reset -> start_task -> step_task, then close.
Within a chain, reset only at the start; consecutive retries keep the oracle reference state. The step limit is set by the caller.

observe returns RGB images (static/wrist), the robot state (15,) and the scene state (24,).
Robot state: [0:3] TCP xyz, [3:6] Euler, [6] gripper width, [7:14] joint angles, [14] gripper command.
The policy uses only the first 7 dims; scene ground truth is for recovery/diagnosis and must not be a policy input.
Scene state: [0:2] sliding door/drawer, [2:4] button/switch, [4:6] lights, then 6 dims each for the red/blue/pink block poses.
These observations are not a full physics snapshot; exact restoration is handled by s04_replay.
Actions (H,7) are relative position, relative Euler and gripper command; CALVIN applies scaling, the gripper is binarised to +-1.
Image preprocessing, state normalisation and action denormalisation reuse OpenPI.

Interface sources (existing implementations are called):
RLinf, Apache-2.0, local commit 0505431899574619da86f551bad70b71e0ea2177:
    https://github.com/RLinf/RLinf/blob/0505431899574619da86f551bad70b71e0ea2177/toolkits/standalone_eval_scripts/openpi/calvin_eval.py
CALVIN, MIT, local commit fa03f01f19c65920e18cf37398a9ce859274af76:
    https://github.com/mees/calvin/blob/fa03f01f19c65920e18cf37398a9ce859274af76/calvin_models/calvin_agent/evaluation/utils.py
CALVIN env, MIT, local commit 1431a46bd36bde5903fb6345e68b5ccc30def666:
    https://github.com/mees/calvin_env/blob/1431a46bd36bde5903fb6345e68b5ccc30def666/calvin_env/robot/robot.py
"""

import hashlib
import json
import os
import sys
from collections import deque
from numbers import Integral
from pathlib import Path
from types import SimpleNamespace
from typing import cast


class CalvinAdapter:
    def __init__(self, paths: dict, settings: dict) -> None:
        """Store configuration references and add dependency paths; does not load the environment or policy yet."""
        self.paths, self.settings = paths, settings
        self.env = self.policy = self.oracle = None
        self.task = self.prompt = self.start_info = None
        self.annotations = {}
        self.actions = deque()
        for path in [paths["rlinf_root"], Path(paths["calvin_root"]) / "calvin_models",
                     Path(paths["calvin_root"]) / "calvin_env"]:
            if str(path) not in sys.path:
                sys.path.insert(0, str(path))

    def vocabulary(self) -> list[str]:
        """Read and cache CALVIN language instructions; return the sorted task names."""
        import yaml
        path = Path(self.paths["calvin_root"]) / "calvin_models/conf/annotations/new_playtable_validation.yaml"
        self.annotations = yaml.safe_load(path.read_text(encoding="utf-8"))
        return sorted(self.annotations)

    def protocol(self) -> dict:
        """Return the execution protocol; policy_sha256 is the checkpoint identifier from the configuration."""
        return dict(environment="calvin", scene="D", policy="pi05_calvin",
                    policy_sha256=self.settings["policy_sha256"], inference_mode="eager",
                    max_steps=self.settings["max_steps"], action_chunk=self.settings["action_chunk"],
                    flow_steps=self.settings["flow_steps"], grouping="initial_state_v1")

    def open(self, gpu: int, seed: int) -> None:
        """Select the GPU and load the policy, environment and oracle; the caller must close resources in a finally block."""
        if self.env is not None or self.policy is not None:
            raise RuntimeError("Adapter is already open or was not cleaned up; call close() first.")
        if isinstance(gpu, bool) or not isinstance(gpu, Integral) or gpu < 0:
            raise ValueError("gpu must be a non-negative integer.")
        if isinstance(seed, bool) or not isinstance(seed, Integral) or not 0 <= seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32).")
        gpu, seed = int(gpu), int(seed)
        for key in ("max_steps", "action_chunk", "flow_steps"):
            value = self.settings[key]
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{key} must be a positive integer.")
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)
        os.environ["JAX_PLATFORMS"] = "cpu"
        os.environ.setdefault("OMP_NUM_THREADS", "4")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        from ..s04_replay import seed_all
        seed_all(seed)
        import hydra
        from calvin_env.envs.play_table_env import get_env
        from calvin_env.utils.utils import get_egl_device_id
        from omegaconf import OmegaConf
        from rlinf.envs.calvin import ENV_CFG_DIR
        from toolkits.standalone_eval_scripts.openpi import setup_policy
        print("Loading frozen pi0.5...", flush=True)
        self.policy = setup_policy(SimpleNamespace(config_name="pi05_calvin",
            pretrained_path=str(self.paths["policy_dir"]), num_steps=self.settings["flow_steps"]))
        self.policy._model.eval()
        from torch.nn import Module
        cast(Module, self.policy._model).requires_grad_(False)
        method = self.policy._model.sample_actions
        method = getattr(method, "_torchdynamo_orig_callable", method)
        self.policy._model.sample_actions = self.policy._sample_actions = method
        # CUDA_VISIBLE_DEVICES has already mapped the selected GPU to cuda:0 within this process.
        os.environ["EGL_VISIBLE_DEVICES"] = str(get_egl_device_id(0))
        os.environ["EGL_VISIBLE_DEVICE"] = os.environ["EGL_VISIBLE_DEVICES"]
        self.env = get_env(ENV_CFG_DIR, show_gui=False)
        self.env.seed(seed)
        path = Path(self.paths["calvin_root"]) / "calvin_models/conf/callbacks/rollout/tasks/new_playtable_tasks.yaml"
        self.oracle = hydra.utils.instantiate(OmegaConf.load(path))
        self.vocabulary()

    def sequences(self, start: int, count: int) -> list[dict]:
        """Read task chains from the default pool of 1000 or from a given JSON candidate pool.

        parent_id depends only on the initial state; sequence_id also covers the ordered task chain.
        Custom files are cached by path; the content at a given path should not change."""
        for key, value in (("start", start), ("count", count)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
                raise ValueError(f"{key} must be a non-negative integer.")
        start, count = int(start), int(count)
        catalog = self.settings.get("sequence_catalog")
        if not catalog and start + count > 1000:
            raise ValueError("Sequence range exceeds the default candidate pool size of 1000.")
        from calvin_agent.evaluation.utils import get_env_state_for_initial_condition
        from rlinf.envs.calvin.utils import get_sequences
        # For larger collections a pre-fixed candidate pool can be read; by default the original 1000 sequences are used.
        if catalog:
            catalog_path = Path(catalog).expanduser().resolve()
            if getattr(self, "_sequence_catalog_path", None) != catalog_path:
                pool = json.loads(catalog_path.read_text(encoding="utf-8"))
                if not isinstance(pool, list):
                    raise ValueError("The top level of sequence_catalog must be a list of sequences.")
                self._sequence_pool = pool
                self._sequence_catalog_path = catalog_path
            pool = self._sequence_pool
        else:
            pool = get_sequences(1000)
        if start + count > len(pool):
            raise ValueError(f"Sequence range [{start}, {start + count}) exceeds the candidate pool size {len(pool)}.")
        result = []
        for index, (initial, tasks) in enumerate(pool[start:start + count], start):
            robot, scene = get_env_state_for_initial_condition(initial)
            state = dict(environment="calvin", scene="D", robot=robot.tolist(), scene_obs=scene.tolist())
            parent = hashlib.sha256(json.dumps(state, sort_keys=True).encode()).hexdigest()
            chain = hashlib.sha256(json.dumps([state, list(tasks)], sort_keys=True).encode()).hexdigest()
            result.append(dict(index=index, parent_id=parent, sequence_id=chain,
                               initial=state, tasks=list(tasks)))
        return result

    def reset(self, sequence: dict) -> None:
        """Restore the given initial state and clear the task reference state; do not call at natural handoffs between tasks."""
        if self.env is None:
            raise RuntimeError("open() must be called before reset().")
        import numpy as np
        initial = sequence["initial"]
        robot, scene = np.asarray(initial["robot"]), np.asarray(initial["scene_obs"])
        for key, value, shape in (("robot", robot, (15,)), ("scene_obs", scene, (24,))):
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"initial.{key} must be a finite numeric array of shape {shape}.")
        # Invalidate the old reference state before the environment changes, so a failed reset does not keep using the previous task's actions.
        self.actions.clear()
        self.task = self.prompt = self.start_info = None
        self.env.reset(robot_obs=robot, scene_obs=scene)

    def observe(self) -> dict:
        """Return copies of the HWC uint8 RGB images, the 15-dim robot state and the 24-dim scene state."""
        if self.env is None:
            raise RuntimeError("open() must be called before observe().")
        import numpy as np
        raw = self.env.get_obs()
        observation: dict = dict(images={"static": np.asarray(raw["rgb_obs"]["rgb_static"]).copy(),
                            "wrist": np.asarray(raw["rgb_obs"]["rgb_gripper"]).copy()},
                    state=np.asarray(raw["robot_obs"]).copy(),
                    # For simulator diagnostics: object and scene state; not added to the default policy or rho inputs.
                    scene_state=np.asarray(raw["scene_obs"]).copy())
        for key, shape in (("state", (15,)), ("scene_state", (24,))):
            value = observation[key]
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"{key} must be a finite numeric array of shape {shape}.")
        for key, value in observation["images"].items():
            if value.ndim != 3 or value.shape[-1] != 3 or not value.size or value.dtype != np.uint8:
                raise ValueError(f"images.{key} must be a non-empty HWC uint8 RGB image.")
        return observation

    def start_task(self, task: str) -> None:
        """Set the task, language instruction and oracle reference state and clear the action buffer; does not reset physics."""
        if self.env is None or not self.annotations:
            raise RuntimeError("open() must be called to load the environment and task instructions before start_task().")
        if not isinstance(task, str) or task not in self.annotations:
            raise ValueError(f"Unknown CALVIN task: {task!r}.")
        prompts = self.annotations[task]
        prompt = prompts if isinstance(prompts, str) else prompts[0] if prompts else None
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Task {task!r} has no non-empty language instruction.")
        start_info = self.env.get_info()
        self.task, self.prompt, self.start_info = task, prompt, start_info
        self.actions.clear()

    def step_task(self) -> bool:
        """Execute one control step and return whether the task succeeded; predict the next action chunk when the buffer is empty."""
        if self.env is None or self.policy is None or self.oracle is None:
            raise RuntimeError("open() must be called to load the environment, policy and oracle before step_task().")
        if self.task is None or self.start_info is None:
            raise RuntimeError("start_task() must be called before step_task(); after reset() the task must also be restarted.")
        import numpy as np
        import torch
        if not self.actions:
            length = self.settings["action_chunk"]
            if isinstance(length, bool) or not isinstance(length, Integral) or length < 1:
                raise ValueError("action_chunk must be a positive integer.")
            length = int(length)
            obs = self.observe()
            state = obs["state"]
            # The OpenPI interface has no batch axis: two HWC images, a (7,) state, and (3,)/(3,)/(1,) components.
            # state[6:7] is the measured gripper width; state[14] is the previous gripper command; they are not interchangeable.
            inputs = {"observation/image": obs["images"]["static"],
                      "observation/wrist_image": obs["images"]["wrist"],
                      "observation/state": state[:7], "observation/state_ee_pos": state[:3],
                      "observation/state_ee_rot": state[3:6], "observation/state_gripper": state[6:7],
                      "prompt": self.prompt}
            with torch.inference_mode():
                actions = np.asarray(self.policy.infer(inputs)["actions"], dtype=np.float32)
            if actions.ndim != 2 or actions.shape[1] != 7 or len(actions) < length:
                raise ValueError(f"Policy actions must have shape (H, 7) with H >= action_chunk={length}; got {actions.shape}.")
            if not np.isfinite(actions).all():
                raise ValueError("Policy actions contain NaN or Inf; stopping execution.")
            self.actions.extend(actions[:length].copy())
        action = self.actions.popleft().copy()
        action[-1] = 1.0 if action[-1] > 0 else -1.0
        _, _, _, info = self.env.step(action)
        return bool(self.oracle.get_task_info_for_set(self.start_info, info, {self.task}))

    def close(self) -> None:
        """Clear adapter state and close the environment; idempotent; exceptions during close still propagate."""
        env = self.env
        self.env = self.policy = self.oracle = None
        self.task = self.prompt = self.start_info = None
        self.actions.clear()
        if env is not None:
            try:
                env.close()
            finally:
                # CALVIN calls close again on destruction; prevent it from disconnecting a new environment that may later reuse the same client id.
                env.cid = -1
