"""Tests that set_state restores the gripper's finger joints, not only its command."""

from pathlib import Path

import numpy as np

import kinder
from kinder.envs.dynamic3d.envs import ObjectCentricTidyBot3DEnv

_TASK_CONFIG_PATH = (
    Path(kinder.__path__[0])
    / "envs"
    / "dynamic3d"
    / "tasks"
    / "Tossing3D"
    / "Tossing3D-o1.json"
)

# The gripper component of a TidyBot action, in [0, 1].
_OPEN = 0.0
_CLOSED = 1.0
# Control steps for the fingers to travel their whole range and settle.
_SETTLE_STEPS = 50


def _make_env() -> ObjectCentricTidyBot3DEnv:
    return ObjectCentricTidyBot3DEnv(
        num_objects=1,
        task_config_path=str(_TASK_CONFIG_PATH),
        scene_bg=False,
        allow_state_access=True,
    )


def _finger_qpos(env: ObjectCentricTidyBot3DEnv) -> np.ndarray:
    """The gripper's driver joint angles, read straight out of MuJoCo."""
    robot_env = env._robot_env  # pylint: disable=protected-access
    assert robot_env is not None
    assert robot_env.qpos is not None
    return np.array(robot_env.qpos["gripper"][:], dtype=float)


def _drive_gripper(
    env: ObjectCentricTidyBot3DEnv, gripper: float, steps: int = _SETTLE_STEPS
) -> None:
    """Step with zero base and arm deltas, so only the gripper moves."""
    action = np.zeros(11, dtype=np.float32)
    action[10] = gripper
    for _ in range(steps):
        env.step(action)


def test_tidybot3d_set_state_restores_the_finger_joints():
    """set_state must restore where the fingers are, not only what they were told.

    `pos_gripper` is the commanded ctrl value, so two states whose fingers sit far apart
    physically can carry the same command. Restoring the command alone leaves the
    fingers wherever the simulator happened to leave them.
    """
    env = _make_env()
    env.reset(seed=0)

    _drive_gripper(env, _CLOSED)
    closed_fingers = _finger_qpos(env)
    closed_state = env._get_current_state()  # pylint: disable=protected-access

    _drive_gripper(env, _OPEN)
    open_fingers = _finger_qpos(env)

    assert np.max(np.abs(closed_fingers - open_fingers)) > 0.1, (
        f"the fingers did not move between the closed and open commands, so this "
        f"test cannot discriminate: closed={closed_fingers}, open={open_fingers}"
    )

    env.set_state(closed_state)
    restored_fingers = _finger_qpos(env)

    assert np.allclose(restored_fingers, closed_fingers, atol=1e-6), (
        f"set_state left the fingers at {restored_fingers}, not at the "
        f"{closed_fingers} they held when the state was captured"
    )
    env.close()


def test_tidybot3d_restoring_a_grasp_after_a_release_restores_the_grasp():
    """A state captured mid-grasp must restore a grasp, even after a release.

    This is the retry-after-release case a trajectory sampler hits: it rolls out from a
    state in which the cube is held, the rollout ends with the gripper released, and it
    then restores that same start state to sample again. If the state carries only the
    gripper's command, the fingers stay wherever the release left them -- open, around
    a cube the state says is held.

    What this does *not* claim is that the replay is bit-faithful. Stepping on from the
    restored state still diverges from the original rollout by ~1e-3 after ten control
    steps, because MuJoCo's solver warm-start is not part of the object-centric state
    and a release leaves it far from where the grasp left it. The fingers are necessary
    for the retry to start from a grasp at all; they are not sufficient for a
    byte-identical replay.
    """
    env = _make_env()
    env.reset(seed=0)

    # Grasp the cube: open the fingers, put the cube between them, close.
    _drive_gripper(env, _OPEN)
    sim = env._robot_env.sim  # pylint: disable=protected-access
    pinch = np.array(sim.data.get_site_xpos("robot_pinch_site"), dtype=float)
    grasp_state = env._get_current_state()  # pylint: disable=protected-access
    cube = grasp_state.get_object_from_name("cube_0")
    grasp_state.set(cube, "x", pinch[0])
    grasp_state.set(cube, "y", pinch[1])
    grasp_state.set(cube, "z", pinch[2])
    env.set_state(grasp_state)
    _drive_gripper(env, _CLOSED)

    holding_state = env._get_current_state()  # pylint: disable=protected-access
    holding_fingers = _finger_qpos(env)
    held_z = holding_state.get(cube, "z")

    # The fingers must have stopped on the cube rather than closing on air, or the
    # cube is not in the hand and the rest of this test proves nothing.
    assert np.max(holding_fingers) < 0.5, (
        f"the fingers closed to {holding_fingers}, so they closed on nothing: the "
        f"cube is not between them"
    )

    # Release, as a failed rollout would end: the hand opens and the cube falls out.
    _drive_gripper(env, _OPEN)
    released_state = env._get_current_state()  # pylint: disable=protected-access
    assert released_state.get(cube, "z") < held_z - 0.05, (
        "the cube did not fall when the gripper opened, so a released simulator is "
        "not distinguishable from a holding one and this test proves nothing"
    )

    # Retry: restore the same start state, as a sampler would before resampling.
    env.set_state(holding_state)

    assert np.allclose(_finger_qpos(env), holding_fingers, atol=1e-6), (
        f"the restored fingers are at {_finger_qpos(env)}, not the {holding_fingers} "
        f"they held on the cube: the retry starts from an open hand"
    )

    # And the grasp holds when the retry steps on from there.
    _drive_gripper(env, _CLOSED, steps=10)
    retried_state = env._get_current_state()  # pylint: disable=protected-access
    assert retried_state.get(cube, "z") > held_z - 0.02, (
        f"the cube slid from {held_z:.4f} to {retried_state.get(cube, 'z'):.4f} over "
        f"ten steps of the retry, so the restored grasp did not hold it"
    )
    env.close()
