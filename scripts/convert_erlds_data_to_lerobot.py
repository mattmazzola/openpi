"""
Script to convert Echelon RLDS dataset (ERLDS) into LeRobot dataset format.

Usage:
uv run scripts/convert_erlds_data_to_lerobot.py --data_dir /path/to/your/data

"""

import shutil

from lerobot.common.datasets.lerobot_dataset import LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tensorflow_datasets as tfds
import tyro
import numpy as np
import tensorflow as tf
from scipy.spatial.transform import Rotation
from typing import Any


def compute_rotation_delta(current, target):
    assert current.shape[-1] == 4, f"Expected quaternions with 4 values, got shape {current.shape}"
    assert target.shape[-1] == 4, f"Expected quaternions with 4 values, got shape {target.shape}"

    # Create Rotation objects for current and target quaternions
    current_r = Rotation.from_quat(current.numpy())
    target_r = Rotation.from_quat(target.numpy())

    # Compute the delta rotation: q_delta = q_target * q_current^(-1)
    delta_r = target_r * current_r.inv()

    # Convert to euler angles (roll, pitch, yaw in radians)
    result = delta_r.as_euler("xyz")

    return result


def get_eef_state_action(step: dict[str, Any]) -> tuple[tf.Tensor, tf.Tensor]:
    observation = step["observation"]
    action = step["action"]

    # Robot state (8) - EEF XYZ (3) + Quaternion (4) + Gripper Open/Close (1)
    state_eef_position = observation["tip_cartesian_euler_position_r"]
    state_eef_activation = observation["end_effector_action_r"]
    state_eef_activation = tf.clip_by_value(state_eef_activation, 0.0, 1.0)

    robot_state = tf.concat(
        [
            state_eef_position,
            state_eef_activation,
        ],
        axis=-1,
    )

    assert robot_state is not None

    state_eef_xyz = state_eef_position[:3]
    state_eef_orientation = state_eef_position[3:7]

    # Robot action (7) - EEF XYZ (3) + EEF RPY (3) + Gripper Open/Close (1)
    action_eef_position = action["tip_cartesian_euler_position_r"]
    action_eef_xyz = action_eef_position[:3]
    action_eef_orientation = action_eef_position[3:7]

    # Calculate the delta (target state - current state)
    action_xyz_delta = action_eef_xyz - state_eef_xyz

    # Wrap the SciPy operation with tf.py_function
    action_rpy_delta = tf.py_function(
        func=compute_rotation_delta,
        inp=[state_eef_orientation, action_eef_orientation],
        Tout=tf.float64,
    )
    action_eef_activation = action["end_effector_action_r"]
    action_eef_activation = tf.clip_by_value(action_eef_activation, 0.0, 1.0)

    robot_action = tf.concat(
        [
            action_xyz_delta,
            action_rpy_delta,
            action_eef_activation,
        ],
        axis=-1,
    )

    return robot_state, robot_action


def get_joint_angles_state_action(step: dict[str, Any]) -> tuple[tf.Tensor, tf.Tensor]:
    observation = step["observation"]
    action = step["action"]
    ur3e_joints_max_index = 6

    # Robot state (7) - Joint angles (6) + Gripper Action (1)
    # Joint angles are in radians, gripper action is a float between 0 and 1
    state_joint_angles = observation["joint_position_r"][:ur3e_joints_max_index]
    state_eef_action = observation["end_effector_action_r"]
    state_eef_action = tf.clip_by_value(state_eef_action, 0.0, 1.0)

    robot_state = tf.concat(
        [
            state_joint_angles,
            state_eef_action,
        ],
        axis=-1,
    )

    assert robot_state is not None

    # Robot action (7) - Joint angles (6) + Gripper Action (1)
    action_joint_angles = action["joint_position_r"][:ur3e_joints_max_index]
    action_eef_action = action["end_effector_action_r"]
    action_eef_action = tf.clip_by_value(action_eef_action, 0.0, 1.0)

    robot_action = tf.concat(
        [
            action_joint_angles,
            action_eef_action,
        ],
        axis=-1,
    )

    return robot_state, robot_action


def main(
    data_dir: str,
    dataset_name: str,
    *,
    push_to_hub: bool = False,
    # Dataset Naming Conventions:
    # echelon-[data_class]-[state_type]-[cameras]
    # data_class: original, orig_and_fuzzed
    # state: eef, ja
    # cameras: main, main_and_right_wrist
    #
    # ERDS dataset (Does not need state because includes all data)
    # echelon_original-main
    # echelon_fuzzed-main_right_wrist
    #
    # Pi/Lerobot Dataset, Model
    # echelon-original-eef-main
    # echelon-original-eef-main_and_right_wrist
    # echelon-original-ja-main
    # echelon-original-ja-main_and_right_wrist
    # TODO: Compute repo name from args given
    repo_name: str = "mattmazzola/echelon-original-eef-main_and_right_wrist",
):
    # Clean up any existing dataset in the output directory
    output_path = LEROBOT_HOME / repo_name
    if output_path.exists():
        shutil.rmtree(output_path)

    # Create LeRobot dataset, define features to store
    # OpenPi assumes that proprio is stored in `state` and actions in `action`
    # LeRobot assumes that dtype of image data is `image`
    lerobot_dataset = LeRobotDataset.create(
        repo_id=repo_name,
        robot_type="ur3e",
        fps=5,
        features={
            "images.main": {
                "dtype": "image",
                "shape": (512, 910, 3),
                "names": ["height", "width", "channel"],
            },
            "images.arm_right": {
                "dtype": "image",
                "shape": (512, 910, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "action": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["action"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    erlds_dataset = tfds.load(dataset_name, data_dir=data_dir, split="train")
    for episode in erlds_dataset:
        for step in episode["steps"].as_numpy_iterator():
            image_main = step["observation"]["image_main"]
            image_arm_right = step["observation"]["image_r"]
            state, action = get_eef_state_action(step)
            frame = {
                "images.main": image_main,
                "images.arm_right": image_arm_right,
                "state": state,
                "action": action,
            }
            task = step["natural_language_instruction"].decode()

            lerobot_dataset.add_frame(frame)
        lerobot_dataset.save_episode(task=task)

    # Consolidate the dataset, skip computing stats since we will do that later
    lerobot_dataset.consolidate(run_compute_stats=False)

    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        lerobot_dataset.push_to_hub(
            tags=["ur3e", "rlds"],
            private=False,
            push_videos=False,
            license="apache-2.0",
        )


if __name__ == "__main__":
    tyro.cli(main)
