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


def get_state_action(step: dict[str, Any]) -> tuple[tf.Tensor, tf.Tensor]:
    observation = step["observation"]
    action = step["action"]
    ur3e_joints_max_index = 6

    # Robot state - Joint angles (6) + Gripper Action (1)
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
    repo_name: str = "mattmazzola/echelon-joint-angles",
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
                "shape": (7,),
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
            state, action = get_state_action(step)
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
