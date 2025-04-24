import json
from pathlib import Path
import re

import numpy as np
from openpi.training import config as _config
from openpi.policies import policy_config
from openpi.shared import download
from openpi.policies import echelon_policy


config = _config.get_config("pi0_echelon")
checkpoint_dir = download.maybe_download("/home/mattm/openpi/checkpoints/pi0_echelon_sim/echelon_train_01/19999")

# Create a trained policy.
policy = policy_config.create_trained_policy(config, checkpoint_dir)

# Run inference on a dummy example.
example = echelon_policy.make_echelon_example()
result = policy.infer(example)

# Delete the policy to free up memory.
del policy


def get_action_components(action: list[float]) -> dict[str, float]:
    """Extracts the components of the action."""
    action_components = {
        "dx": action[0],
        "dy": action[1],
        "dz": action[2],
        "rx": action[3],
        "ry": action[4],
        "rz": action[5],
        "gripper": action[6],
    }

    eef_pos = action_components["dx"], action_components["dy"], action_components["dz"]
    eef_rot = action_components["rx"], action_components["ry"], action_components["rz"]
    gripper = action_components["gripper"]

    return {
        "eef_pos": eef_pos,
        "eef_rot": eef_rot,
        "gripper": gripper,
    }


print("Actions shape:", result["actions"].shape)
print("Actions:")
print(json.dumps([get_action_components(a) for a in result["actions"][:2]], indent=2))
print("...")
print(json.dumps([get_action_components(a) for a in result["actions"][-1:]], indent=2))
