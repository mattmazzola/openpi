import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms


def make_echelon_example() -> dict:
    """Creates a random input example for the Echelon policy."""
    return {
        "state": np.ones((8,)),
        "images": {
            "cam_low": np.random.randint(256, size=(3, 512, 920), dtype=np.uint8),
        },
        "prompt": "do something",
    }


@dataclasses.dataclass(frozen=True)
class EchelonInputs(transforms.DataTransformFn):
    """Inputs for the Echelon policy.

    Expected inputs:
    - images: dict[name, img] where img is [channel, height, width]. name must be in EXPECTED_CAMERAS.
    - state: [8] - [x,y,y,qx,qy,qz,qw,gripper]
    - action: [7] - [dx,dy,dz,rx,ry,rz,gripper]
    """

    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int

    # The expected cameras names. All input cameras must be in this set. Missing cameras will be
    # replaced with black images and the corresponding `image_mask` will be set to False.
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = "cam_low"

    def __call__(self, data: dict) -> dict:
        # Get the state. We are padding from 14 to the model action dim.
        state = transforms.pad_to_dim(data["state"], self.action_dim)

        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        # Assume that base image always exists.
        base_image = in_images["cam_low"]

        images = {
            "base_0_rgb": base_image,
        }
        image_masks = {
            "base_0_rgb": np.True_,
        }

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": state,
        }

        # Actions are only available during training.
        if "action" in data:
            action = np.asarray(data["action"])
            inputs["action"] = transforms.pad_to_dim(action, self.action_dim)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class EchelonOutputs(transforms.DataTransformFn):
    """Outputs for the Echelon policy."""

    def __call__(self, data: dict) -> dict:
        # Only return the first 7 dims.
        action = np.asarray(data["action"][:7])
        return {"action": action}
