import dataclasses
from typing import ClassVar

import einops
import numpy as np

from openpi import transforms


def make_echelon_example() -> dict:
    """Creates a random input example for the Echelon policy."""
    return {
        "state": np.ones((7,)),
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
    - state: [7] - [ja0,ja1,ja2,ja3,ja4,ja5,gripper_action] (6 joint angles in radians + gripper action 0.0 or 1.0)
    - action: [7] - [ja0,ja1,ja2,ja3,ja4,ja5,gripper_action] (6 joint angles in radians + gripper action 0.0 or 1.0)
    """

    # state_dim: int
    # The action dimension of the model. Will be used to pad state and actions.
    action_dim: int

    # The expected cameras names. All input cameras must be in this set. Missing cameras will be
    # replaced with black images and the corresponding `image_mask` will be set to False.
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = ("cam_low",)

    def __call__(self, data: dict) -> dict:
        state = transforms.pad_to_dim(data["state"], self.action_dim)

        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images to contain {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        def convert_image(img):
            img = np.asarray(img)
            # Convert to uint8 if using float images.
            if np.issubdtype(img.dtype, np.floating):
                img = (255 * img).astype(np.uint8)
            # Convert from [channel, height, width] to [height, width, channel].
            return einops.rearrange(img, "c h w -> h w c")

        images_dict = {cam_name: convert_image(img) for cam_name, img in data["images"].items()}

        # Assume that base image always exists.
        base_image = images_dict["cam_low"]

        images = {
            "base_0_rgb": base_image,
        }
        image_masks = {
            "base_0_rgb": np.True_,
        }

        # Add the extra images.
        extra_image_names = {
            "left_wrist_0_rgb": "cam_left_wrist",
            "right_wrist_0_rgb": "cam_right_wrist",
        }
        for dest, source in extra_image_names.items():
            if source in in_images:
                images[dest] = in_images[source]
                image_masks[dest] = np.True_
            else:
                images[dest] = np.zeros_like(base_image)
                image_masks[dest] = np.False_

        inputs = {
            "image": images,
            "image_mask": image_masks,
            "state": state,
        }

        # Actions are only available during training.
        if "actions" in data:
            actions = np.asarray(data["actions"])
            inputs["actions"] = transforms.pad_to_dim(actions, self.action_dim)

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class EchelonOutputs(transforms.DataTransformFn):
    """Outputs for the Echelon policy."""

    action_output_dim: int

    def __call__(self, data: dict) -> dict:
        # Only return the first 7 dims.
        actions = np.asarray(data["actions"][:, : self.action_output_dim])
        return {"actions": actions}
