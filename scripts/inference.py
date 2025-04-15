from openpi.training import config
from openpi.policies import policy_config
from openpi.shared import download

config = config.get_config("pi0_fast_droid")
checkpoint_dir = download.maybe_download("s3://openpi-assets/checkpoints/pi0_fast_droid")

# Create a trained policy.
policy = policy_config.create_trained_policy(config, checkpoint_dir)

# Run inference on a dummy example.
example = {
    "observation/exterior_image_1_left": ...,
    "observation/wrist_image_left": ...,
    "prompt": "pick up the fork"
}
action_chunk = policy.infer(example)["actions"]