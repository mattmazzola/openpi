from pathlib import Path
import re
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

print("Actions shape:", result["actions"].shape)
print("Actions:")
print(result["actions"][:3])
print("...")
print(result["actions"][-3:])
