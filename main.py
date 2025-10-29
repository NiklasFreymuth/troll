"""
Entry point for running "PPO-like" policy gradient training using Ray and Hydra.
This is adapted from verl/verl/trainer/main_ppo.py
"""
# import os
# os.environ["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"

# import os, torch
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"   # more precise traces on CUDA
# torch.autograd.set_detect_anomaly(True)    # pinpoints bad backward nodes

import hydra
import ray
from omegaconf import DictConfig, OmegaConf

from verl.experimental.dataset.sampler import AbstractSampler
from verl.trainer.constants_ppo import PPO_RAY_RUNTIME_ENV
from verl.trainer.main_ppo import TaskRunner
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.reward import load_reward_manager
from verl.utils.device import is_cuda_available
from verl.utils.import_utils import load_extern_type

# Register the custom resolver before the @hydra.main decorator
OmegaConf.register_new_resolver("mul", lambda x, y: None if x is None or y is None else int(x) * int(y))
OmegaConf.register_new_resolver("add", lambda x, y: None if x is None or y is None else int(x) + int(y))


@hydra.main(config_path="config", config_name="training_config", version_base=None)
def main(config):
    """Main entry point for PPO training with Hydra configuration management.

    Args:
        config: Hydra configuration dictionary containing training parameters.
    """

    # Resolve interpolations in-place; cfg remains a DictConfig
    OmegaConf.resolve(config)

    error_to_raise = None
    try:
        run_ppo(config)
    except Exception as e:
        error_to_raise = e
    finally:
        if ray.is_initialized():
            ray.shutdown()

    # Re-raise the error, if one occurred
    if error_to_raise is not None:
        raise error_to_raise


# Define a function to run the PPO-like training process
def run_ppo(config) -> None:
    """Initialize Ray cluster and run distributed PPO training process.

    Args:
        config: Training configuration object containing all necessary parameters
                for distributed PPO training including Ray initialization settings,
                model paths, and training hyperparameters.
    """
    # Check if Ray is not initialized
    if not ray.is_initialized():
        # Initialize Ray with a local cluster configuration
        # Set environment variables in the runtime environment to control tokenizer parallelism,
        # NCCL debug level, VLLM logging level, and allow runtime LoRA updating
        # `num_cpus` specifies the number of CPU cores Ray can use, obtained from the configuration
        # @Niklas: We also set `ROCR_VISIBLE_DEVICES` to an empty string to avoid GPU visibility issues.
        # This is required on the basement kluster, but not yet tested on other clusters.
        debug = config.get("debug", None)
        ray_runtime_env = PPO_RAY_RUNTIME_ENV.copy()  # Make a copy to avoid modifying the original
        ray_runtime_env["env_vars"]["ROCR_VISIBLE_DEVICES"] = ""

        if debug is not None:
            # set to True to enable debugging with vscode plugin
            if debug is True:
                ray_runtime_env["env_vars"]["RAY_DEBUG"] = "1"
            # set to 'legecy' to enable pdb debugging
            else:
                ray_runtime_env["env_vars"]["RAY_DEBUG"] = str(debug)
            # ray_runtime_env["env_vars"]["RAY_DEBUG"] =
        else:
            ray_runtime_env["env_vars"]["RAY_DEBUG"] = "0"

        ray.init(
            #           local_mode=debug, #.get("debug", False),
            runtime_env=ray_runtime_env,
            num_cpus=config.ray_init.num_cpus,
        )

    # Create a remote instance of the TaskRunner class, and
    # Execute the `run` method of the TaskRunner instance remotely and wait for it to complete
    if is_cuda_available and config.trainer.get("profile_steps") is not None and len(config.trainer.get("profile_steps", [])) > 0:
        nsight_options = OmegaConf.to_container(config.trainer.controller_nsight_options)
        runner = TaskRunner.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = TaskRunner.remote()

    try:
        ray.get(runner.run.remote(config))
    except Exception as e:
        print(f"Run failed with the following exception: {e}")
        import traceback

        traceback.print_exc()
        raise

    # [Optional] get the path of the timeline trace file from the configuration, default to None
    # This file is used for performance analysis
    timeline_json_file = config.ray_init.get("timeline_json_file", None)
    if timeline_json_file:
        ray.timeline(filename=timeline_json_file)


if __name__ == "__main__":
    main()
