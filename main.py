"""
Entry point for running "PPO-like" policy gradient training using Ray and Hydra.
This is adapted from verl/verl/trainer/main_ppo.py
"""
# import os
# os.environ["RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES"] = "1"

# import os, torch
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"   # more precise traces on CUDA
# torch.autograd.set_detect_anomaly(True)    # pinpoints bad backward nodes

import atexit
import os
import signal
import subprocess
import sys

import hydra
import ray
from omegaconf import OmegaConf

from verl.trainer.main_ppo import TaskRunner
from verl.utils.device import is_cuda_available
from verl.trainer.constants_ppo import get_ppo_ray_runtime_env

# Register the custom resolver before the @hydra.main decorator
OmegaConf.register_new_resolver("mul", lambda x, y: None if x is None or y is None else int(x) * int(y))
OmegaConf.register_new_resolver("add", lambda x, y: None if x is None or y is None else int(x) + int(y))


@hydra.main(config_path="config", config_name="training_config", version_base=None)
def main(config):
    """Main entry point for PPO training with Hydra configuration management.

    Args:
        config: Hydra configuration dictionary containing training parameters.
    """
    import os
    os.environ["FLASH_ATTENTION_FORCE_DISABLE"] = "1"
    os.environ["ATTN_BACKEND"] = "eager"  # optional extra nudge for some models

    # Resolve interpolations in-place; cfg remains a DictConfig
    OmegaConf.resolve(config)

    
    if config.reward_model.get("sandbox_fusion") is not None and config.reward_model.sandbox_fusion.get("url") is not None:
        # Start Sandbox
        assert config.reward_model.sandbox_fusion.url == "http://127.0.0.1:8080/run_code"

        # leave this NOT configurable from yaml to avoid weird cmd execution issues
        cmd = [
            "uvicorn",
            "sandbox.server.server:app",
            "--host",
            "127.0.0.1",
            "--port",
            "8080",
        ]

        # don't care about stdout, maybe stderr is useful to keep though
        DEVNULL = subprocess.DEVNULL

        p = subprocess.Popen(cmd, start_new_session=True, stdout=DEVNULL)

        def kill_child():
            try:
                os.killpg(os.getpgid(p.pid), signal.SIGTERM)
            except Exception:
                pass

        atexit.register(kill_child)

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
def run_ppo(config, task_runner_class=None) -> None:
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
        # We also set `ROCR_VISIBLE_DEVICES` to an empty string to avoid GPU visibility issues.
        # This is required on the basement kluster, but not yet tested on other clusters.
        default_runtime_env = get_ppo_ray_runtime_env()
        ray_init_kwargs = config.ray_kwargs.get("ray_init", {})
        runtime_env_kwargs = ray_init_kwargs.get("runtime_env", {})

        if config.transfer_queue.enable:
            # Add runtime environment variables for transfer queue
            runtime_env_vars = runtime_env_kwargs.get("env_vars", {})
            runtime_env_vars["TRANSFER_QUEUE_ENABLE"] = "1"
            runtime_env_kwargs["env_vars"] = runtime_env_vars

        default_runtime_env["env_vars"]["ROCR_VISIBLE_DEVICES"] = ""
        debug = config.get("debug", None)
        if debug is not None:
            # set to True to enable debugging with vscode plugin
            if debug is True:
                default_runtime_env["env_vars"]["RAY_DEBUG"] = "1"
            # set to 'legecy' to enable pdb debugging
            else:
                default_runtime_env["env_vars"]["RAY_DEBUG"] = str(debug)
        else:
            default_runtime_env["env_vars"]["RAY_DEBUG"] = "0"



        runtime_env = OmegaConf.merge(default_runtime_env, runtime_env_kwargs)
        ray_init_kwargs = OmegaConf.create({**ray_init_kwargs, "runtime_env": runtime_env})
        print(f"ray init kwargs: {ray_init_kwargs}")
        ray.init(**OmegaConf.to_container(ray_init_kwargs))


    if task_runner_class is None:
        task_runner_class = ray.remote(num_cpus=1)(TaskRunner)  # please make sure main_task is not scheduled on head

    # Create a remote instance of the TaskRunner class, and
    # Execute the `run` method of the TaskRunner instance remotely and wait for it to complete
    if (
        is_cuda_available
        and config.global_profiler.tool == "nsys"
        and config.global_profiler.get("steps") is not None
        and len(config.global_profiler.get("steps", [])) > 0
    ):
        from verl.utils.import_utils import is_nvtx_available

        assert is_nvtx_available(), "nvtx is not available in CUDA platform. Please 'pip3 install nvtx'"
        nsight_options = OmegaConf.to_container(
            config.global_profiler.global_tool_config.nsys.controller_nsight_options
        )
        runner = task_runner_class.options(runtime_env={"nsight": nsight_options}).remote()
    else:
        runner = task_runner_class.remote()

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
