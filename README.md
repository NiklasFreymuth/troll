# TROLL

[TROLL: Trust Regions improve Reinforcement Learning for Large Language Models](https://arxiv.org/abs/2510.03817)

# Getting Started

## Setting up the project

### Mamba
This project uses [mamba](https://github.com/conda-forge/miniforge)/[conda](https://docs.conda.io/en/latest/) and pip for handling packages and dependencies.
To install mamba on Unix, use either of:

```
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh
```

For Windows please see the documentation in the link above or use (not recommended).
```
conda install -c conda-forge mamba
```

### Installs

This project uses [verl](https://verl.readthedocs.io/en/latest/) for training and evaluation of large language models.
We set up the installs to work within conda and without sudo access for cluster compatibility. 
We only tested this on Linux, but you probably wouldn't want to train an LLM on Windows anyway.

You should be able to install all requirements using the commands below. 
This largely follows the [verl install instructions](https://verl.readthedocs.io/en/latest/start/install.html), 
except that we use conda/mamba instead of docker for simplicity.

```
# Install conda env
# This installs cuda, cudnn and apex.
mamba env create -f ./env/env.yaml
mamba activate troll 

# At this point, check if cuda is installed correctly.
# You can do this by running `nvcc --version` or `nvidia-smi`. You can also do
conda list cuda
# and should see some cuda packages installed, hopefully at, e.g., 12.4

# Install verl dependencies
# We install the version without megatron, as we do not have 200 H100s
USE_MEGATRON=0 bash verl/scripts/install_vllm_sglang_mcore.sh
pip install --no-deps -e ./verl

# Install flash attn by hand (-v because this takes forever)
pip install -v flash-attn==2.7.4.post1

# Steal (copy) a crypt.h from /usr/, as this is not covered by the conda/mamba installs.
# This is required for verl to work, as it uses the crypt module.
# Executing this script will also extend your CPATH when activating the troll conda environment.
bash env/steal_crypt_h.sh

# Activate env, log into wandb, and install pre-commit
wandb login
pre-commit install

# Add discrete trpl dependencies
cd dependencies
git clone git@github.com:pbecker93/discrete_trpl.git
pip install -e ./discrete_trpl
```


### Minimum example

We follow the [verl quickstart PPO example](https://verl.readthedocs.io/en/latest/start/quickstart.html) as our starting point. 
This requires a GPU with at least 24GB of memory, i.e., at least a 3090.
The following commands will download a model and run the example from our custom config.
Make sure you are in the `troll` directory before running these commands.

*Side comment*
You **could** download an example dataset via, e.g.,
```
python verl/examples/data_preprocess/gsm8k.py --local_dir ./data/gsm8k
```
However, we currently simply have the datasets in our git repo. They are sufficiently small and this avoids any download issues.
So you can skip this step, as the dataset is already in `./data/gsm8k`.


Download a model. We download the models locally to avoid huggingface rate limits. We use Qwen3-0.6B as it is small and powerful enough for debugging

```
python download_model.py --model_type "Qwen/Qwen3-0.6B"
```

For other model types, simply change the `--model_type` argument to any huggingface model.

Actually run the example using hydra.
We build our hydra config locally in troll/config. 
For this, we adapt the verl ppo base config, so we need to adapt the config path a bit
We can run this with locally 4 or 1 GPU(s) as follows.

```
N_GPUS=4
python main.py +_runs=debug n_gpus=$N_GPUS
```


# Project Structure

### Configs

The folder `config` contains the hydra configuration files for the project. 
We still inherit from the verl base config, but tried to expose all potentially relevant parameters. These parameters
are listed in
* `config/performance`: Anything performance-related that does *not* change the actual algorithm behavior. 
  E.g., Micro batch sizes, which fsdp to use, offloading behavior etc.
* `config/method`: Specifies the actual algorithm to run (independent of a model or dataset). The default here is 
  `abstract_pg`, which sets sane defaults for any policy gradient method. PPO, GRPO, Dr.GRPO and GSPO all inherit from this.
*  `config/task`: Specifies the actual task to run. This is mainly the dataset, but also details on how many epochs to run for,
  how often to evaluate, etc.
* `config/platform`: Specifies the SLURM sbatch config. We have presets for Horeka, BWC and the basement Kluster.
* `config/_runs`: Contains individual *experiments*, each of which is a separate `.yaml`. Each experiment should
  * Start with an  `# @package _global_` comment for hydra routing
    * Import a `method` and a `task`
    * Define a `model_path`. This directs to one of the models downloaded in the `playground/download_model.py` step. E.g.,
      `model_path: "./models/Qwen_Qwen2.5-0.5B-Instruct"` (note the `_` instead of `/`).
    * Meta information in the form of an `_idx` , a `_version` and an `exp_name`. These are purely for keeping an overview in wandb.
    * A number `n_gpus` to run this with

```bash
python main.py +_runs/path/to=exp_name
```

This will sequentially run any number of runs within the experiment, as defined in the corresponding yaml file.

A recent example is

``` 
python main.py +_runs/gsm8k/phase1_debug=164_7b_performance
```
Which grids over ppo and grpo with different batch sizes for a 7b model (and happens to have `_idx164`, hence the name)
If you want to submit an experiment to SLURM, you can add a `+platform=...` to the above. E.g.,

```bash
python main.py +_runs/gsm8k/phase1_debug=164_7b_performance
```
