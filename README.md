# CS503 - Foundation Model Homework  

Welcome to the foundation model exercises! This homework consists of three parts:  

1) In part 1, we start by implementing the necessary building blocks to construct an autoregressive Transformer, like GPT.
2) In part 2, we will build a masked model in the style of MaskGIT.  
3) In part 3, we will build a simple 4M-like multimodal model.

Parts 1 and 2 are available now, and Part 3 will be available next week.

### Instructions

The instructions for each of these three parts are provided in the notebooks, which you can find under `./notebooks/`. They will introduce the problem statement to you, explain what parts in the codebase need to be completed, and you will use them to perform inference on the trained models. You will be asked to run the cells in those notebooks, provide answers to questions, etc. 

## **Installation**  
To begin the experiments, you first need to install the required packages and dependencies. To do this, please run the [setup_env.sh](setup_env.sh) script.

```bash
bash setup_env.sh
```

## Getting Started

We will implement the building blocks of autoregressive, masked, and multimodal models and train them on language and image modeling tasks.

You will primarily run the following files:
1. Jupyter notebooks: `nano4M/notebooks/` 
   - Usage: Introduction of the week's tasks and inference (post-training result generation and analysis).
2. Main training script: `run_training.py` 
   - Usage: Train your models after implementing the building blocks (refer to the notebooks for more details).

### Jupyter notebooks `nano4M/notebooks/`:
To use the Jupyter notebooks, activate the `nano4m` environment and run the notebooks. 
### Main training script `run_training.py`:

You can run the training job in one of two ways:

1. **Interactively using `srun`** – great for debugging.
2. **Using a SLURM batch script** – better for running longer jobs.

> **Before you begin**:  
> Make sure to have your Weights & Biases (W&B) account and obtain your W&B API key.  
> Follow the instructions in **Section 1.3 (Weights & Biases setup)** of the Jupyter Notebook.

---

#### Option 1: Run Interactively via `srun`

Start an interactive session on a compute node (eg, 2 GPUs case):

```bash
srun -t 120 -A cs-503 --qos=cs-503 --gres=gpu:2 --mem=16G --pty bash
```
Then, on the compute node:

```bash
conda activate nanofm
wandb login
OMP_NUM_THREADS=1 torchrun --nproc_per_node=2 run_training.py --config cfgs/nanoGPT/tinystories_d8w512.yaml
```
> **Note:**  
> To run the job on **one GPU**, make sure to:
> - Adjust the `--gres=gpu:1` option in the `srun` command, and  
> - Set `--nproc_per_node=1` in the `torchrun` command.

#### Option 2: Submit as a Batch Job via SLURM
You can use the provided submit_job.sh script to request GPUs and launch training.

Run:
```bash
sbatch submit_job.sh <config_file> <your_wandb_key> <num_gpus>
```
Replace the placeholders as follows:

- <config_file> — Path to your YAML config file

- <your_wandb_key> — Your W&B API key

- <num_gpus> — Set to 1 or 2 depending on your setup

Example Usage:
```bash
sbatch submit_job.sh cfgs/nanoGPT/tinystories_d8w512.yaml abcdef1234567890 2
```
