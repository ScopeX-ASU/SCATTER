"""
Date: 2024-03-26 14:23:47
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-03-26 14:23:47
FilePath: /SparseTeMPO/scripts/fmnist/cnn/train/train.py
"""

import os
import subprocess
from multiprocessing import Pool

import mlflow
from pyutils.general import ensure_dir, logger
from pyutils.config import configs

dataset = "cifar10"
model = "vgg8"
root = f"log/{dataset}/{model}/train/Sparsity"
script = "sparse_train.py"
config_file = f"configs/{dataset}/{model}/train/sparse_train.yml"
configs.load(config_file, recursive=True)
keep_same = False

def task_launcher(args):
    lr, density, w_bit, in_bit, death_mode, growth_mode, init_mode, conv_block, row_col, id, gpu_id  = args
    pres = [
        f"export CUDA_VISIBLE_DEVICES={gpu_id};",
        "python3", script, config_file]
    with open(
        os.path.join(
            root, f"{model}_{dataset}_lr-{lr:.3f}_wb-{w_bit}_ib-{in_bit}_dm-{death_mode}_gm-{growth_mode}_im-{init_mode}_cb-{conv_block}_{row_col}-without-opt_den-{density}_run-{id}.log"
        ),
        "w",
    ) as wfid:
        exp = [
            f"--optimizer.lr={lr}",
            f"--run.random_state={42}",
            f"--model.conv_cfg.w_bit={w_bit}",
            f"--model.linear_cfg.w_bit={w_bit}",
            f"--dst_scheduler.death_mode={death_mode}",
            f"--dst_scheduler.growth_mode={growth_mode}",
            f"--dst_scheduler.init_mode={init_mode}",
            f"--dst_scheduler.density={density}",
            f"--dst_scheduler.keep_same={keep_same}",
            f"--dst_scheduler.pruning_type={row_col}",
            f"--dst_scheduler.skip_first_layer={False}",
            f"--dst_scheduler.skip_last_layer={True}",
            f"--model.conv_cfg.miniblock=[{','.join([str(i) for i in conv_block])}]",
            f"--model.linear_cfg.miniblock=[{','.join([str(i) for i in conv_block])}]",
            f"--model.conv_cfg.in_bit={in_bit}",
            f"--model.linear_cfg.in_bit={in_bit}",
            f"--checkpoint.model_comment={row_col}-only-without-opt_lr-{lr:.4f}_wb-{w_bit}_ib-{in_bit}_dm-{death_mode}_gm-{growth_mode}_im-{init_mode}_cb-[{','.join([str(i) for i in conv_block])}]_density-{density}_run-{id}",
        ]
        cmd = " ".join(pres + exp)
        logger.info(f"running command:\n\t{cmd}")
        subprocess.call(cmd, stderr=wfid, stdout=wfid, shell=True)
     


if __name__ == "__main__":
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first

    tasks = [
        (0.002, 0.3, 8, 6, "magnitude", "gradient", "uniform", [1, 1, 16, 16], "structure_row_col", 1, 0),
        (0.002, 0.3, 8, 6, "magnitude_power", "gradient_power", "uniform_power", [1, 1, 16, 16], "structure_row_col", 1, 0),
    ]
    with Pool(9) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: {configs.run.experiment} Done.")
