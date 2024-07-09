"""
Date: 2024-04-27 23:07:24
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2024-04-27 23:07:32
FilePath: /SparseTeMPO/scripts/fmnist/cnn/train/test_crosstalk.py
"""
import os
import subprocess
from multiprocessing import Pool

import mlflow
from pyutils.general import ensure_dir, logger
from pyutils.config import configs


dataset = "fmnist"
model = "cnn"
root = f"log/{dataset}/{model}/test"
script = "crosstalk_spacing.py"
config_file = f"configs/{dataset}/{model}/train/sparse_train.yml"
configs.load(config_file, recursive=True)
keep_same = False

def task_launcher(args):
    lr, density, w_bit, in_bit, death_mode, growth_mode, init_mode, conv_block, crosstalk, interv_s_min, interv_s_max, interv_g_min, interv_g_max, redist, input_gate, output_gate, out_noise_std, id, gpu_id = args
    pres = [f"export CUDA_VISIBLE_DEVICES={gpu_id};", "python3", script, config_file]
    log_path = os.path.join(root, f"{density}_[{','.join([str(i) for i in conv_block])}]/{int(redist)}_ig-{int(input_gate)}_og-{int(output_gate)}_id-{id}")
    ensure_dir(os.path.join(
            root,
            f"{density}_[{','.join([str(i) for i in conv_block])}]"))
    with open(
        os.path.join(
            root,
            f"{density}_[{','.join([str(i) for i in conv_block])}]/{model}_{dataset}_lr-{lr:.3f}_wb-{w_bit}_ib-{in_bit}_dm-{death_mode}_gm-{growth_mode}_im-{init_mode}_cb-{conv_block}-without-opt_den-{density}_rd-{int(redist)}_ig-{int(input_gate)}_og-{int(output_gate)}_run-{id}.log",
        ),
        "w",
    ) as wfid:
        exp = [
            f"--run.batch_size={200}",
            f"--optimizer.lr={lr}",
            f"--run.random_state={42}",
            f"--model.conv_cfg.w_bit={w_bit}",
            f"--model.linear_cfg.w_bit={w_bit}",
            f"--dst_scheduler.death_mode={death_mode}",
            f"--dst_scheduler.growth_mode={growth_mode}",
            f"--dst_scheduler.init_mode={init_mode}",
            f"--dst_scheduler.density={density}",
            ##################################################################
            # Change this one to match the pruning model you used in training#
            ##################################################################
            # Modes are 
            # structure_row, structure_col, strcuture_row_col #

            "--dst_scheduler.pruning_type=structure_row",


            f"--noise.noise_flag={crosstalk}",
            f"--noise.crosstalk_flag={crosstalk}",
            f"--noise.output_noise_std={out_noise_std}",
            f"--loginfo={log_path}",
            f"--noise.crosstalk_scheduler.interv_s_min={interv_s_min}",
            f"--noise.crosstalk_scheduler.interv_s_max={interv_s_max}",
            f"--noise.crosstalk_scheduler.interv_g_min={interv_g_min}",
            f"--noise.crosstalk_scheduler.interv_g_max={interv_g_max}",
            f"--dst_scheduler.skip_first_layer={False}",
            f"--dst_scheduler.skip_last_layer={True}",
            f"--model.conv_cfg.miniblock=[{','.join([str(i) for i in conv_block])}]",
            f"--model.linear_cfg.miniblock=[{','.join([str(i) for i in conv_block])}]",
            f"--model.conv_cfg.in_bit={in_bit}",
            f"--model.linear_cfg.in_bit={in_bit}",
            f"--noise.light_redist={redist}",
            f"--noise.input_power_gating={input_gate}",
            f"--noise.output_power_gating={output_gate}",
            f"--checkpoint.resume={True}",
            #######################################################
            # Change this one into the actual checkpoint directory#
            #######################################################
            "--checkpoint.restore_checkpoint=./checkpoint/fmnist/cnn/train/TeMPO_CNN_lr-0.0020_wb-8_ib-6_cb-[4,4,16,16]_run-1_acc-91.42_epoch-8.pt",

            f"--checkpoint.model_comment=pretrain_lr-{lr:.4f}_wb-{w_bit}_ib-{in_bit}_cb-[{','.join([str(i) for i in conv_block])}]_run-{id}",
        ]
        cmd = " ".join(pres + exp)
        logger.info(f"running command:\n\t{cmd}")
        subprocess.call(cmd, stderr=wfid, stdout=wfid, shell=True)


if __name__ == "__main__":
    ensure_dir(root)
    mlflow.set_experiment(configs.run.experiment)  # set experiments first
    
    tasks = [
        (0.002, 1, 8, 6, "magnitude", "gradient", "uniform", [4, 4, 16, 16], 1, 9, 10, 1, 5, 0, 0, 0, 0.01, 0, 0),
        (0.002, 1, 8, 6, "magnitude_power", "gradient_power", "uniform_power", [4, 4, 16, 16], 1, 9, 10, 1, 5, 0, 0, 0, 0.01, 0, 0),
        (0.002, 1, 8, 6, "magnitude", "gradient", "uniform", [4, 4, 16, 16], 1, 9, 10, 1, 5, 1, 1, 1, 0.01, 1, 0),
        (0.002, 1, 8, 6, "magnitude_power", "gradient_power", "uniform_power", [4, 4, 16, 16], 1, 9, 10, 1, 5, 1, 1, 1, 0.01, 1, 0),

    ]

    with Pool(9) as p:
        p.map(task_launcher, tasks)
    logger.info(f"Exp: {configs.run.experiment} Done.")
