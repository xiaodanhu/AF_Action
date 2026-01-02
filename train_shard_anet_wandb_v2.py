import os
import argparse
import time
from datetime import datetime
import pprint
import math
import json
import warnings
import re
import torch
from torch.utils.data import DistributedSampler
import deepspeed
import wandb
import multiprocessing as mp
import torch.distributed as dist
import sys
import subprocess

from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader, make_data_loader_distributed
from libs.modeling import make_meta_arch
from libs.utils import (
    train_one_epoch, valid_one_epoch, ANETdetection,
    make_optimizer, make_scheduler,
    fix_random_seed, TrainingLogger
)
# --- Free GPU memory and clean up ---
import gc


warnings.filterwarnings("ignore")


def train_worker(args, cfg):

    # Distributed setup
    torch.cuda.set_device(args.local_rank)
    deepspeed.init_distributed()

    # Load configs
    with open(args.deepspeed_config, 'r') as f:
        ds_config = json.load(f)

    # Load wandb config
    if args.local_rank == 0:
        wandb.init(
            config=cfg,
            reinit=True
        )
        run_config = wandb.config
        cfg["dataset"]["json_file"] = run_config["json_file"]
        cfg["dataset"]["max_seq_len"] = run_config["max_seq_len"]
        cfg["opt"]["learning_rate"] = run_config["learning_rate"]
        cfg["loader"]["batch_size"] = run_config["batch_size"]
        ds_config["gradient_accumulation_steps"] = run_config["gradient_accumulation_steps"]

    # Extract window size from json_file name (e.g., anet_win128_int32.json)
    json_file = cfg["dataset"]["json_file"]
    match = re.search(r'win(\d+)', os.path.basename(json_file))
    window_size = int(match.group(1))
    max_seq_len = cfg["dataset"]["max_seq_len"]
    # Check on all ranks
    error_flag = torch.tensor([0], device='cuda')
    if window_size >= max_seq_len:
        if args.local_rank == 0:
            print(f"Window size ({window_size}) must be smaller than max_seq_len ({max_seq_len})! Skipping sweep parameters.")
        error_flag[0] = 1

    # Broadcast error to all ranks
    if dist.is_initialized():
        dist.broadcast(error_flag, src=0)

    if error_flag.item() == 1:
        # All ranks exit cleanly
        os._exit(1)


    # Prepare output folder (rank 0)
    if not os.path.exists(cfg['output_folder']) and args.local_rank == 0:
        os.makedirs(cfg['output_folder'], exist_ok=True)
    cfg_filename = os.path.basename(args.config).replace('.yaml', '')
    ts = datetime.fromtimestamp(int(time.time()))
    if len(args.output) == 0:
        ckpt_folder = os.path.join(
            cfg['output_folder'], cfg_filename + '_' + str(ts))
    else:
        ckpt_folder = os.path.join(
            cfg['output_folder'], cfg_filename + '_' + str(args.output))
    if not os.path.exists(ckpt_folder) and args.local_rank == 0:
        os.makedirs(ckpt_folder, exist_ok=True)

    # Seed and adjust for GPUs
    rng = fix_random_seed(cfg['init_rand_seed'], include_cuda=True)
    # num_gpus = torch.distributed.get_world_size()
    cfg['opt']['learning_rate'] *= 2 # num_gpus
    cfg['loader']['num_workers'] *= 2 # num_gpus


    try:
        # Data
        train_ds = make_dataset(cfg['dataset_name'], True, cfg['train_split'], cfg['model']['backbone_type'], cfg['round'], **cfg['dataset'])
        train_db_vars = train_ds.get_attributes()
        cfg['model']['train_cfg']['head_empty_cls'] = train_db_vars['empty_label_ids']

        train_sampler = DistributedSampler(train_ds, shuffle=True)
        train_sampler_test = DistributedSampler(train_ds, shuffle=False)
        train_loader = make_data_loader_distributed(train_ds, train_sampler, True, rng, **cfg['loader'])
        train_loader_test = make_data_loader_distributed(train_ds, train_sampler_test, False, None, batch_size=1, num_workers=cfg['loader']['num_workers'])

        val_ds = make_dataset(cfg['dataset_name'], False, cfg['val_split'], cfg['model']['backbone_type'], cfg['round'], **cfg['dataset'])
        val_loader = make_data_loader(val_ds, False, None, batch_size=1, num_workers=cfg['loader']['num_workers'])

        # Model, optimizer, scheduler
        cfg['model']['active_learning_method'] = cfg.get('active_learning_method', None)
        model = make_meta_arch(cfg['model_name'], **cfg['model'])

        params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = make_optimizer(model, cfg['opt'])
        steps_per_epoch = math.ceil(len(train_loader) / ds_config['gradient_accumulation_steps'])
        scheduler = make_scheduler(optimizer, cfg['opt'], steps_per_epoch)

        model_engine, optimizer, _, scheduler = deepspeed.initialize(
            model=model,
            model_parameters=params,
            optimizer=optimizer,
            lr_scheduler=scheduler,
            config_params="configs/deepspeed_config.json"
        )

        # Resume if needed
        start_epoch = 0
        if args.resume:
            if os.path.exists(os.path.join(ckpt_folder, args.resume)):
                ok, client_sd = model_engine.load_checkpoint(ckpt_folder, args.resume, load_optimizer_states=False, load_lr_scheduler_states=False)
                if ok:
                    start_epoch = client_sd.get('epoch', 0)
                    if args.local_rank == 0:
                        print(f"Resumed from {args.resume} at epoch {start_epoch}")

        # Logger and WandB
        logger = TrainingLogger(os.path.join(ckpt_folder, f"log_{datetime.now():%m%d_%H%M}.txt"))

        if args.local_rank == 0:

            logger.log(pprint.pformat(cfg))

            logger.log("****************** important paras begin *****************")
            logger.log("dataset_name: {}".format(cfg["dataset_name"]))
            logger.log("round: {}".format(cfg["round"]))
            logger.log("active_learning_method: {}".format(cfg["active_learning_method"]))
            logger.log("json_file: {}".format(cfg["dataset"]["json_file"]))
            logger.log("max_seq_len: {}".format(cfg["dataset"]["max_seq_len"]))
            logger.log("num_classes: {}".format(cfg["dataset"]["num_classes"]))
            logger.log("num_frames: {}".format(cfg["dataset"]["num_frames"]))
            logger.log("backbone_type: {}".format(cfg["model"]["backbone_type"]))
            logger.log("backbone_arch: {}".format(cfg["model"]["backbone_arch"]))
            logger.log("regression_range: {}".format(cfg["model"]["regression_range"]))
            logger.log("epochs: {}".format(cfg["opt"]["epochs"]))
            logger.log("learning_rate: {}".format(cfg["opt"]["learning_rate"]))
            logger.log("batch_size: {}".format(cfg["loader"]["batch_size"]))
            logger.log("gradient_accumulation_steps: {}".format(ds_config["gradient_accumulation_steps"]))
            logger.log("****************** important paras end *****************")

        """ training / validation loop """
        print("\nStart training model {:s} ...".format(cfg['model_name']))

        # Training loop
        max_epochs = cfg['opt'].get(
            'early_stop_epochs',
            cfg['opt']['epochs'] + cfg['opt']['warmup_epochs']
        )
        best_mAP, best_epoch = 0.0, 0
        for epoch in range(start_epoch, max_epochs):
            train_loss = train_one_epoch(
                train_loader, model_engine, optimizer, scheduler, epoch,
                logger, model_ema=None,
                clip_grad_l2norm=cfg['train_cfg']['clip_grad_l2norm'],
                print_freq=args.print_freq, save_log=(args.local_rank==0)
            )
            if args.local_rank == 0:
                wandb.log({"train/loss": train_loss, "train/epoch": epoch})

            # Checkpoint & eval
            if (epoch+1) == max_epochs or (args.ckpt_freq > 0 and (epoch+1) % args.ckpt_freq == 0):
                model_engine.save_checkpoint(ckpt_folder, f'epoch_{epoch+1:03d}', client_state={'epoch': epoch})
                
                # train mAP
                train_eval = ANETdetection(train_ds.json_file, train_ds.split[0], tiou_thresholds=train_db_vars['tiou_thresholds'])

                mAP_train = valid_one_epoch(train_loader_test, model_engine, epoch, evaluator=train_eval, print_freq=args.print_freq, if_save_data=False)
                print("Epoch: ", epoch, ", Train mAP: ", mAP_train)
                if args.local_rank == 0:
                    logger.log(f"[Train] Epoch {epoch}: Trainset mAP = {mAP_train:.4f}")
                    wandb.log({"train/mAP": mAP_train, "epoch": epoch})

                ##############################

                # val mAP
                val_attrs = val_ds.get_attributes()
                val_eval = ANETdetection(val_ds.json_file, val_ds.split[0], tiou_thresholds=val_attrs['tiou_thresholds'])

                mAP_val = valid_one_epoch(val_loader, model_engine, epoch, evaluator=val_eval, output_file=str(ts), print_freq=args.print_freq, if_save_data=True)
                print("Epoch: ", epoch, ", Test mAP: ", mAP_val)
                if args.local_rank == 0:
                    logger.log(f"[Test] Epoch {epoch}: Testset mAP = {mAP_val:.4f}")
                    wandb.log({"val/mAP": mAP_val, "epoch": epoch})

                # Save the best model
                if mAP_val > best_mAP:
                    best_mAP, best_epoch = mAP_val, epoch
                    if args.local_rank==0:
                        model_engine.save_checkpoint(ckpt_folder, 'best_model', client_state={'epoch': epoch})
                    print(f"New best model at epoch {epoch}: {best_mAP:.4f}")

        if args.local_rank == 0:
            # wrap up
            logger.log(f"Best model at epoch {best_epoch}: mAP = {best_mAP:.4f}")
            print("Training complete.")
            wandb.summary['best_mAP'] = best_mAP
            wandb.finish()

    except Exception as e:
        print(f"Exception occurred during training: {e}")
        import traceback
        traceback.print_exc()
        os._exit(1)

    finally:
        # Delete model and optimizer references
        try:
            del model_engine
        except Exception:
            pass
        try:
            del model
        except Exception:
            pass
        try:
            del optimizer
        except Exception:
            pass
        try:
            del scheduler
        except Exception:
            pass

        # Empty CUDA cache
        torch.cuda.empty_cache()
        gc.collect()
        # If using deepspeed, also call deepspeed's cleanup if available
        if hasattr(deepspeed, "zero"):
            try:
                deepspeed.zero.Init.deallocate()
            except Exception:
                pass

def _run_single(args, cfg):
    """Helper for one sweep trial; runs in its own process."""
    try:
        train_worker(args, cfg)
    except Exception as e:
        print(f"[trial failed] {e}", file=sys.stderr)
        # fatal: kill the whole process
        os._exit(1)
    # clean success: also kill the process so GPU contexts go with it
    os._exit(0)

def run_sweep(args, cfg):
    wandb.login()
    sweep_config = {
        'method': 'random',
        'metric': {'name': 'val/mAP', 'goal': 'maximize'},
        'parameters': {
            'json_file': {'values': ['/data3/xiaodan8/Activitynet/anet_win128_int32.json', '/data3/xiaodan8/Activitynet/anet_win256_int16.json']},
            'max_seq_len': {'values': [144, 192, 288, 576]},
            'learning_rate': {'distribution': 'log_uniform_values', 'min': 1e-5, 'max': 1e-2},
            'batch_size': {'distribution': 'q_uniform', 'min': 1, 'max': 6, 'q': 2},
            'gradient_accumulation_steps': {'values': [1, 2, 4, 8]}
        }
    }
    sweep_id = wandb.sweep(sweep_config, project='deepSpeed')

    def _agent_fn():
        config = wandb.config
        # build the exact same CLI you’d use on the command line:
        cmd = [
        "deepspeed",
        "--include=localhost:2,3",
        "--master_port=12106",
        "train_shard_anet_wandb.py",
        "--config", args.config
        ]
        # Let the OS reclaim *all* GPU memory when this exits:
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"Trial job failed (exit code {result.returncode}). GPUs freed.")
        else:
            print("Trial job finished successfully. GPUs freed.")

    wandb.agent(sweep_id, function=_agent_fn, count=5)

def main(args):
    cfg = load_config(args.config)
    run_sweep(args, cfg)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Training with DeepSpeed & W&B')
    parser.add_argument('--config', type=str, default='./configs/anet_i3d_v2.yaml')
    parser.add_argument('--deepspeed-config', type=str, default='./configs/deepspeed_config.json')
    parser.add_argument('--print-freq', type=int, default=10)
    parser.add_argument('--ckpt-freq', type=int, default=1)
    parser.add_argument('--output', type=str, default='deepspeed')
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--local_rank', type=int, default=-1)
    args = parser.parse_args()
    print(args.local_rank)

    main(args)
    # Run the script with the following command:

'''
deepspeed --include="localhost:2,3" --master_port 12106 train_shard_anet_wandb.py
'''