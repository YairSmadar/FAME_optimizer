from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import json
import logging
import os
import pprint
import sys
import time
from copy import copy

import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import wandb
from torch.utils.collect_env import get_pretty_env_info

# Assuming the 'CvT' directory is in the parent directory of the current script
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import _init_paths
from CvT.lib.config import config
from CvT.lib.config import update_config
from CvT.lib.config import save_config
from CvT.lib.core.loss import build_criterion
from CvT.lib.core.function import train_one_epoch, test
from CvT.lib.dataset import build_dataloader
from CvT.lib.models import build_model
from CvT.lib.optim import build_optimizer
from CvT.lib.scheduler import build_lr_scheduler
from CvT.lib.utils.comm import comm
from CvT.lib.utils.utils import create_logger
from CvT.lib.utils.utils import init_distributed
from CvT.lib.utils.utils import setup_cudnn
from CvT.lib.utils.utils import summary_model_on_master
from CvT.lib.utils.utils import resume_checkpoint
from CvT.lib.utils.utils import save_checkpoint_on_master
from CvT.lib.utils.utils import save_model_on_master


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train classification network')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        required=False,
                        type=str)

    # distributed training
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--port", type=int, default=9000)
    parser.add_argument("--optimizer", type=str, default="sgd")


    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    parser.add_argument('--config_json', type=str)
    args = parser.parse_args()

    return args

def apply_config(args: argparse.Namespace, config_path: str):
    """Overwrite the values in an arguments object by values of namesake
    keys in a JSON config file.

    :param args: The arguments object
    :param config_path: the path to a config JSON file.
    """
    config_path = copy(config_path)
    if config_path:
        # Opening JSON file
        f = open(config_path)
        config_overwrite = json.load(f)
        for k, v in config_overwrite.items():
            if k.startswith('_'):
                continue
            setattr(args, k, v)


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def generate_wandb_name(args):
    name = f"model-{args.model_name}"
    name += f"_dataset-{args.dataset}"
    name += f"_optim-{args.optimizer}"

    if args.optimizer == 'fame':
        name += f"_b3-{args.beta3}"
        name += f"_b4-{args.beta4}"

    name += f'_seed-{args.seed}'

    return name

def main():
    args = parse_args()
    apply_config(args, args.config_json)

    set_seed(args.seed)
    init_distributed(args)
    setup_cudnn(config)

    wandb_name = generate_wandb_name(args)
    if args.use_wandb:
        wandb.init(project="FAME_optimizer",
                   entity="the-smadars",
                   name=wandb_name,
                   config=args)
        wandb.run.summary["best_test_accuracy"] = 0
        wandb.run.summary["best_test_loss"] = 999

    update_config(config, args)
    final_output_dir = create_logger(config, args.cfg, 'train')
    tb_log_dir = final_output_dir

    # if comm.is_main_process():
        # logging.info("=> collecting env info (might take some time)")
        # logging.info("\n" + get_pretty_env_info())
        # logging.info(pprint.pformat(args))
        # logging.info(config)
        # logging.info("=> using {} GPUs".format(args.num_gpus))

    output_config_path = os.path.join(final_output_dir, 'config.yaml')
    logging.info("=> saving config into: {}".format(output_config_path))
    save_config(config, output_config_path)

    model = build_model(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # copy model file
    summary_model_on_master(model, config, final_output_dir, True)

    if config.AMP.ENABLED and config.AMP.MEMORY_FORMAT == 'nhwc':
        logging.info('=> convert memory format to nhwc')
        model.to(memory_format=torch.channels_last)


    best_perf = 0.0
    best_model = True
    begin_epoch = config.TRAIN.BEGIN_EPOCH
    optimizer = build_optimizer(config, model, args)

    best_perf, begin_epoch = resume_checkpoint(
        model, optimizer, config, final_output_dir, True
    )

    train_loader = build_dataloader(config, True, args.distributed, args)
    valid_loader = build_dataloader(config, False, args.distributed, args)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.local_rank],
            output_device=args.local_rank,
            find_unused_parameters=True
        )

    criterion = build_criterion(config)
    criterion.cuda()
    criterion_eval = build_criterion(config, train=False)
    criterion_eval.cuda()

    lr_scheduler = build_lr_scheduler(config, optimizer, begin_epoch)

    scaler = torch.cuda.amp.GradScaler(enabled=config.AMP.ENABLED)

    logging.info('=> start training')
    for epoch in range(begin_epoch, config.TRAIN.END_EPOCH):
        head = 'Epoch[{}]:'.format(epoch)
        logging.info('=> {} epoch start'.format(head))

        start = time.time()
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)

        # train for one epoch
        logging.info('=> {} train start'.format(head))
        with torch.autograd.set_detect_anomaly(config.TRAIN.DETECT_ANOMALY):
            train_one_epoch(config, train_loader, model, criterion, optimizer,
                            epoch, final_output_dir, tb_log_dir,
                            scaler=scaler)
        logging.info(
            '=> {} train end, duration: {:.2f}s'
            .format(head, time.time()-start)
        )

        # evaluate on validation set
        logging.info('=> {} validate start'.format(head))
        val_start = time.time()

        if epoch >= config.TRAIN.EVAL_BEGIN_EPOCH:
            perf = test(
                config, valid_loader, model, criterion_eval,
                final_output_dir, tb_log_dir,
                args.distributed, args, wandb_name
            )

            best_model = (perf > best_perf)
            best_perf = perf if best_model else best_perf

        logging.info(
            '=> {} validate end, duration: {:.2f}s'
            .format(head, time.time()-val_start)
        )

        lr_scheduler.step(epoch=epoch+1)
        if config.TRAIN.LR_SCHEDULER.METHOD == 'timm':
            lr = lr_scheduler.get_epoch_values(epoch+1)[0]
        else:
            lr = lr_scheduler.get_last_lr()[0]
        logging.info(f'=> lr: {lr}')

        save_checkpoint_on_master(
            model=model,
            distributed=args.distributed,
            model_name=config.MODEL.NAME,
            optimizer=optimizer,
            output_dir=final_output_dir,
            in_epoch=True,
            epoch_or_step=epoch,
            best_perf=best_perf,
        )

        if best_model and comm.is_main_process():
            save_model_on_master(
                model, args.distributed, final_output_dir, 'model_best.pth'
            )

        if config.TRAIN.SAVE_ALL_MODELS and comm.is_main_process():
            save_model_on_master(
                model, args.distributed, final_output_dir, f'model_{epoch}.pth'
            )

        logging.info(
            '=> {} epoch end, duration : {:.2f}s'
            .format(head, time.time()-start)
        )

    save_model_on_master(
        model, args.distributed, final_output_dir, 'final_state.pth'
    )

    if config.SWA.ENABLED and comm.is_main_process():
        save_model_on_master(
             args.distributed, final_output_dir, 'swa_state.pth'
        )

    logging.info('=> finish training')


if __name__ == '__main__':
    main()
