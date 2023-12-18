# ---------------------------------------------------------------
# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for Denoising Diffusion GAN. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------
import argparse
import torch
import numpy as np
import os
import torch.nn as nn
import torch.optim as optim
import yaml
import logging
import sys
import time
from torch.multiprocessing import Process
import torch.distributed as dist
import shutil
from datasets import get_dataset, data_transform, inverse_data_transform
from functions import get_optimizer
from models.diffusion import Model
from functions.losses import loss_registry
from ema import EMA

def init_processes(rank, size, fn, args, config):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = '6026'
    torch.cuda.set_device(args.local_rank)
    gpu = args.local_rank
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=size)
    fn(rank, gpu, args, config)
    dist.barrier()
    cleanup()

def cleanup():
    dist.destroy_process_group()  

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (
            np.linspace(
                beta_start ** 0.5,
                beta_end ** 0.5,
                num_diffusion_timesteps,
                dtype=np.float64,
            )
            ** 2
        )
    elif beta_schedule == "linear":
        betas = np.linspace(
            beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(
            num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64
        )
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas

#%%
def train(rank, gpu, args, config):
    def broadcast_params(params):
        for param in params:
            dist.broadcast(param.data, src=0)
    torch.manual_seed(args.seed + rank)
    torch.cuda.manual_seed(args.seed + rank)
    torch.cuda.manual_seed_all(args.seed + rank)
    device = torch.device('cuda:{}'.format(gpu))
    
    batch_size = config.training.batch_size
    betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
    betas = torch.from_numpy(betas).float().to(device)
    num_timesteps = betas.shape[0]
    
    dataset,_ = get_dataset(args, config)
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                                    num_replicas=args.world_size,
                                                                    rank=rank)
    data_loader = torch.utils.data.DataLoader(dataset,
                                            batch_size=batch_size,
                                            shuffle=False,
                                            num_workers=config.data.num_workers,
                                            pin_memory=True,
                                            sampler=train_sampler,
                                            drop_last = True)
    args.layout = False
    model = Model(config)
    model = model.to(device)
    broadcast_params(model.parameters())
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu],find_unused_parameters=True)
    optimizer = get_optimizer(config, model.parameters())
    
    if config.model.ema:
        optimizer = EMA(optimizer, ema_decay=config.model.ema_rate)
        optimizer.ema_start()
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.training.n_epochs, eta_min=1e-5)    
    exp_path = os.path.join(args.exp, "logs", args.doc)
    if rank == 0:
        if not os.path.exists(exp_path):
            os.makedirs(exp_path)
    
    if args.resume_training:
        checkpoint_file = os.path.join(exp_path, 'ckpt.pth')
        checkpoint = torch.load(checkpoint_file, map_location=device)
        init_epoch = checkpoint["epoch"]
        epoch = init_epoch
        model.load_state_dict(checkpoint["model_dict"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        step = checkpoint["step"]
        logging.info("=> loaded checkpoint (epoch {}, step {})".format(epoch, step))
    else:
        step, epoch, init_epoch = 0, 0, 0
    
    for epoch in range(init_epoch, config.training.n_epochs):
        train_sampler.set_epoch(epoch)
        data_start = time.time()
        data_time = 0
        for iteration, (x, y) in enumerate(data_loader):
            model.zero_grad()
            n = x.size(0)
            data_time += time.time() - data_start
            model.train()
            step += 1
            x = x.to(device, non_blocking=True)
            x = data_transform(config, x)
            e = torch.randn_like(x)
            b = betas
                      
            # antithetic sampling
            t = torch.randint(
                low=0, high=num_timesteps, size=(n // 2 + 1,)
            ).to(device)
            t = torch.cat([t, num_timesteps - t - 1], dim=0)[:n]
            if config.model.type == "simple":
                loss = loss_registry[config.model.type](model, x, t, e, b)
            elif config.model.type == "sa":
                loss = loss_registry[config.model.type](model, x, t, e, b, args.num_consecutive_steps, args.lamda)
            else:
                raise NotImplementedError("Loss type is not defined")

            if rank == 0:
                logging.info(
                    f"epoch: {epoch} step: {step}, loss: {loss.item()}, data time: {data_time / (iteration+1)}"
                )

            loss.backward()
            try:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config.optim.grad_clip
                )
            except Exception:
                pass
            optimizer.step()
            
        if rank == 0:
            if epoch % config.training.snapshot_freq == 0:
                states = dict({'model_dict': model.state_dict(),
                            'epoch': epoch,
                            'args': args,
                            'optimizer': optimizer.state_dict(),
                            'scheduler': scheduler.state_dict(),
                            'step': step})
                torch.save(
                    states,
                    os.path.join(exp_path, "ckpt_{}.pth".format(epoch)),
                )
                torch.save(states, os.path.join(exp_path, "ckpt.pth"))
                if config.model.ema:
                    optimizer.swap_parameters_with_ema(store_params_in_ema=True)               
                torch.save(model.state_dict(), os.path.join(exp_path, 'model_{}_ema.pth'.format(epoch)))
                if config.model.ema:
                    optimizer.swap_parameters_with_ema(store_params_in_ema=True)

        # if not args.no_lr_decay:
        #     scheduler.step() 

#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser('dpm parameters')
    parser.add_argument('--seed', type=int, default=1024,
                        help='seed used for initialization')
        
    ###ddp
    parser.add_argument('--num_proc_node', type=int, default=1,
                        help='The number of nodes in multi node env.')
    parser.add_argument('--num_process_per_node', type=int, default=1,
                        help='number of gpus')
    parser.add_argument('--node_rank', type=int, default=0,
                        help='The index of node.')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='rank of process in the node')
    parser.add_argument('--master_address', type=str, default='127.0.0.1',
                        help='address for master')

    #diffusion
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config file"
    )
    parser.add_argument(
        "--exp", type=str, default="exp", help="Path for saving running related data."
    )
    parser.add_argument(
        "--doc",
        type=str,
        required=True,
        help="A string for documentation purpose. "
        "Will be the name of the log folder.",
    )
    parser.add_argument(
        "--comment", type=str, default="", help="A string for experiment comment"
    )
    parser.add_argument(
        "--verbose",
        type=str,
        default="info",
        help="Verbose level: info | debug | warning | critical",
    )
    parser.add_argument(
        "--resume_training", action="store_true", help="Whether to resume training"
    )
    parser.add_argument(
        "--ni",
        action="store_true",
        help="No interaction. Suitable for Slurm Job launcher",
    )
    parser.add_argument("--lamda", type=float, default=1., help="lambda coef of SA loss")
    parser.add_argument("--num_consecutive_steps", type=int, default=2, help="number of consecutive steps in SA loss")

    args = parser.parse_args()

    with open(os.path.join("configs", args.config), "r") as f:
        config = yaml.safe_load(f)
    def dict2namespace(config):
        namespace = argparse.Namespace()
        for key, value in config.items():
            if isinstance(value, dict):
                new_value = dict2namespace(value)
            else:
                new_value = value
            setattr(namespace, key, new_value)
        return namespace
    config = dict2namespace(config)

    args.log_path = os.path.join(args.exp, "logs", args.doc)
    if not args.resume_training:
        if os.path.exists(args.log_path):
            overwrite = False
            if args.ni:
                overwrite = True
            else:
                response = input("Folder already exists. Overwrite? (Y/N)")
                if response.upper() == "Y":
                    overwrite = True

            if overwrite:
                shutil.rmtree(args.log_path)
                os.makedirs(args.log_path)
            else:
                print("Folder exists. Program halted.")
                sys.exit(0)
        else:
            os.makedirs(args.log_path)

        with open(os.path.join(args.log_path, "config.yml"), "w") as f:
            yaml.dump(config, f, default_flow_style=False)
    
    # setup logger
    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError("level {} not supported".format(args.verbose))

    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(os.path.join(args.log_path, "stdout.txt"))
    formatter = logging.Formatter(
        "%(levelname)s - %(filename)s - %(asctime)s - %(message)s"
    )
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    logger.setLevel(level)

    args.world_size = args.num_proc_node * args.num_process_per_node
    size = args.num_process_per_node
    
    # torch.multiprocessing.set_start_method('spawn')# good solution !!!!
    if size > 1:
        processes = []
        for rank in range(size):
            args.local_rank = rank
            global_rank = rank + args.node_rank * args.num_process_per_node
            global_size = args.num_proc_node * args.num_process_per_node
            args.global_rank = global_rank
            print('Node rank %d, local proc %d, global proc %d' % (args.node_rank, rank, global_rank))
            p = Process(target=init_processes, args=(global_rank, global_size, train, args, config))
            p.start()
            processes.append(p)
            
        for p in processes:
            p.join()
    else:
        print('starting in debug mode') 
        init_processes(0, size, train, args, config)