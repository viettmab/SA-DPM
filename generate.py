import argparse
import torch
import numpy as np
import os
import torch.nn as nn
import yaml
import tqdm
from torch.multiprocessing import Process
import torch.distributed as dist
from datasets import inverse_data_transform
from models.diffusion import Model
from pytorch_fid import fid_score
import re
import torchvision.utils as tvu

device=torch.device('cuda')

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

def sample_image(args, config,  x, model, randn_like=torch.randn_like, last=True):
    betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )
    betas = torch.from_numpy(betas).float().to(device)
    num_timesteps = betas.shape[0]
    try:
        skip = args.skip
    except Exception:
        skip = 1
    if args.sample_type == "generalized":
        if args.skip_type == "uniform":
            skip = num_timesteps // args.timesteps
            seq = range(0, num_timesteps, skip)
        elif args.skip_type == "quad":
            seq = (
                np.linspace(
                    0, np.sqrt(num_timesteps * 0.8), args.timesteps
                )
                ** 2
            )
            seq = [int(s) for s in list(seq)]
        else:
            raise NotImplementedError
        from functions.denoising import generalized_steps

        xs = generalized_steps(x, seq, model, betas, randn_like, eta=args.eta)
        x = xs
    elif args.sample_type == "ddpm_noisy":
        if args.skip_type == "uniform":
            skip = num_timesteps // args.timesteps
            seq = range(0, num_timesteps, skip)
        elif args.skip_type == "quad":
            seq = (
                np.linspace(
                    0, np.sqrt(num_timesteps * 0.8), args.timesteps
                )
                ** 2
            )
            seq = [int(s) for s in list(seq)]
        else:
            raise NotImplementedError
        from functions.denoising import ddpm_steps

        x = ddpm_steps(x, seq, model, betas)
    else:
        raise NotImplementedError
    if last:
        x = x[0][-1]
    return x

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])
        
def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

def init_processes(rank, size, fn, args, config):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = args.master_address
    os.environ['MASTER_PORT'] = '6020'
    torch.cuda.set_device(args.local_rank)
    gpu = args.local_rank
    dist.init_process_group(backend='nccl', init_method='env://', rank=rank, world_size=size)
    fn(rank, gpu, args, config)
    dist.barrier()
    cleanup()  

def cleanup():
    dist.destroy_process_group()

def sample(rank, gpu, args, config):
    def broadcast_params(params):
        for param in params:
            dist.broadcast(param.data, src=0)
    seeds=args.seeds
    num_batches = ((len(args.seeds) - 1) // (config.sampling.batch_size * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]

    # Load network.
    model = Model(config)
    if rank == 0:
        print(f"Loading checkpoint from model_{args.ckpt_id}_ema.pth")
    states = torch.load(
        os.path.join(
            args.log_path, f"model_{args.ckpt_id}_ema.pth"
        ),
        map_location=device,
    )
    model = model.to(device)
    broadcast_params(model.parameters())
    model = nn.parallel.DistributedDataParallel(model, device_ids=[gpu],find_unused_parameters=True)
    model.load_state_dict(states, strict=True)
    model.eval()

    # Loop over batches.
    if rank == 0:
        print(f'Generating {len(seeds)} images to "{args.image_folder}"...')
    for batch_seeds in tqdm.tqdm(rank_batches, unit='batch', disable=(dist.get_rank() != 0)):
        torch.distributed.barrier()
        batch_size = len(batch_seeds)
        if batch_size == 0:
            continue

        # Pick latents and labels.
        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn([batch_size, config.data.channels, config.data.image_size, config.data.image_size], device=device)
        images = sample_image(args, config, latents, model, randn_like=rnd.randn_like)

        # Save images.
        images = inverse_data_transform(config, images)
        os.makedirs(args.image_folder, exist_ok=True)

        img_id = 0
        for seed in batch_seeds:
            image_path = os.path.join(args.image_folder, f'{seed:06d}.png')
            tvu.save_image(
                images[img_id], image_path
            )
            img_id += 1

    # Done.
    dist.barrier()
    if rank == 0:
        print('Done.')
        if config.data.dataset == "CIFAR10":
            fid_value = fid_score.calculate_fid_given_paths([args.image_folder,
                                                            "pytorch_fid/cifar10_train_stat.npy"], 50, "cuda", 2048)
        elif config.data.dataset == "CELEBA":
            fid_value = fid_score.calculate_fid_given_paths([args.image_folder,
                                                            "pytorch_fid/fid_stats_celeba64.npz"], 50, "cuda", 2048)
        elif config.data.dataset == "LSUN" and config.data.category == "church_outdoor":
            fid_value = fid_score.calculate_fid_given_paths([args.image_folder,
                                                            "pytorch_fid/lsun_church_stat.npy"], 50, "cuda", 2048)
        elif config.data.dataset == "CELEBAHQ":
            fid_value = fid_score.calculate_fid_given_paths([args.image_folder,
                                                            "pytorch_fid/celebahq_stat.npy"], 50, "cuda", 2048)
        elif config.data.dataset == "FFHQ64":
            fid_value = fid_score.calculate_fid_given_paths([args.image_folder,
                                                            "pytorch_fid/ffhq-64x64.npz"], 50, "cuda", 2048)
        elif config.data.dataset == "AFHQ":
            fid_value = fid_score.calculate_fid_given_paths([args.image_folder,
                                                        "pytorch_fid/afhqv2-64x64.npz"], 50, "cuda", 2048)
        path_name = '/'.join(args.image_folder.split('/')[-2:])
        with open(args.fid_log, 'a') as f:
            f.write(f'Checkpoint {path_name}  --> FID {fid_value}\n')
        print(f'Checkpoint {path_name}  --> FID {fid_value}\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser('dpm parameters')

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
        "-i",
        "--image_folder",
        type=str,
        default="images",
        help="The folder name of samples",
    )
    parser.add_argument(
        "--ni",
        action="store_true",
        help="No interaction. Suitable for Slurm Job launcher",
    )
    parser.add_argument(
        "--sample_type",
        type=str,
        default="generalized",
        help="sampling approach (generalized or ddpm_noisy)",
    )
    parser.add_argument(
        "--skip_type",
        type=str,
        default="uniform",
        help="skip according to (uniform or quadratic)",
    )
    parser.add_argument(
        "--timesteps", type=int, default=1000, help="number of steps involved"
    )
    parser.add_argument(
        "--eta",
        type=float,
        default=0.0,
        help="eta used to control the variances of sigma",
    )
    parser.add_argument(
        "--fid_log",
        type=str,
        default="fid.txt",
        help="File to log FID",
    )
    parser.add_argument("--ckpt_id", type=int, default=500000, help="ckpt id")
    parser.add_argument("--num_samples", type=int, default=50000, help="Number of generated samples")
    parser.add_argument("--model_ema", action="store_true")
    parser.add_argument('--seeds', help='Random seeds (e.g. 1,2,5-10)', metavar='LIST', type=parse_int_list, default='0-63')

    args = parser.parse_args()
    args.log_path = os.path.join(args.exp, "logs", args.doc)
    args.image_folder = os.path.join(
            args.exp, "image_samples", args.image_folder
        )
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

    args.world_size = args.num_proc_node * args.num_process_per_node
    size = args.num_process_per_node
    
    if size > 1:
        processes = []
        for rank in range(size):
            args.local_rank = rank
            global_rank = rank + args.node_rank * args.num_process_per_node
            global_size = args.num_proc_node * args.num_process_per_node
            args.global_rank = global_rank
            print('Node rank %d, local proc %d, global proc %d' % (args.node_rank, rank, global_rank))
            p = Process(target=init_processes, args=(global_rank, global_size, sample, args, config))
            p.start()
            processes.append(p)
            
        for p in processes:
            p.join()
    else:
        print('starting in debug mode') 
        init_processes(0, size, sample, args, config)

    # main(args, config)

#----------------------------------------------------------------------------
