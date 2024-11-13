import inspect
import logging
import submitit
import torch
import os
import stable_ssl.utils


def patch_setup_distributed(args):
    """Set up the distributed environment for PyTorch."""
    logging.info("Setting up Distributed model.")
    logging.info("Exporting PyTorch distributed environment variables.")

    try:
        submitit_env = submitit.helpers.TorchDistributedEnvironment().export()
        dist_env = {
            "world_size": submitit_env.world_size,
            "global_rank": submitit_env.rank,
            "local_rank": submitit_env.local_rank,
            "local_world_size": submitit_env.local_world_size,
            "host_name": submitit_env.master_addr,
            "port": submitit_env.master_port,
        }
        args.port = submitit_env.master_port
    except Exception as e:
        logging.error("Submitit environment not detected:", exc_info=e)
        raise RuntimeError("Submitit environment not detected.")
    if "SLURM_JOB_NODELIST" in os.environ:
        logging.info("Running on SLURM with submitit configs!")
    else:
        logging.info("Running on local machine with sumbitit configs!")
        # local host being used irrespective of hydra

    dist_url = f"tcp://{dist_env['host_name']}:{args.port}"

    os.environ["MASTER_ADDR"] = dist_env['host_name']
    os.environ["MASTER_PORT"] = str(args.port)

    logging.info(f"MASTER_ADDR:\n\t{os.getenv('MASTER_ADDR')}")
    logging.info(f"MASTER_PORT:\n\t{os.getenv('MASTER_PORT')}")
    logging.info(f"\trank: {dist_env['global_rank']}")
    logging.info(f"\tworld size: {dist_env['world_size']}")
    logging.info(f"\tlocal rank: {dist_env['local_rank']}")

    if not torch.distributed.is_available():
        raise RuntimeError(
            "torch.distributed is not available. Cannot initialize "
            "distributed process group."
        )
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(
            "nccl",
            init_method=dist_url,
            rank=dist_env["global_rank"],
            world_size=dist_env['world_size'],
        )
        args.world_size = dist_env['world_size']
        args.gpu_id = dist_env["local_rank"]
        assert (
            dist_env.get("global_rank", 0) == torch.distributed.get_rank()
        ), logging.error(
            "Torch and submitit global ranks do not match. "
            f"{dist_env.get('global_rank', 0)}, {torch.distributed.get_rank()}"
        )
        assert (dist_env['world_size']) == torch.distributed.get_world_size(), logging.error(
            "Torch and submitit world size do not match. "
            f"{dist_env['world_size']}, {torch.distributed.get_world_size()}"
        )
    return args


def patch_stable_ssl():
    changed = []
    stable_ssl.utils.setup_distributed = patch_setup_distributed
    changed.append(stable_ssl.utils.setup_distributed)
    return changed


if __name__ == "__main__":
    changed = patch_stable_ssl()
    print("Patched functions:", changed)
    print(inspect.getsource(changed[0]))
