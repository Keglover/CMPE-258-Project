import torch

def cuda_snapshot(tag):
    if torch.cuda.is_available():
        print(
            f"{tag}: alloc={torch.cuda.memory_allocated()/2**20:.2f} MiB, "
            f"reserved={torch.cuda.memory_reserved()/2**20:.2f} MiB, "
            f"peak_alloc={torch.cuda.max_memory_allocated()/2**20:.2f} MiB"
        )