from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import torch
from torch import cuda

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

try:
    from agents import CNNAttacker, Defender
except ImportError as exc:
    raise ImportError(
        "Unable to import project modules. Run this script from the project root or ensure src/ is on PYTHONPATH."
    ) from exc


def fmt_bytes(value: int) -> str:
    if value is None:
        return "n/a"
    for unit in ["B", "KiB", "MiB", "GiB", "TiB"]:
        if value < 1024.0 or unit == "TiB":
            return f"{value:.2f} {unit}"
        value /= 1024.0
    return f"{value:.2f} TiB"


def query_nvidia_smi() -> str:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return "nvidia-smi not available"


def cuda_stats(device: torch.device) -> dict[str, str]:
    if not cuda.is_available() or device.type != "cuda":
        return {}

    return {
        "allocated": fmt_bytes(cuda.memory_allocated(device)),
        "reserved": fmt_bytes(cuda.memory_reserved(device)),
        "max_allocated": fmt_bytes(cuda.max_memory_allocated(device)),
        "max_reserved": fmt_bytes(cuda.max_memory_reserved(device)),
        "summary": cuda.memory_summary(device, abbreviated=True),
    }


def print_cuda_stats(label: str, device: torch.device, show_summary: bool = False) -> None:
    stats = cuda_stats(device)
    if not stats:
        print(f"{label}: CUDA not available on {device}")
        return

    print("""
============================================================
""")
    print(f"{label}")
    print(f"  device: {device}")
    print(f"  allocated:      {stats['allocated']}")
    print(f"  reserved:       {stats['reserved']}")
    print(f"  max allocated:  {stats['max_allocated']}")
    print(f"  max reserved:   {stats['max_reserved']}")
    if show_summary:
        print("""
--- memory summary ---
""")
        print(stats["summary"])
    print("""
============================================================
""")


def build_dummy_batch(
    batch_size: int,
    seq_len: int,
    adv_len: int,
    vocab_size: int,
    edit_mask: bool,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    x_mal = torch.randint(0, vocab_size - 1, (batch_size, seq_len), dtype=torch.long, device=device)
    if edit_mask:
        mask = torch.zeros((batch_size, seq_len), dtype=torch.float32, device=device)
        # mark the last adv_len positions as editable when using an edit mask.
        end = min(seq_len, adv_len)
        mask[:, -end:] = 1.0
        return x_mal, mask
    return x_mal, None


def sync_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_adversary_audit(
    batch_size: int,
    seq_len: int,
    adv_len: int,
    vocab_size: int,
    edit_mask: bool,
    runs: int,
    show_summary: bool,
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for this audit script.")

    print("Running adversary CUDA memory audit")
    print(f"Project root: {ROOT}")
    print(f"Using device: {device}")
    print(query_nvidia_smi())

    cuda.empty_cache()
    cuda.reset_peak_memory_stats(device)

    adversary = CNNAttacker()
    defender = Defender()

    adversary.to(device)
    defender.to(device)

    x_mal, mask = build_dummy_batch(batch_size, seq_len, adv_len, vocab_size, edit_mask, device)

    print_cuda_stats("Before forward", device, show_summary=False)

    for run_index in range(1, runs + 1):
        print(f"\n=== Run {run_index}/{runs} ===")
        cuda.reset_peak_memory_stats(device)
        cuda.empty_cache()

        adversary.zero_grad()
        adversary.train()
        defender.eval()

        sync_cuda(device)
        pre_forward = cuda_stats(device)

        adv_logits = adversary(x_mal, mask)
        sync_cuda(device)
        after_forward = cuda_stats(device)

        dist = torch.distributions.Categorical(logits=adv_logits)
        adv_bytes = dist.sample()
        log_probs = dist.log_prob(adv_bytes)

        x_adv = adversary._apply_adv_bytes(x_mal, adv_bytes)
        sync_cuda(device)
        after_apply = cuda_stats(device)

        with torch.no_grad():
            def_logits = defender(x_adv)
            mal_probs = torch.sigmoid(def_logits)
            reward = 1.0 - mal_probs

        sequence_log_prob = log_probs.sum(dim=1)
        loss = -(reward.detach() * sequence_log_prob).mean()
        sync_cuda(device)
        after_loss = cuda_stats(device)

        loss.backward()
        sync_cuda(device)
        after_backward = cuda_stats(device)

        adversary.optim.step()
        sync_cuda(device)
        after_step = cuda_stats(device)

        print(f"Batch shape: x_mal={x_mal.shape} adv_bytes={adv_bytes.shape} x_adv={x_adv.shape}")
        print(f"Loss: {loss.item():.6f}")
        print(f"Reward mean: {reward.mean().item():.6f}")
        print(f"ESR estimate: {((mal_probs < 0.5).float().mean().item()):.6f}")

        print("\nMemory snapshot summary:")
        print(f"  before_forward.allocated:      {pre_forward['allocated']}")
        print(f"  after_forward.allocated:       {after_forward['allocated']}")
        print(f"  after_apply.allocated:         {after_apply['allocated']}")
        print(f"  after_loss.allocated:          {after_loss['allocated']}")
        print(f"  after_backward.allocated:      {after_backward['allocated']}")
        print(f"  after_step.allocated:          {after_step['allocated']}")
        print(f"  peak_allocated (run):         {fmt_bytes(cuda.max_memory_allocated(device))}")
        print(f"  peak_reserved (run):          {fmt_bytes(cuda.max_memory_reserved(device))}")

        if show_summary:
            print_cuda_stats("Final summary", device, show_summary=True)

        # Clean up the current run before the next iteration.
        del adv_logits, dist, adv_bytes, log_probs, x_adv, def_logits, mal_probs, reward, sequence_log_prob, loss
        torch.cuda.empty_cache()

    print("Audit complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit adversary CUDA memory usage for a single adversary batch pass."
    )
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for the adversary audit batch.")
    parser.add_argument("--seq-len", type=int, default=1024, help="Length of the malware input sequence.")
    parser.add_argument("--adv-len", type=int, default=256, help="Length of appended adversarial bytes.")
    parser.add_argument("--vocab-size", type=int, default=257, help="Byte vocabulary size used by the attacker.")
    parser.add_argument("--edit-mask", action="store_true", help="Enable a dummy editable mask for the adversary input.")
    parser.add_argument("--runs", type=int, default=1, help="Number of repeated audit runs to execute.")
    parser.add_argument("--summary", action="store_true", help="Print the full CUDA memory summary at the end of each run.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_adversary_audit(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        adv_len=args.adv_len,
        vocab_size=args.vocab_size,
        edit_mask=args.edit_mask,
        runs=args.runs,
        show_summary=args.summary,
    )
