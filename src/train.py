import os
import argparse
import time
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from rich.console import Console

from nvtx_helpers import nvtx_range
from model import TinyTransformerLM


console = Console()


class SyntheticTextDataset(Dataset):
    """
    Synthetic integer-token dataset. Produces (input_tokens, target_tokens) pairs.
    This keeps the demo self-contained and repeatable.
    """
    def __init__(self, length=10_000, seq_len=128, vocab_size=32_000, seed=42):
        super().__init__()
        self.length = length
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        rng = np.random.default_rng(seed)
        self._data = rng.integers(low=0, high=vocab_size, size=(length, seq_len), dtype=np.int32)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        x = torch.tensor(self._data[idx], dtype=torch.long)
        # For simplicity, predict the same tokens (teacher forcing style)
        y = x.clone()
        return x, y


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=128)
    ap.add_argument("--vocab-size", type=int, default=32000)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--nheads", type=int, default=8)
    ap.add_argument("--ffn-dim", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--dataset-size", type=int, default=10_000)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--amp", action="store_true", help="Enable mixed precision (AMP)")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--profile-sleep", type=float, default=0.0, help="Optional small sleep to make NVTX ranges clear")
    return ap.parse_args()


def build_model(args):
    model = TinyTransformerLM(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        nhead=args.nheads,
        num_layers=args.layers,
        dim_feedforward=args.ffn_dim,
        seq_len=args.seq_len,
        dropout=0.1,
    )
    return model


def main():
    args = parse_args()
    set_seed(123)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    console.print(f"[bold green]Using device:[/bold green] {device}")

    # Perf-friendly flags
    torch.backends.cudnn.benchmark = True

    # Data
    with nvtx_range("dataloader_setup"):
        dataset = SyntheticTextDataset(
            length=args.dataset_size,
            seq_len=args.seq_len,
            vocab_size=args.vocab_size,
            seed=123,
        )
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
        )

    # Model/opt
    model = build_model(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = GradScaler(enabled=args.amp)

    # Training
    model.train()
    iters_per_epoch = len(loader)
    total_steps = iters_per_epoch * args.epochs

    pbar = tqdm(total=total_steps, desc="training")

    for epoch in range(args.epochs):
        with nvtx_range(f"epoch_{epoch}"):
            for it, (x, y) in enumerate(loader):
                step0 = time.time()

                with nvtx_range("batch_to_device"):
                    x = x.to(device, non_blocking=True)
                    y = y.to(device, non_blocking=True)

                if args.profile_sleep > 0:
                    time.sleep(args.profile_sleep)  # help you see ranges in Nsight Systems

                # Forward
                with nvtx_range("forward"):
                    with autocast(enabled=args.amp):
                        logits = model(x)
                        loss = model.loss(logits, y)

                # Backward
                with nvtx_range("backward"):
                    optimizer.zero_grad(set_to_none=True)
                    if args.amp:
                        scaler.scale(loss).backward()
                    else:
                        loss.backward()

                # Optimizer step
                with nvtx_range("optimizer_step"):
                    if args.amp:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()

                if it % 50 == 0:
                    console.print(f"epoch {epoch} iter {it}/{iters_per_epoch} | loss {loss.item():.4f}")

                pbar.update(1)

    pbar.close()
    console.print("[bold cyan]Done.[/bold cyan]")


if __name__ == "__main__":
    main()