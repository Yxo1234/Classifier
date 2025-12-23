import argparse
import csv
import os
import time
from pathlib import Path
from typing import Iterable, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms

from model import create_model


def is_distributed_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def is_main_process() -> bool:
    return not is_distributed_initialized() or dist.get_rank() == 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CNN to classify lithology images.")
    parser.add_argument("--data-root", type=str, default="dataset", help="Root directory containing train/val/test.")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--workers", type=int, default=4, help="Number of data loading workers.")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for optimizer.")
    parser.add_argument("--val-interval", type=int, default=100, help="Validate every N training steps.")
    parser.add_argument("--max-val-batches", type=int, default=0, help="Limit validation to first N batches (0 = full).")
    parser.add_argument("--image-size", type=int, default=224, help="Input image size (square).")
    parser.add_argument("--log-csv", type=str, default="training_log.csv", help="Where to store training metrics CSV.")
    parser.add_argument("--ckpt-dir", type=str, default="homework/ckpt", help="Directory to save checkpoints each epoch.")
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Use DistributedDataParallel (torchrun launch) when more than one GPU is available.",
    )
    parser.add_argument(
        "--local-rank",
        type=int,
        default=0,
        help="Local rank passed by torchrun (will also read LOCAL_RANK env).",
    )
    args = parser.parse_args()
    args.local_rank = int(os.environ.get("LOCAL_RANK", args.local_rank))
    return args


def setup_distributed(args: argparse.Namespace) -> Tuple[bool, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = (args.distributed or world_size > 1) and dist.is_available()
    if distributed:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        dist.init_process_group(backend=backend)
        if torch.cuda.is_available():
            torch.cuda.set_device(args.local_rank)
            device = torch.device("cuda", args.local_rank)
        else:
            device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return distributed, device


def unwrap_model(model: nn.Module) -> nn.Module:
    if isinstance(model, (nn.DataParallel, DDP)):
        return model.module
    return model


def save_checkpoint(
    model: nn.Module, optimizer: optim.Optimizer, epoch: int, ckpt_dir: Path, class_names: Iterable[str]
) -> Path:
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model_to_save = unwrap_model(model)
    state = {
        "epoch": epoch,
        "model_state": model_to_save.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "classes": list(class_names),
    }
    ckpt_path = ckpt_dir / f"epoch_{epoch:03d}.pth"
    torch.save(state, ckpt_path)
    return ckpt_path


def build_dataloaders(
    args: argparse.Namespace, distributed: bool
) -> Tuple[
    DataLoader,
    DataLoader,
    Optional[DistributedSampler],
    Optional[DistributedSampler],
    Iterable[str],
]:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.RandomAffine(degrees=20, translate=(0.1, 0.1), shear=5.7, scale=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )

    train_dir = Path(args.data_root) / "train"
    val_dir = Path(args.data_root) / "val"

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(val_dir, transform=val_transform)

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if distributed else None
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=not distributed,
        sampler=train_sampler,
        num_workers=args.workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=args.workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, train_sampler, val_sampler, train_dataset.classes


def log_metrics(log_path: Path, row: dict) -> None:
    write_header = not log_path.exists()
    with log_path.open("a", newline="") as f:
        fieldnames = ["timestamp", "epoch", "step", "train_loss", "val_loss", "val_acc", "lr"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    max_batches: int = 0,
    distributed: bool = False,
) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_idx, (images, targets) in enumerate(data_loader, start=1):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        total_correct += (preds == targets).sum().item()
        total_samples += targets.size(0)

        if max_batches and batch_idx >= max_batches:
            break

    if distributed and is_distributed_initialized():
        totals = torch.tensor([total_loss, total_correct, total_samples], device=device)
        dist.all_reduce(totals, op=dist.ReduceOp.SUM)
        total_loss, total_correct, total_samples = totals.tolist()

    avg_loss = total_loss / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)
    return avg_loss, accuracy


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    train_sampler: Optional[DistributedSampler],
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    global_step: int,
    args: argparse.Namespace,
    log_path: Path,
    distributed: bool,
) -> int:
    model.train()
    running_loss = 0.0
    evaluated_this_epoch = False
    last_batch_idx = 0

    if train_sampler is not None:
        train_sampler.set_epoch(epoch)

    for batch_idx, (images, targets) in enumerate(train_loader, start=1):
        last_batch_idx = batch_idx
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        global_step += 1

        if global_step % args.val_interval == 0:
            avg_train_loss = running_loss / batch_idx
            val_loss, val_acc = evaluate(
                model=model,
                data_loader=val_loader,
                device=device,
                criterion=criterion,
                max_batches=args.max_val_batches,
                distributed=distributed,
            )
            evaluated_this_epoch = True
            current_lr = optimizer.param_groups[0]["lr"]
            if is_main_process():
                log_metrics(
                    log_path,
                    {
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "epoch": epoch,
                        "step": global_step,
                        "train_loss": f"{avg_train_loss:.4f}",
                        "val_loss": f"{val_loss:.4f}",
                        "val_acc": f"{val_acc:.4f}",
                        "lr": f"{current_lr:.6f}",
                    },
                )

                print(
                    f"[Epoch {epoch} Step {global_step}] "
                    f"train_loss={avg_train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
                )

    if not evaluated_this_epoch:
        avg_train_loss = running_loss / max(last_batch_idx, 1)
        val_loss, val_acc = evaluate(
            model=model,
            data_loader=val_loader,
            device=device,
            criterion=criterion,
            max_batches=args.max_val_batches,
            distributed=distributed,
        )
        current_lr = optimizer.param_groups[0]["lr"]
        if is_main_process():
            log_metrics(
                log_path,
                {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "epoch": epoch,
                    "step": global_step,
                    "train_loss": f"{avg_train_loss:.4f}",
                    "val_loss": f"{val_loss:.4f}",
                    "val_acc": f"{val_acc:.4f}",
                    "lr": f"{current_lr:.6f}",
                },
            )
            print(
                f"[Epoch {epoch} End] train_loss={avg_train_loss:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

    return global_step


def main() -> None:
    args = parse_args()
    distributed, device = setup_distributed(args)
    torch.backends.cudnn.benchmark = torch.cuda.is_available()

    train_loader, val_loader, train_sampler, val_sampler, class_names = build_dataloaders(
        args, distributed=distributed
    )
    num_classes = len(class_names)

    model = create_model(num_classes=num_classes)
    model = model.to(device)
    if distributed:
        model = DDP(model, device_ids=[device.index] if device.type == "cuda" else None)
    elif torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs with DataParallel.")
        model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    log_path = Path(args.log_csv)
    ckpt_dir = Path(args.ckpt_dir)
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        global_step = train_one_epoch(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            train_sampler=train_sampler,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            global_step=global_step,
            args=args,
            log_path=log_path,
            distributed=distributed,
        )

        if is_main_process():
            ckpt_path = save_checkpoint(
                model=model, optimizer=optimizer, epoch=epoch, ckpt_dir=ckpt_dir, class_names=class_names
            )
            print(f"Epoch {epoch} checkpoint saved to {ckpt_path}")

    if is_main_process():
        print("Training finished. Metrics saved to", log_path.resolve())

    if distributed and is_distributed_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
