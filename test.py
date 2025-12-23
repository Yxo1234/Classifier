import argparse
import os
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms

from model import create_model

def is_distributed_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()

def is_main_process() -> bool:
    return not is_distributed_initialized() or dist.get_rank() == 0

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference script for lithology classification")
    parser.add_argument("--data-root", type=str, default="dataset", 
                        help="Root directory containing test data")
    parser.add_argument("--batch-size", type=int, default=32, 
                        help="Batch size for inference")
    parser.add_argument("--workers", type=int, default=4, 
                        help="Data loading workers")
    parser.add_argument("--image-size", type=int, default=224, 
                        help="Input image size")
    parser.add_argument("--distributed", action="store_true", 
                        help="Enable distributed inference")
    parser.add_argument("--local-rank", type=int, default=0, 
                        help="Local rank for distributed testing")
    return parser.parse_args()

def setup_distributed(args: argparse.Namespace) -> Tuple[bool, torch.device]:
    distributed = args.distributed or (torch.cuda.device_count() > 1)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    if distributed:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
    
    return distributed, device

def build_test_dataloader(args: argparse.Namespace, distributed: bool) -> Tuple[DataLoader, list]:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    test_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        normalize
    ])
    
    test_dir = Path(args.data_root) / "test"
    test_dataset = datasets.ImageFolder(test_dir, transform=test_transform)
    
    # 动态获取类别名和数量
    class_names = test_dataset.classes
    num_classes = len(class_names)
    
    test_sampler = DistributedSampler(test_dataset) if distributed else None
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        sampler=test_sampler,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available()
    )
    return test_loader, class_names, num_classes

def evaluate(model: nn.Module, data_loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            
            outputs = model(images)
            loss = nn.CrossEntropyLoss()(outputs, targets)
            
            total_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            total_correct += (preds == targets).sum().item()
            total_samples += targets.size(0)
    
    avg_loss = total_loss / max(total_samples, 1)
    accuracy = total_correct / max(total_samples, 1)
    return avg_loss, accuracy

def main():
    # 硬编码参数
    CHECKPOINT_PATH = "epoch_002.pth"  # 检查点路径
    OUTPUT_FILE = "test_results.txt"  # 结果输出文件
    
    args = parse_args()
    distributed, device = setup_distributed(args)
    
    # 构建数据加载器
    test_loader, class_names, num_classes = build_test_dataloader(args, distributed)
    
    # 初始化模型
    model = create_model(num_classes=num_classes).to(device)
    
    # 加载检查点
    if not os.path.exists(CHECKPOINT_PATH):
        raise FileNotFoundError(f"Checkpoint {CHECKPOINT_PATH} not found!")
    
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    
    if distributed:
        model = DDP(model, device_ids=[args.local_rank])
    
    # 执行推理
    test_loss, test_acc = evaluate(model, test_loader, device)
    
    # 保存结果
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "classes": class_names,
        "num_classes": num_classes,
        "test_loss": f"{test_loss:.4f}",
        "test_acc": f"{test_acc:.4f}"
    }
    
    if is_main_process():
        with open(OUTPUT_FILE, "w") as f:
            f.write("Test Results:\n")
            f.write("="*30 + "\n")
            f.write(f"Classes: {results['classes']}\n")
            f.write(f"Number of Classes: {results['num_classes']}\n")
            f.write(f"Test Loss: {results['test_loss']}\n")
            f.write(f"Test Accuracy: {results['test_acc']}\n")
        
        print(f"Results saved to {OUTPUT_FILE}")
        print(f"[Final Result] Loss: {test_loss:.4f}, Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()