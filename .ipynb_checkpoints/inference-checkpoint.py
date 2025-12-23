import argparse
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from torchvision import transforms

from model import create_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inference script for lithology classifier.")
    parser.add_argument("image", type=str, help="Path to the image to classify.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/epoch_001.pth",
        help="Checkpoint file saved during training.",
    )
    parser.add_argument("--image-size", type=int, default=224, help="Resize image to this square size.")
    parser.add_argument("--device", type=str, default=None, help="Device to run on (e.g., cuda:0 or cpu).")
    return parser.parse_args()


def load_checkpoint(ckpt_path: Path, device: torch.device) -> Tuple[torch.nn.Module, List[str], int]:
    checkpoint = torch.load(ckpt_path, map_location=device)
    classes = checkpoint.get("classes")
    if not classes:
        raise ValueError("Checkpoint is missing class names under key 'classes'.")

    model = create_model(num_classes=len(classes))
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    epoch = int(checkpoint.get("epoch", -1))
    return model, classes, epoch


def build_transform(image_size: int) -> transforms.Compose:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            normalize,
        ]
    )


@torch.no_grad()
def predict(image_path: Path, model: torch.nn.Module, device: torch.device, classes: List[str], image_size: int):
    transform = build_transform(image_size)
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    outputs = model(tensor)
    probs = torch.softmax(outputs, dim=1)
    prob, pred = torch.max(probs, dim=1)

    return classes[pred.item()], prob.item()


def main() -> None:
    args = parse_args()
    image_path = Path(args.image)
    ckpt_path = Path(args.checkpoint)

    if not image_path.is_file():
        raise FileNotFoundError(f"Image not found: {image_path}")
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, classes, epoch = load_checkpoint(ckpt_path, device)
    class_name, confidence = predict(image_path, model, device, classes, args.image_size)

    epoch_str = f"(from epoch {epoch})" if epoch >= 0 else ""
    print(f"Predicted: {class_name} with confidence {confidence:.4f} {epoch_str}".strip())


if __name__ == "__main__":
    main()
