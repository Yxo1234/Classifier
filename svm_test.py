import argparse
import pickle
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser("SVM 测试集评估")
    parser.add_argument("--data-root", type=str, default="dataset")
    parser.add_argument("--model", type=str, default="svm_model.pkl")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()

def main():
    args = parse_args()

    # ---------- 加载模型 ----------
    with open(args.model, "rb") as f:
        data = pickle.load(f)

    clf = data["model"]
    scaler = data["scaler"]
    classes = data["classes"]

    print("📦 已加载模型，类别：", classes)

    # ---------- 数据 ----------
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))
    ])

    test_dir = Path(args.data_root) / "test"
    test_set = datasets.ImageFolder(root=test_dir, transform=transform)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

    print("🧪 测试集类别:", test_set.classes)
    print(f"🧪 测试集样本数: {len(test_set)}")

    # ---------- 推理 ----------
    all_preds = []
    all_gts = []

    for imgs, labels in tqdm(test_loader, desc="Testing"):
        X = imgs.numpy()
        y = labels.numpy()
        Xs = scaler.transform(X)
        preds = clf.predict(Xs)

        all_preds.append(preds)
        all_gts.append(y)

    all_preds = np.concatenate(all_preds)
    all_gts = np.concatenate(all_gts)

    # ---------- 指标 ----------
    acc = accuracy_score(all_gts, all_preds)
    print("\n✅ Test Accuracy:", f"{acc * 100:.2f}%")

    print("\n📊 分类报告：")
    print(classification_report(
        all_gts,
        all_preds,
        labels=list(range(len(classes))),
        target_names=classes
    ))

if __name__ == "__main__":
    main()
