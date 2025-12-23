import argparse
from pathlib import Path
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import StandardScaler
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import pickle


def parse_args():
    parser = argparse.ArgumentParser("使用增量SVM进行图像分类训练")
    parser.add_argument("--data-root", type=str, default="dataset", help="数据集根目录")
    parser.add_argument("--image-size", type=int, default=64, help="输入图像尺寸(需与测试一致)")
    parser.add_argument("--batch-size", type=int, default=32, help="batch size")
    return parser.parse_args()


def main():
    args = parse_args()

    # -------- 图像预处理 --------
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))  # 展平成一维
    ])

    # -------- 只读取 train --------
    train_dir = Path(args.data_root) / "train"
    dataset = datasets.ImageFolder(root=train_dir, transform=transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    print("📂 训练数据路径:", train_dir)
    print("🏷️  训练类别:", dataset.classes)

    num_classes = len(dataset.classes)

    # -------- 模型与归一化 --------
    clf = SGDClassifier(loss="hinge", max_iter=1, tol=None)
    scaler = StandardScaler()

    print("👉 开始分批训练 SVM ...")

    first_batch = True
    for imgs, labels in tqdm(loader):
        X = imgs.numpy()
        y = labels.numpy()

        if first_batch:
            scaler.partial_fit(X)
            Xs = scaler.transform(X)
            clf.partial_fit(Xs, y, classes=np.arange(num_classes))
            first_batch = False
        else:
            scaler.partial_fit(X)
            Xs = scaler.transform(X)
            clf.partial_fit(Xs, y)

    print("✅ 训练完成")

    # -------- 保存模型 --------
    with open("svm_model.pkl", "wb") as f:
        pickle.dump({
            "model": clf,
            "scaler": scaler,
            "classes": dataset.classes
        }, f)

    print("📦 模型已保存为 svm_model.pkl")


if __name__ == "__main__":
    main()
