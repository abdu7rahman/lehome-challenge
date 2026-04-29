import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm


class GarmentClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)

    def forward(self, x):
        return self.model(x)


def parse_labeled_roots(dataset_roots: str):
    """
    Expected format:
    top_long=Datasets/example/top_long_merged,top_short=Datasets/example/top_short_merged
    """
    pairs = {}

    for item in dataset_roots.split(","):
        if "=" not in item:
            raise ValueError(f"Invalid dataset root entry: {item}")

        label, path = item.split("=", 1)
        pairs[label.strip()] = path.strip()

    return pairs


class FrameDataset(torch.utils.data.Dataset):
    def __init__(self, datasets, labels, camera_key="observation.images.top_rgb"):
        self.datasets = datasets
        self.labels = labels
        self.camera_key = camera_key
        self.samples = []

        for dataset_idx, ds in enumerate(datasets):
            for frame_idx in range(len(ds)):
                self.samples.append((dataset_idx, frame_idx, labels[dataset_idx]))

        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        dataset_idx, frame_idx, label = self.samples[idx]
        item = self.datasets[dataset_idx][frame_idx]

        img = item[self.camera_key]
        img = self.transform(img)

        return img, label


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    class_map = {
        "top_long": 0,
        "top_short": 1,
        "pant_long": 2,
        "pant_short": 3,
    }

    labeled_roots = parse_labeled_roots(args.dataset_roots)

    datasets = []
    labels = []

    for garment_class, root in labeled_roots.items():
        if garment_class not in class_map:
            raise ValueError(
                f"Unknown garment class '{garment_class}'. "
                f"Expected one of: {list(class_map.keys())}"
            )

        ds = LeRobotDataset(
            repo_id=args.repo_id,
            root=root,
            episodes=None,
        )

        print(f"Loaded {garment_class}: {len(ds)} frames from {root}")

        datasets.append(ds)
        labels.append(class_map[garment_class])

    total_frames = sum(len(ds) for ds in datasets)
    print(f"Total classifier frames: {total_frames}")

    train_dataset = FrameDataset(
        datasets=datasets,
        labels=labels,
        camera_key=args.camera,
    )

    dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = GarmentClassifier(num_classes=len(class_map)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    os.makedirs(args.output_dir, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()

        running_loss = 0.0
        correct = 0
        total = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}")

        for imgs, labels_batch in pbar:
            imgs = imgs.to(device)
            labels_batch = labels_batch.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels_batch.size(0)
            correct += (predicted == labels_batch).sum().item()

            pbar.set_postfix({
                "loss": loss.item(),
                "acc": 100 * correct / total,
            })

        avg_loss = running_loss / len(dataloader)
        avg_acc = 100 * correct / total

        print(
            f"Epoch {epoch + 1} Summary: "
            f"Loss={avg_loss:.4f}, Accuracy={avg_acc:.2f}%"
        )

        save_path = os.path.join(args.output_dir, f"classifier_ep{epoch + 1}.pt")

        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
            "accuracy": avg_acc,
            "class_map": class_map,
            "camera": args.camera,
        }, save_path)

        print(f"Saved checkpoint: {save_path}")

    print(f"Training complete. Models saved to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Garment Classifier")

    parser.add_argument(
        "--dataset_roots",
        type=str,
        required=True,
        help=(
            "Comma-separated labeled dataset roots, e.g. "
            "top_long=Datasets/example/top_long_merged,"
            "top_short=Datasets/example/top_short_merged,"
            "pant_long=Datasets/example/pant_long_merged,"
            "pant_short=Datasets/example/pant_short_merged"
        ),
    )

    parser.add_argument(
        "--repo_id",
        type=str,
        default="repo_act_all",
        help="LeRobot repo_id",
    )

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output_dir", type=str, default="outputs/classifier")
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument(
        "--camera",
        type=str,
        default="observation.images.top_rgb",
        help="Camera key to use for classification",
    )

    args = parser.parse_args()
    train(args)
