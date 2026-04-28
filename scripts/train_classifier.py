import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, models
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from tqdm import tqdm
from pathlib import Path
import json

class GarmentClassifier(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        # Use a pretrained ResNet18 model
        self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        # Change the last layer
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, num_classes)
        
    def forward(self, x):
        return self.model(x)

def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load the merged dataset using LeRobotDataset
    dataset = LeRobotDataset(
        repo_id=args.repo_id,
        root=args.dataset_root,
        episodes=None # Load all
    )
    print(f"Dataset loaded. Total frames: {len(dataset)}")
    
    # Garment classes map (example mapping based on standard challenge)
    class_map = {
        "top_long": 0,
        "top_short": 1,
        "pant_long": 2,
        "pant_short": 3,
    }
    
    # Load garment info from dataset meta to get labels per episode
    meta_path = Path(args.dataset_root) / "meta" / "garment_info.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}. Cannot get labels.")
        
    with open(meta_path, 'r') as f:
        garment_info = json.load(f)
        
    # Build episode to class_idx mapping
    episode_to_class = {}
    for garment_name, episodes in garment_info.items():
        if garment_name not in class_map:
            print(f"Warning: Unknown garment class {garment_name}")
            continue
        for ep_id in episodes.keys():
            episode_to_class[int(ep_id)] = class_map[garment_name]
            
    # We don't need the whole sequence for a classifier, just random frames
    # Create a custom dataset that grabs frames and their labels
    class FrameDataset(torch.utils.data.Dataset):
        def __init__(self, lerobot_dataset, episode_to_class, camera_key="observation.images.top_rgb"):
            self.dataset = lerobot_dataset
            self.episode_to_class = episode_to_class
            self.camera_key = camera_key
            
            # Data augmentation for training
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
            ])
            
        def __len__(self):
            return len(self.dataset)
            
        def __getitem__(self, idx):
            item = self.dataset[idx]
            ep_idx = item['episode_index'].item()
            img = item[self.camera_key] # Usually FloatTensor [C, H, W] in [0, 1]
            
            label = self.episode_to_class[ep_idx]
            
            # Apply transforms
            img = self.transform(img)
            return img, label

    train_dataset = FrameDataset(dataset, episode_to_class, camera_key=args.camera)
    dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)
    
    model = GarmentClassifier(num_classes=len(class_map)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # Save directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for imgs, labels in pbar:
            imgs, labels = imgs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({'loss': loss.item(), 'acc': 100 * correct / total})
            
        print(f"Epoch {epoch+1} Summary: Loss={running_loss/len(dataloader):.4f}, Accuracy={100 * correct / total:.2f}%")
        
        # Save checkpoint
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': running_loss/len(dataloader),
            'class_map': class_map
        }, os.path.join(args.output_dir, f'classifier_ep{epoch+1}.pt'))
        
    print(f"Training complete. Models saved to {args.output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train Garment Classifier")
    parser.add_argument('--dataset_root', type=str, required=True, help="Path to all_garments_merged dataset")
    parser.add_argument('--repo_id', type=str, default="repo_act_all", help="LeRobot repo_id")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--output_dir', type=str, default="outputs/classifier")
    parser.add_argument('--camera', type=str, default="observation.images.top_rgb", help="Camera key to use for classification")
    
    args = parser.parse_args()
    train(args)
