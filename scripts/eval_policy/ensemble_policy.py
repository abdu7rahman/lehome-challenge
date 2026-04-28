import torch
import numpy as np
from typing import Dict, Any
from torchvision import transforms
from .base_policy import BasePolicy
from .registry import PolicyRegistry
from .lerobot_policy import LeRobotPolicy
from lehome.utils.logger import get_logger
from scripts.train_classifier import GarmentClassifier

logger = get_logger(__name__)

@PolicyRegistry.register("ensemble")
class EnsemblePolicy(BasePolicy):
    """
    Ensemble Policy that runs a Garment Classifier on the observation images
    and routes the request to either a specialist policy or a generalist policy.
    """
    def __init__(
        self,
        classifier_path: str,
        generalist_path: str,
        dataset_root: str,
        task_description: str,
        specialist_paths: Dict[str, str] = None,
        confidence_threshold: float = 0.85,
        device: str = "cuda"
    ):
        super().__init__()
        self.device = torch.device(device)
        self.confidence_threshold = confidence_threshold
        
        logger.info("Loading Classifier...")
        checkpoint = torch.load(classifier_path, map_location=self.device)
        self.class_map = checkpoint['class_map']
        self.idx_to_class = {v: k for k, v in self.class_map.items()}
        
        self.classifier = GarmentClassifier(num_classes=len(self.class_map))
        self.classifier.load_state_dict(checkpoint['model_state_dict'])
        self.classifier.to(self.device)
        self.classifier.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        
        logger.info("Loading Generalist Policy...")
        self.generalist_policy = LeRobotPolicy(
            policy_path=generalist_path,
            dataset_root=dataset_root,
            task_description=task_description,
            device=device
        )
        
        self.specialist_policies = {}
        if specialist_paths:
            for garment_type, policy_path in specialist_paths.items():
                logger.info(f"Loading Specialist Policy for {garment_type}...")
                self.specialist_policies[garment_type] = LeRobotPolicy(
                    policy_path=policy_path,
                    dataset_root=dataset_root,
                    task_description=task_description,
                    device=device
                )
                
        self.active_policy = self.generalist_policy
        self.active_policy_name = "generalist"

    def reset(self):
        self.generalist_policy.reset()
        for policy in self.specialist_policies.values():
            policy.reset()
        # By default we reset the state, and we should detect class again on first step
        self.detected_class = None

    def select_action(self, observation: Dict[str, np.ndarray]) -> np.ndarray:
        if self.detected_class is None:
            # We only run the classifier on the first frame of the episode to set the policy
            img = observation.get("observation.images.top_rgb")
            if img is not None:
                # numpy img [H, W, C] to tensor [C, H, W]
                img_tensor = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
                img_tensor = self.transform(img_tensor).unsqueeze(0).to(self.device)
                
                with torch.inference_mode():
                    logits = self.classifier(img_tensor)
                    probs = torch.softmax(logits, dim=1)[0]
                    confidence, pred_idx = torch.max(probs, dim=0)
                    
                pred_class = self.idx_to_class[pred_idx.item()]
                logger.info(f"Classifier predicted: {pred_class} with confidence {confidence.item():.2f}")
                
                if confidence.item() >= self.confidence_threshold and pred_class in self.specialist_policies:
                    self.active_policy = self.specialist_policies[pred_class]
                    self.active_policy_name = f"specialist_{pred_class}"
                    self.detected_class = pred_class
                    logger.info(f"Routing to {self.active_policy_name}")
                else:
                    self.active_policy = self.generalist_policy
                    self.active_policy_name = "generalist"
                    self.detected_class = "generalist"
                    logger.info("Routing to generalist policy")
            else:
                self.active_policy = self.generalist_policy
                self.active_policy_name = "generalist"
                self.detected_class = "generalist"
                logger.info("No top_rgb image found, falling back to generalist")

        return self.active_policy.select_action(observation)
