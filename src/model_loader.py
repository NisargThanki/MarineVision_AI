
from pathlib import Path

import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# Project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent

CHECKPOINT_PATH = (
    PROJECT_DIR
    / "models"
    / "fresh_22_epochs"
    / "marinevision_best_val_loss.pth"
)


def load_marine_model():

    # Select GPU if available, otherwise use CPU
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {CHECKPOINT_PATH}"
        )

    # Load the saved checkpoint
    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
        weights_only=False
    )

    # Restore the original class mapping
    class_mapping = checkpoint["class_mapping"]

    num_classes = len(class_mapping) + 1

    # Recreate Faster R-CNN architecture
    model = fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None
    )

    # Configure the prediction layer
    in_features = (
        model.roi_heads.box_predictor.cls_score.in_features
    )

    model.roi_heads.box_predictor = FastRCNNPredictor(
        in_features,
        num_classes
    )

    # Restore image-size settings
    if "min_size" in checkpoint:
        model.transform.min_size = checkpoint["min_size"]

    if "max_size" in checkpoint:
        model.transform.max_size = checkpoint["max_size"]

    # Load trained weights
    model.load_state_dict(checkpoint["model_state_dict"])

    # Move model to selected device
    model.to(device)

    # Set inference mode
    model.eval()

    # Convert class IDs back to category names
    idx_to_class = {
        label: name
        for name, label in class_mapping.items()
    }

    return model, device, idx_to_class
