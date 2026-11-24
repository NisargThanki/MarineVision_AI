
import json
from pathlib import Path


# Project directories
PROJECT_DIR = Path(__file__).resolve().parent.parent

DATASET_DIR = PROJECT_DIR / "dataset"

ANNOTATION_PATH = DATASET_DIR / "dataset.json"

SPLIT_PATH = PROJECT_DIR / "results" / "dataset_split.json"


def load_test_images():

    # Load COCO annotations
    with open(ANNOTATION_PATH, "r", encoding="utf-8") as file:
        coco_data = json.load(file)

    # Load dataset split
    with open(SPLIT_PATH, "r", encoding="utf-8") as file:
        split_data = json.load(file)

    # Create image ID mapping
    image_data = {
        item["id"]: item
        for item in coco_data["images"]
    }

    # Find all original JPG images
    image_paths = {
        path.name: path
        for path in DATASET_DIR.rglob("*.jpg")
    }

    # Build test image records
    test_images = []

    for image_id in split_data["test"]:

        if image_id not in image_data:
            continue

        image_info = image_data[image_id]

        file_name = image_info["file_name"]

        image_path = image_paths.get(file_name)

        if image_path is None:
            continue

        test_images.append({
            "image_id": image_id,
            "file_name": file_name,
            "image_path": image_path
        })

    return test_images
