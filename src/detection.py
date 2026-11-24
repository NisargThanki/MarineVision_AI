
import torch

from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms import functional as F
from torchvision.ops import nms


def detect_objects(
    image,
    model,
    device,
    idx_to_class,
    confidence_threshold=0.50,
    iou_threshold=0.30
):

    # 1. Prepare image
    image = image.convert("RGB")
    image_tensor = F.to_tensor(image).to(device)

    # 2. Model prediction
    model.eval()

    with torch.no_grad():
        prediction = model([image_tensor])[0]

    boxes = prediction["boxes"].detach().cpu()
    scores = prediction["scores"].detach().cpu()
    labels = prediction["labels"].detach().cpu()

    # 3. Confidence threshold
    valid = scores >= confidence_threshold

    boxes = boxes[valid]
    scores = scores[valid]
    labels = labels[valid]

    # 4. Non-Maximum Suppression
    if len(boxes) > 0:
        keep = nms(
            boxes,
            scores,
            iou_threshold=iou_threshold
        )

        boxes = boxes[keep]
        scores = scores[keep]
        labels = labels[keep]

    # 5. Prepare annotated image
    annotated_image = image.copy()
    draw = ImageDraw.Draw(annotated_image)

    image_width, image_height = annotated_image.size

    try:
        font = ImageFont.truetype("arial.ttf", 18)
    except OSError:
        font = ImageFont.load_default()

    detections = []

    # Store previous label positions
    used_label_positions = []

    # 6. Draw bounding boxes and labels
    for label, score, box in zip(labels, scores, boxes):

        class_id = int(label.item())
        class_name = idx_to_class.get(class_id, "Unknown")
        confidence = float(score.item())

        x1, y1, x2, y2 = map(int, box.tolist())

        # Keep coordinates inside image
        x1 = max(0, min(x1, image_width - 1))
        y1 = max(0, min(y1, image_height - 1))
        x2 = max(0, min(x2, image_width - 1))
        y2 = max(0, min(y2, image_height - 1))

        # Draw bounding box
        draw.rectangle(
            (x1, y1, x2, y2),
            outline="#00E676",
            width=4
        )

        # Create label
        text = f"{class_name} | {confidence:.2f}"

        text_bbox = draw.textbbox(
            (0, 0),
            text,
            font=font
        )

        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        padding = 6

        label_width = text_width + padding * 2
        label_height = text_height + padding * 2

        # Keep label inside image width
        label_x = min(
            x1,
            max(0, image_width - label_width)
        )

        # Possible label positions
        candidate_positions = [
            y1 - label_height,       # Above box
            y2,                      # Below box
            y1,                      # Inside top
            y2 - label_height        # Inside bottom
        ]

        label_y = None

        # Find a position that does not overlap previous labels
        for candidate_y in candidate_positions:

            candidate_y = max(
                0,
                min(candidate_y, image_height - label_height)
            )

            candidate_rect = (
                label_x,
                candidate_y,
                label_x + label_width,
                candidate_y + label_height
            )

            overlaps = any(
                candidate_rect[0] < old[2]
                and candidate_rect[2] > old[0]
                and candidate_rect[1] < old[3]
                and candidate_rect[3] > old[1]
                for old in used_label_positions
            )

            if not overlaps:
                label_y = candidate_y
                break

        # Fallback if all positions are occupied
        if label_y is None:
            label_y = max(
                0,
                min(y1 - label_height, image_height - label_height)
            )

        # Save selected label position
        used_label_positions.append((
            label_x,
            label_y,
            label_x + label_width,
            label_y + label_height
        ))

        # Draw dark green label background
        draw.rectangle(
            (
                label_x,
                label_y,
                label_x + label_width,
                label_y + label_height
            ),
            fill="#003B35"
        )

        # Draw white label text
        draw.text(
            (
                label_x + padding,
                label_y + padding - text_bbox[1]
            ),
            text,
            fill="white",
            font=font
        )

        # Store detection details
        detections.append({
            "class_id": class_id,
            "class_name": class_name,
            "confidence": round(confidence, 4),
            "box": [x1, y1, x2, y2]
        })

    # 7. Return results
    return annotated_image, detections
