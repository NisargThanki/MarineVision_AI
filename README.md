
# MarineVision AI 🌊

An AI-powered underwater marine debris detection and analysis application using Faster R-CNN with a ResNet50 FPN backbone and PyTorch.

## Project Overview

MarineVision AI is designed to detect and identify marine debris in underwater images. It combines deep learning-based object detection with AI-assisted image analysis through a Streamlit web application.

## Technology Stack

- Python
- PyTorch
- Torchvision
- Faster R-CNN
- ResNet50 FPN
- Streamlit
- Pandas
- Pillow

## Dataset

This project uses the SeaClear Marine Debris Dataset.

- 8,610 images
- 31,555 annotations
- 40 object categories
- COCO annotation format

The original dataset is not included in this repository.

## Project Structure

```text
MarineVision_AI/
├── app.py
├── src/
│   ├── ai_analysis.py
│   ├── dataset_gallery.py
│   ├── detection.py
│   └── model_loader.py
├── notebooks/
├── models/
├── results/
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd MarineVision_AI
```

Create a virtual environment:

```bash
python -m venv marinevision_env
```

Activate it on Windows:

```powershell
.\marinevision_env\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Install a compatible PyTorch and Torchvision build for your CPU or CUDA environment if needed.

## Trained Model

The trained model checkpoint is not included because of its large file size.

Download the checkpoint separately and place it at:

```text
models/fresh_22_epochs/marinevision_best_val_loss.pth
```

## Run the Application

```bash
streamlit run app.py
```

## Project Results

Evaluation metrics and experiment outputs are available in the `results/` directory.

## License

Dataset and third-party image usage remain subject to their respective licenses and terms.
