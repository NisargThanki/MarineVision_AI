# MarineVision AI 🌊

An AI-powered underwater marine debris detection and analysis application built using Faster R-CNN with a ResNet50 FPN backbone and PyTorch.

## Project Overview

MarineVision AI is designed to detect and identify marine debris in underwater images. It combines deep learning-based object detection with AI-assisted image analysis through an interactive Streamlit web application.

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

This project uses the **SeaClear Marine Debris Dataset**.

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
│   └── MarineVision_AI_Fresh_22_Epoch_Final.ipynb
├── models/
│   └── fresh_22_epochs/
├── results/
├── requirements.txt
├── .gitignore
└── README.md
```

## Installation

Clone the repository:

```bash
git clone https://github.com/NisargThanki/MarineVision_AI.git
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

Install a compatible PyTorch and Torchvision build for your CPU or CUDA environment if required.

## Trained Model Checkpoints

The trained Faster R-CNN checkpoints are distributed separately because of their large file sizes.

**Download the trained models:**

[MarineVision AI – Trained Model Checkpoints (Release v1.0.0)](https://github.com/NisargThanki/MarineVision_AI/releases/tag/1.0.0)

Download `fresh_22_epochs.rar` from the release assets and extract it using WinRAR or 7-Zip.

Place the extracted files inside:

```text
models/
└── fresh_22_epochs/
    ├── marinevision_best_val_loss.pth
    ├── marinevision_latest.pth
    └── training_history.csv
```

The application uses `marinevision_best_val_loss.pth` by default.

## Run the Application

After installing dependencies and placing the trained checkpoint in the correct location, run:

```bash
streamlit run app.py
```

## Project Results

Evaluation metrics, experimental results, dataset analysis and sample prediction outputs are available in the `results/` directory.

The final training notebook and its PDF export are available in the `notebooks/` directory.

## License

Dataset and third-party image usage remain subject to their respective licenses and terms.