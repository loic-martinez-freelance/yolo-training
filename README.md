# YOLOv11 Custom Dataset Manager

This project provides utility scripts to simplify training and testing YOLOv11 models on custom datasets. It is designed to work with specific folder structures, facilitating the management of multiple computer vision projects.

## ✅ Prerequisites

- Python 3.8+
- [Ultralytics](https://github.com/ultralytics/ultralytics)

## 📦 Installation

1. Clone this repository or download the files.
2. Install the necessary dependencies:

```bash
pip install -r requirements.txt
```

## 🚀 Usage

### Expected folder structure

The project expects your datasets to be organized in folders containing the following structure (typically from a Roboflow export):

```text
datasets/
└── <dataset_folder_name>/
    ├── <set_name>/              # Folder containing the dataset (e.g., roboflow)
    │   └── data.yaml            # Dataset configuration
    ├── train_params.yaml        # (Optional) Custom training parameters
    └── ... (images, labels)
```

### Training (`train.py`)

This script starts training a YOLOv11n model on the specified dataset.

**Command:**

```bash
python train.py <dataset_folder_name> <set_name>
```

**Example:**

```bash
python train.py myproject roboflow
```

**Features:**

- Automatically uses `yolo11n.pt` as the base model.
- Looks for configuration in `datasets/<dataset_folder_name>/<set_name>/data.yaml`.
- Default training parameters are:
  - Epochs: 100
  - Image size: 640
  - Batch size: 8
  - Patience: 10
  - Device: mps (Apple Silicon) - _Can be modified in `train.py` if necessary._
- **Custom configuration:** You can place a `train_params.yaml` file in your dataset folder to override default parameters (e.g., `epochs`, `batch`, etc.).
- Saves the best trained model under `datasets/<dataset_folder_name>/<dataset_folder_name>.pt`.

### Testing (`test.py`)

This script allows you to quickly test a trained model on a static image.

**Command:**

```bash
python test.py datasets/<experiment_folder_name> [--image path/to/image.png]
```

**Example:**

```bash
python test.py datasets/myproject --image test.png
```

**Features:**

- Loads the model from `datasets/<folder>/experiment/weights/best.pt`.
- Displays the detection result on the screen.
- Default confidence threshold: 0.6.

## 🛠️ Configuration

The `requirements.txt` file contains the required libraries.

```text
ultralytics
# roboflow  # uncomment if you use the Roboflow python API
```
