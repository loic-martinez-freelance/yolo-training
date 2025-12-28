import argparse
import os
import shutil
import yaml
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Train YOLO model on a specific dataset folder.")
    parser.add_argument("folder", help="The folder name containing the dataset (e.g., 'mydataset')")
    parser.add_argument("set", help="The set name containing the dataset (e.g., 'roboflow')")
    args = parser.parse_args()

    folder_name = args.folder
    set_name = args.set
    project_name = os.path.basename(os.path.normpath(folder_name))
    datasets_path = os.path.join(os.getcwd(), "datasets")
    project_path = os.path.join(datasets_path, folder_name)
    data_path = os.path.join(project_path, set_name, "data.yaml")
    config_path = os.path.join(project_path, "train_params.yaml")

    if not os.path.exists(data_path):
        print(f"Error: Could not find {data_path}")
        return

    # Default training arguments
    train_args = {
        "data": data_path,
        "epochs": 100,
        "imgsz": 640,
        "batch": 8,
        "patience": 10,
        "project": project_path,
        "name": "experiment",
        "device": "mps",
        "plots": True,
        "exist_ok": True
    }

    # Load custom custom arguments from dataset folder if available
    if os.path.exists(config_path):
        print(f"Loading custom training parameters from {config_path}")
        try:
            with open(config_path, 'r') as f:
                custom_args = yaml.safe_load(f)
                if custom_args:
                    train_args.update(custom_args)
                    print(f"Updated parameters: {list(custom_args.keys())}")
        except Exception as e:
            print(f"Warning: Error reading {config_path}: {e}")

    model = YOLO("yolo11n.pt")

    model.train(**train_args)

    save_dir = model.trainer.save_dir
    best_weights_path = os.path.join(save_dir, "weights", "best.pt")
    destination_path = os.path.join(project_path, f"{project_name}.pt")
    if os.path.exists(best_weights_path):
        shutil.copy(best_weights_path, destination_path)
        print(f"Model saved to {destination_path}")
    else:
        print(f"Error: Could not find best weights at {best_weights_path}")

if __name__ == "__main__":
    main()