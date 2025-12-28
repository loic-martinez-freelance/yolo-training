import argparse
import os
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Test YOLO model from a specific experiment folder.")
    parser.add_argument("folder", help="The folder name containing the experiment (e.g., 'mydataset')")
    parser.add_argument("--image", default="test.png", help="Path to the image to test (default: test.png)")
    args = parser.parse_args()

    folder_name = args.folder
    model_path = os.path.join(folder_name, "experiment", "weights", "best.pt")

    if not os.path.exists(model_path):
        print(f"Error: Could not find model weights at {model_path}")
        return

    model = YOLO(model_path)
    
    if not os.path.exists(args.image):
        print(f"Error: Could not find image at {args.image}")
        return

    results = model(args.image, conf=0.6)
    
    for result in results:
        result.show()

if __name__ == "__main__":
    main()