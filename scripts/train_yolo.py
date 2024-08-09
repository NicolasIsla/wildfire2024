import argparse
from ultralytics import YOLO

def train_yolo(model_weights, data_config, epochs=100, img_size=640, devices=None, project="runs/train", name="exp"):
    # Load the pretrained model
    model = YOLO(model_weights)  # Load a pretrained model

    # Train the model
    results = model.train(data=data_config, epochs=epochs, imgsz=img_size, device=devices, project=project, name=name)

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train YOLOv5 using a pretrained model.')
    parser.add_argument('--model_weights', type=str, default='yolov5s.pt', help='Path to the pretrained model weights.')
    parser.add_argument('--data_config', type=str, required=True, help='Path to dataset YAML config file.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--img_size', type=int, default=640, help='Image size for training.')
    parser.add_argument('--devices', type=str, default=None, help='GPUs to use for training (e.g., "0", "0,1").')
    parser.add_argument('--project', type=str, default='runs/train', help='Project directory to save results.')
    parser.add_argument('--name', type=str, default='exp', help='Experiment name to save results.')

    args = parser.parse_args()

    # Convert devices to list of integers if provided
    devices = [int(d) for d in args.devices.split(',')] if args.devices else None

    train_yolo(
        model_weights=args.model_weights,
        data_config=args.data_config,
        epochs=args.epochs,
        img_size=args.img_size,
        devices=devices,
        project=args.project,
        name=args.name
    )
