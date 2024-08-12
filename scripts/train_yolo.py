import argparse
from ultralytics import YOLO
import wandb
from sklearn.metrics import precision_recall_fscore_support

def evaluate_model(model, data_config, iou_threshold=0.1):
    # Load the dataset
    dataset = model.load_dataset(data_config, split='test')

    # Initialize metrics
    all_preds, all_labels = [], []

    # Iterate over the dataset and collect predictions and labels
    for img, label in dataset:
        results = model(img, iou_thres=iou_threshold)
        preds = results.pred[0].numpy()
        labels = label.numpy()

        all_preds.append(preds)
        all_labels.append(labels)

    # Compute precision, recall, and F1 score
    precision, recall, f1, _ = precision_recall_fscore_support(all_labels, all_preds, average='weighted')
    return precision, recall, f1

def train_yolo(model_weights, data_config, epochs=100, img_size=640, batch_size=16, devices=None, project="runs/train", name="exp", log_wandb=True):
    # Initialize W&B
    if log_wandb:
        wandb.init(project=project, name=name)
        wandb.config.update({
            "model_weights": model_weights,
            "data_config": data_config,
            "epochs": epochs,
            "img_size": img_size,
            "batch_size": batch_size,
            "devices": devices,
        })

    # Load the pretrained model
    model = YOLO(model_weights)

    # Train the model
    results = model.train(data=data_config, epochs=epochs, imgsz=img_size, batch=batch_size, device=devices, project=project, name=name)

    # Log the results
    if log_wandb:
        wandb.finish()

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train YOLOv5 using a pretrained model.')
    parser.add_argument('--model_weights', type=str, default='yolov5s.pt', help='Path to the pretrained model weights.')
    parser.add_argument('--data_config', type=str, required=True, help='Path to dataset YAML config file.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--img_size', type=int, default=640, help='Image size for training.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--devices', type=str, default=None, help='GPUs to use for training (e.g., "0", "0,1").')
    parser.add_argument('--project', type=str, default='runs/train', help='Project directory to save results.')
    parser.add_argument('--name', type=str, default='exp', help='Experiment name to save results.')
    parser.add_argument('--iou_threshold', type=float, default=0.1, help='IoU threshold for evaluation.')

    args = parser.parse_args()

    # Convert devices to list of integers if provided
    devices = [int(d) for d in args.devices.split(',')] if args.devices else None

    # Train the model
    train_yolo(
        model_weights=args.model_weights,
        data_config=args.data_config,
        epochs=args.epochs,
        img_size=args.img_size,
        batch_size=args.batch_size,
        devices=devices,
        project=args.project,
        name=args.name
    )

    # Load the trained model
    trained_model = YOLO(f"./{args.project}/{args.name}/weights/best.pt")

    # Evaluate the model
    precision, recall, f1 = evaluate_model(trained_model, args.data_config, iou_threshold=args.iou_threshold)
    
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")


