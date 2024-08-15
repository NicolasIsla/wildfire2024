import argparse
from ultralytics import YOLO
import wandb
import os
import numpy as np
from utils import evaluate_predictions, find_best_conf_threshold_and_plot
import json

def train_and_evaluate(model_weights, data_config, epochs=100, img_size=640, batch_size=16, devices=None, project="runs/train", conf_thres_range=np.linspace(0.1, 1, 10)):
    wandb.init(project=project, config={
        "model_weights": model_weights,
        "data_config": data_config,
        "epochs": epochs,
        "img_size": img_size,
        "batch_size": batch_size,
        "devices": devices,
        "conf_thres_range": conf_thres_range.tolist()
    })
    name = wandb.run.name

    model = YOLO(model_weights)
    results = model.train(data=data_config, epochs=epochs, imgsz=img_size, batch=batch_size, device=devices, project=project, name=name)
    path_weights = os.path.join(project, name, 'weights/best.pt')
    
    # Re-initialize the model with the best weights
    model = YOLO(path_weights)

    # Validate on validation set to find the best threshold
    path_val_images = data_config.replace('data.yaml', 'images/val')
    results_val = model(path_val_images)

    pred_folder = os.path.join(project, name, 'predictions_val')
    os.makedirs(pred_folder, exist_ok=True)
    
    for result in results_val:
        image_name = os.path.basename(result.path).replace('.jpg', '.txt')
        pred_path = os.path.join(pred_folder, image_name)
        with open(pred_path, 'w') as f:
            for box in result.boxes:
                box_info = f"{box.cls.item()} {box.xywhn[0][0].item()} {box.xywhn[0][1].item()} {box.xywhn[0][2].item()} {box.xywhn[0][3].item()} {box.conf.item()}\n"
                f.write(box_info)

    gt_folder = data_config.replace('data.yaml', 'labels/val')

    best_conf_thres, best_f1_score, best_precision, best_recall = find_best_conf_threshold_and_plot(pred_folder, gt_folder, conf_thres_range)

    wandb.log({"best_conf_thres": best_conf_thres, "best_f1_score": best_f1_score, "precision": best_precision, "recall": best_recall})

    # Test on the test set using the best threshold
    path_test_images = data_config.replace('data.yaml', 'images/test')
    results_test = model(path_test_images)

    pred_folder_test = os.path.join(project, name, 'predictions_test')
    os.makedirs(pred_folder_test, exist_ok=True)
    
    for result in results_test:
        image_name = os.path.basename(result.path).replace('.jpg', '.txt')
        pred_path = os.path.join(pred_folder_test, image_name)
        with open(pred_path, 'w') as f:
            for box in result.boxes:
                if box.conf.item() >= best_conf_thres:
                    box_info = f"{box.cls.item()} {box.xywhn[0][0].item()} {box.xywhn[0][1].item()} {box.xywhn[0][2].item()} {box.xywhn[0][3].item()} {box.conf.item()}\n"
                    f.write(box_info)

    gt_folder_test = data_config.replace('data.yaml', 'labels/test')
    
    test_results = evaluate_predictions(pred_folder_test, gt_folder_test, best_conf_thres)
    
    wandb.log({
        "test_f1_score": test_results["f1_score"],
        "test_precision": test_results["precision"],
        "test_recall": test_results["recall"]
    })

    # Save test metrics to a JSON file
    metrics_path = os.path.join(project, name, 'test_metrics.json')
    with open(metrics_path, 'w') as json_file:
        json.dump(test_results, json_file, indent=4)

    print(f"Test metrics saved at: {metrics_path}")

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train YOLO model, find the best threshold on validation, and evaluate on test set.')
    parser.add_argument('--model_weights', type=str, default='yolov5s.pt', help='Path to the pretrained model weights.')
    parser.add_argument('--data_config', type=str, required=True, help='Path to dataset YAML config file.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--img_size', type=int, default=640, help='Image size for training.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--devices', type=str, default=None, help='GPUs to use for training (e.g., "0", "0,1").')
    parser.add_argument('--project', type=str, default='runs/train', help='Project directory to save results.')
    parser.add_argument('--conf_thres_range', type=str, default="0.1,1.0,10", help='Confidence threshold range for evaluation (start,end,steps).')

    args = parser.parse_args()

    devices = [int(d) for d in args.devices.split(',')] if args.devices else None
    conf_thres_range = np.linspace(
        float(args.conf_thres_range.split(',')[0]), 
        float(args.conf_thres_range.split(',')[1]), 
        int(args.conf_thres_range.split(',')[2])
    )

    train_and_evaluate(
        model_weights=args.model_weights,
        data_config=args.data_config,
        epochs=args.epochs,
        img_size=args.img_size,
        batch_size=args.batch_size,
        devices=devices,
        project=args.project,
        conf_thres_range=conf_thres_range,
    )
