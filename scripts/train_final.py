import argparse
from ultralytics import YOLO
import wandb
import json
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import xywh2xyxy, box_iou

def evaluate_predictions(pred_folder, gt_folder, conf_th=0.1, iou_th=0.1, cat=None):
    nb_fp, nb_tp, nb_fn = 0, 0, 0

    gt_filenames = [
        os.path.splitext(os.path.basename(f))[0]
        for f in glob.glob(os.path.join(gt_folder, "*.txt"))
    ]
    pred_filenames = [
        os.path.splitext(os.path.basename(f))[0]
        for f in glob.glob(os.path.join(pred_folder, "*.txt"))
    ]

    all_filenames = set(gt_filenames + pred_filenames)
    if cat is not None:
        all_filenames = [f for f in all_filenames if cat == f.split("_")[0].lower()]
    
    for filename in all_filenames:
        gt_file = os.path.join(gt_folder, f"{filename}.txt")
        pred_file = os.path.join(pred_folder, f"{filename}.txt")

        gt_boxes = []
        if os.path.isfile(gt_file) and os.path.getsize(gt_file) > 0:
            with open(gt_file, "r") as f:
                gt_boxes = [
                    xywh2xyxy(np.array(line.strip().split(" ")[1:5]).astype(float))
                    for line in f.readlines()
                ]

        gt_matches = np.zeros(len(gt_boxes), dtype=bool)

        if os.path.isfile(pred_file) and os.path.getsize(pred_file) > 0:
            with open(pred_file, "r") as f:
                pred_boxes = [line.strip().split(" ") for line in f.readlines()]

            for pred_box in pred_boxes:
                try:
                    _, x, y, w, h, conf = map(float, pred_box)
                except:
                    print(pred_file)
                if conf < conf_th:
                    continue
                pred_box = xywh2xyxy(np.array([x, y, w, h]))

                if gt_boxes:
                    matches = [box_iou(pred_box, gt_box) > iou_th for gt_box in gt_boxes]
                    if any(matches):
                        nb_tp += 1
                        gt_matches = gt_matches | matches
                    else:
                        nb_fp += 1
                else:
                    nb_fp += 1

        if gt_boxes:
            nb_fn += len(gt_boxes) - np.sum(gt_matches)

    precision = nb_tp / (nb_tp + nb_fp) if (nb_tp + nb_fp) > 0 else 0
    recall = nb_tp / (nb_tp + nb_fn) if (nb_tp + nb_fn) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {"precision": precision, "recall": recall, "f1_score": f1_score}

def find_best_conf_threshold(pred_folder, gt_folder, conf_thres_range, iou_th=0.1, cat=None):
    best_conf_thres, best_f1_score, best_precision, best_recall = 0, 0, 0, 0

    for conf_thres in conf_thres_range:
        results = evaluate_predictions(pred_folder, gt_folder, conf_thres, iou_th, cat)
        if results["f1_score"] > best_f1_score:
            best_conf_thres = conf_thres
            best_f1_score = results["f1_score"]
            best_precision = results["precision"]
            best_recall = results["recall"]

    return best_conf_thres, best_f1_score, best_precision, best_recall

def find_best_conf_threshold_and_plot(pred_folder, gt_folder, conf_thres_range, iou_th=0.1, plot=True):
    f1_scores, precisions, recalls = [], [], []

    for conf_thres in conf_thres_range:
        results = evaluate_predictions(pred_folder, gt_folder, conf_thres, iou_th)
        f1_scores.append(results["f1_score"])
        precisions.append(results["precision"])
        recalls.append(results["recall"])

    best_idx = np.argmax(f1_scores)
    best_conf_thres = conf_thres_range[best_idx]
    best_f1_score = f1_scores[best_idx]
    best_precision = precisions[best_idx]
    best_recall = recalls[best_idx]

    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(conf_thres_range, f1_scores, label="F1 Score", color="blue", marker="o")
        plt.plot(conf_thres_range, precisions, label="Precision", color="green", linestyle="--")
        plt.plot(conf_thres_range, recalls, label="Recall", color="red", linestyle="-.")
        plt.scatter(best_conf_thres, best_f1_score, color="blue", s=100, edgecolor="k", zorder=5)
        plt.scatter(best_conf_thres, best_precision, color="green", s=100, edgecolor="k", zorder=5)
        plt.scatter(best_conf_thres, best_recall, color="red", s=100, edgecolor="k", zorder=5)
        plt.text(best_conf_thres, best_f1_score, f" Best F1: {best_f1_score:.2f}\n Precision: {best_precision:.2f}\n Recall: {best_recall:.2f}", fontsize=9, verticalalignment="bottom")
        plt.title("F1 Score, Precision, and Recall vs. Confidence Threshold")
        plt.xlabel("Confidence Threshold")
        plt.ylabel("Metric Value")
        plt.legend()
        plt.grid(True)
        plt.show()

    return best_conf_thres, best_f1_score, best_precision, best_recall

def train_yolo(model_weights, data_config, epochs=100, img_size=640, batch_size=16, devices=None, project="runs/train", conf_thres_range=np.linspace(0.1, 1, 10)):
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
    model = YOLO(path_weights)

    path_test_images = data_config.replace('data.yaml', 'images/test')
    results_test = model(path_test_images)

    pred_folder = os.path.join(project, name, 'predictions')
    os.makedirs(pred_folder, exist_ok=True)
    
    for result in results_test:
        image_name = os.path.basename(result.path).replace('.jpg', '.txt')
        pred_path = os.path.join(pred_folder, image_name)
        with open(pred_path, 'w') as f:
            for box in result.boxes:
                box_info = f"{box.cls.item()} {box.xywhn[0][0].item()} {box.xywhn[0][1].item()} {box.xywhn[0][2].item()} {box.xywhn[0][3].item()} {box.conf.item()}\n"
                f.write(box_info)

    gt_folder = data_config.replace('data.yaml', 'labels/test')

    best_conf_thres, best_f1_score, best_precision, best_recall = find_best_conf_threshold(pred_folder, gt_folder, conf_thres_range)

    wandb.log({"best_conf_thres": best_conf_thres, "best_f1_score": best_f1_score, "precision": best_precision, "recall": best_recall})

    metrics_path = os.path.join(project, name, 'best_metrics.json')
    metrics = {
        'best_conf_thres': best_conf_thres,
        'f1_score': best_f1_score,
        'precision': best_precision,
        'recall': best_recall
    }

    with open(metrics_path, 'w') as json_file:
        json.dump(metrics, json_file, indent=4)

    print(f"Métricas guardadas en: {metrics_path}")

    wandb.finish()

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train YOLO using a pretrained model.')
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
    train_yolo(
        model_weights=args.model_weights,
        data_config=args.data_config,
        epochs=args.epochs,
        img_size=args.img_size,
        batch_size=args.batch_size,
        devices=devices,
        project=args.project,
        conf_thres_range=conf_thres_range,
    )