import argparse
from ultralytics import YOLO
import os
import json
import numpy as np
import wandb

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection

    return intersection / union if union > 0 else 0

def parse_label_file(label_path, img_width, img_height):
    boxes = []
    with open(label_path, 'r') as f:
        lines = f.readlines()
        for line in lines:
            parts = line.strip().split()
            cls = int(parts[0])
            x_center = float(parts[1]) * img_width
            y_center = float(parts[2]) * img_height
            width = float(parts[3]) * img_width
            height = float(parts[4]) * img_height
            xmin = x_center - width / 2
            ymin = y_center - height / 2
            xmax = x_center + width / 2
            ymax = y_center + height / 2
            boxes.append((cls, xmin, ymin, xmax, ymax))
    return boxes

def compute_metrics(predictions, ground_truths, iou_threshold=0.5):
    tp = 0
    fp = 0
    fn = 0

    for pred_box in predictions:
        max_iou = 0
        best_gt = None
        for gt_box in ground_truths:
            if pred_box[0] == gt_box[0]:  # Si la clase es la misma
                iou = calculate_iou(pred_box[1:], gt_box[1:])
                if iou > max_iou:
                    max_iou = iou
                    best_gt = gt_box
        if max_iou > iou_threshold:
            tp += 1
            ground_truths.remove(best_gt)  # Evitar que se cuente dos veces
        else:
            fp += 1

    fn = len(ground_truths)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1

def train_and_test_yolo(model_weights, data_config, epochs=100, img_size=640, batch_size=16, devices=None, project="runs/train", name="exp"):
    # Inicializar Weights & Biases
    run = wandb.init(project=project, name=name, config={
        "model_weights": model_weights,
        "data_config": data_config,
        "epochs": epochs,
        "img_size": img_size,
        "batch_size": batch_size,
        "devices": devices
    })

    # Cargar el modelo preentrenado
    model = YOLO(model_weights)

    # Entrenar el modelo
    results = model.train(data=data_config, epochs=epochs, imgsz=img_size, batch=batch_size, device=devices, project=project, name=name)

    # Cargar los mejores pesos
    path_weights = os.path.join(project, name, 'weights', 'best.pt')
    model = YOLO(path_weights)

    # Realizar predicciones sobre las imágenes de prueba
    path_test_images = data_config.replace('data.yaml', 'images/test')
    results_test = model(path_test_images, device=devices)

    # Asumimos que todas las imágenes tienen el mismo tamaño
    img_width, img_height = 640, 480
    
    # Ruta para los labels de prueba
    path_test_labels = data_config.replace('data.yaml', 'labels/test')

    # Guardar las bounding boxes y probabilidades
    bounding_boxes = []
    all_precisions = []
    all_recalls = []
    all_f1s = []

    for result in results_test:
        predictions = []
        for box in result.boxes:
            box_info = {
                'image': result.path,  # Ruta de la imagen
                'xmin': box.xyxy[0][0].item(),  # Coordenada X mínima
                'ymin': box.xyxy[0][1].item(),  # Coordenada Y mínima
                'xmax': box.xyxy[0][2].item(),  # Coordenada X máxima
                'ymax': box.xyxy[0][3].item(),  # Coordenada Y máxima
                'confidence': box.conf.item(),  # Probabilidad
                'class': box.cls.item()  # Clase predicha
            }
            predictions.append((box_info['class'], box_info['xmin'], box_info['ymin'], box_info['xmax'], box_info['ymax']))
            bounding_boxes.append(box_info)

        # Parsear el archivo de label correspondiente
        label_path = os.path.join(path_test_labels, os.path.basename(result.path).replace('.jpg', '.txt'))
        ground_truth_boxes = parse_label_file(label_path, img_width, img_height)

        # Calcular métricas para esta imagen
        precision, recall, f1 = compute_metrics(predictions, ground_truth_boxes, iou_threshold=0.5)
        all_precisions.append(precision)
        all_recalls.append(recall)
        all_f1s.append(f1)

    # Guardar las bounding boxes y probabilidades en un archivo JSON
    output_json_path = os.path.join(project, name, 'test_bounding_boxes.json')
    with open(output_json_path, 'w') as json_file:
        json.dump(bounding_boxes, json_file, indent=4)

    print(f"Bounding boxes y probabilidades guardadas en: {output_json_path}")

    # Calcular mAP@0.5, Recall y F1 Score promedio
    mean_precision = np.mean(all_precisions)
    mean_recall = np.mean(all_recalls)
    mean_f1 = np.mean(all_f1s)

    print(f"mAP@0.5: {mean_precision:.4f}")
    print(f"Recall: {mean_recall:.4f}")
    print(f"F1 Score: {mean_f1:.4f}")

    # Loguear las métricas en Weights & Biases
    wandb.log({
        "mAP@0.5": mean_precision,
        "Recall": mean_recall,
        "F1 Score": mean_f1
    })

    # Finalizar la sesión de W&B
    wandb.finish()

    return results_test

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train and test YOLOv5 using a pretrained model.')
    parser.add_argument('--model_weights', type=str, default='yolov5s.pt', help='Path to the pretrained model weights.')
    parser.add_argument('--data_config', type=str, required=True, help='Path to dataset YAML config file.')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
    parser.add_argument('--img_size', type=int, default=640, help='Image size for training.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--devices', type=str, default=None, help='GPUs to use for training (e.g., "0", "0,1").')
    parser.add_argument('--project', type=str, default='runs/train', help='Project directory to save results.')
    parser.add_argument('--name', type=str, default='exp', help='Experiment name to save results.')

    args = parser.parse_args()

    # Convertir devices a una lista de enteros si se proporcionan
    devices = [int(d) for d in args.devices.split(',')] if args.devices else None

    train_and_test_yolo(
        model_weights=args.model_weights,
        data_config=args.data_config,
        epochs=args.epochs,
        img_size=args.img_size,
        batch_size=args.batch_size,
        devices=devices,
        project=args.project,
        name=args.name
    )
