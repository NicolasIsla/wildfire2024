import argparse
from ultralytics import YOLO
import wandb
import json
import os
from sklearn.metrics import precision_score, recall_score, f1_score
import numpy as np

def iou(box1, box2):
    """Calcula el IoU (Intersection over Union) entre dos cajas en formato xywhn."""
    # Convertir xywhn a formato xmin, ymin, xmax, ymax
    box1_xmin = box1[0] - box1[2] / 2
    box1_ymin = box1[1] - box1[3] / 2
    box1_xmax = box1[0] + box1[2] / 2
    box1_ymax = box1[1] + box1[3] / 2

    box2_xmin = box2[0] - box2[2] / 2
    box2_ymin = box2[1] - box2[3] / 2
    box2_xmax = box2[0] + box2[2] / 2
    box2_ymax = box2[1] + box2[3] / 2

    # Calcular la intersección
    xA = max(box1_xmin, box2_xmin)
    yA = max(box1_ymin, box2_ymin)
    xB = min(box1_xmax, box2_xmax)
    yB = min(box1_ymax, box2_ymax)

    interArea = max(0, xB - xA) * max(0, yB - yA)

    # Calcular el área de cada caja
    box1Area = (box1_xmax - box1_xmin) * (box1_ymax - box1_ymin)
    box2Area = (box2_xmax - box2_xmin) * (box2_ymax - box2_ymin)

    # Calcular IoU
    iou = interArea / float(box1Area + box2Area - interArea)

    return iou

def calculate_metrics(bounding_boxes, data_config, iou_threshold=0.1):
    true_labels_dir = data_config.replace('data.yaml', 'labels/test')
    pred_labels = []

    for result in bounding_boxes:
        image_path = result['image']
        image_name = os.path.basename(image_path).replace('.jpg', '')
        true_label_path = os.path.join(true_labels_dir, image_name + '.txt')
        
        if not os.path.exists(true_label_path):
            continue

        # Cargar las etiquetas verdaderas en formato xywhn
        with open(true_label_path, 'r') as f:
            true_labels = np.array([list(map(float, line.split()[1:])) for line in f.readlines()])
        
        # Extraer la caja predicha del diccionario
        pred_box = [
            result['cx'],  # X centro
            result['cy'],  # Y centro
            result['w'],   # Anchura
            result['h']    # Altura
        ]

        y_true = []
        y_pred = []
        for true_box in true_labels:
            iou_value = iou(true_box, pred_box)

            if iou_value > iou_threshold:
                y_true.append(1)
                y_pred.append(1)
            else:
                y_true.append(1)
                y_pred.append(0)

        # Si hay predicciones adicionales que no coinciden con etiquetas verdaderas
        y_pred.extend([1] * (len(bounding_boxes) - len(y_true)))
        y_true.extend([0] * (len(bounding_boxes) - len(y_true)))

        pred_labels.append((y_true, y_pred))

    # Calcular las métricas globales
    all_y_true = []
    all_y_pred = []
    for y_true, y_pred in pred_labels:
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)

    precision = precision_score(all_y_true, all_y_pred)
    recall = recall_score(all_y_true, all_y_pred)
    f1 = f1_score(all_y_true, all_y_pred)

    return precision, recall, f1

def train_yolo(model_weights, data_config, epochs=100, img_size=640, batch_size=16, devices=None, project="runs/train", iou_threshold=0.1):
    # Iniciar sesión en W&B
    wandb.init(project=project, config={
        "model_weights": model_weights,
        "data_config": data_config,
        "epochs": epochs,
        "img_size": img_size,
        "batch_size": batch_size,
        "devices": devices,
        "iou_threshold": iou_threshold
    })
    name = wandb.run.name  # Nombre de la ejecución de W&B

    # Cargar el modelo preentrenado
    model = YOLO(model_weights)  # Load a pretrained model
    # Entrenar el modelo
    results = model.train(data=data_config, epochs=epochs, imgsz=img_size, batch=batch_size, device=devices, project=project, name=name)
    # Cargar los mejores pesos
    path_weights = os.path.join(project, name, 'weights/best.pt')
    model = YOLO(path_weights)

    # Evaluar el modelo en el conjunto de prueba
    path_test_images = data_config.replace('data.yaml', 'images/test')
    results_test = model(path_test_images)

    # Guardar las bounding boxes y probabilidades en un archivo JSON
    bounding_boxes = []
    for result in results_test:
        for box in result.boxes:
            box_info = {
                'image': result.path,  # Ruta de la imagen
                'cx': box.xywhn[0][0].item(),  # X centro (normalizado)
                'cy': box.xywhn[0][1].item(),  # Y centro (normalizado)
                'w': box.xywhn[0][2].item(),   # Anchura (normalizado)
                'h': box.xywhn[0][3].item(),   # Altura (normalizado)
                'confidence': box.conf.item(),  # Probabilidad
                'class': box.cls.item()  # Clase predicha
            }
            bounding_boxes.append(box_info)

    output_json_path = os.path.join(project, name, 'test_bounding_boxes.json')
    with open(output_json_path, 'w') as json_file:
        json.dump(bounding_boxes, json_file, indent=4)

    print(f"Bounding boxes y probabilidades guardadas en: {output_json_path}")

    # Calcular métricas de precisión, recall y F1
    precision, recall, f1 = calculate_metrics(bounding_boxes, data_config, iou_threshold=iou_threshold)

    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}")

    # Guardar las métricas en W&B y en un archivo JSON
    wandb.log({"precision": precision, "recall": recall, "f1_score": f1})

    metrics_path = os.path.join(project, name, 'test_metrics.json')
    metrics = {
        'precision': precision,
        'recall': recall,
        'f1_score': f1
    }

    with open(metrics_path, 'w') as json_file:
        json.dump(metrics, json_file, indent=4)

    print(f"Métricas guardadas en: {metrics_path}")

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
    parser.add_argument('--iou_threshold', type=float, default=0.1, help='IoU threshold for calculating metrics.')

    args = parser.parse_args()

    # Convertir devices a una lista de enteros si se proporcionan
    devices = [int(d) for d in args.devices.split(',')] if args.devices else None

    train_yolo(
        model_weights=args.model_weights,
        data_config=args.data_config,
        epochs=args.epochs,
        img_size=args.img_size,
        batch_size=args.batch_size,
        devices=devices,
        project=args.project,
        iou_threshold=args.iou_threshold,
    )
