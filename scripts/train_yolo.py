import argparse
from ultralytics import YOLO
import wandb
import json
import os

def train_yolo(model_weights, data_config, epochs=100, img_size=640, batch_size=16, devices=None, project="runs/train"):
    # Iniciar sesión en W&B
    wandb.init(project=project, config={
        "model_weights": model_weights,
        "data_config": data_config,
        "epochs": epochs,
        "img_size": img_size,
        "batch_size": batch_size,
        "devices": devices
    })
    name = wandb.run.name  # Nombre de la ejecución de W&B

    # Cargar el modelo preentrenado
    model = YOLO(model_weights)  # Load a pretrained model
    # Entrenar el modelo
    results = model.train(data=data_config, epochs=epochs, imgsz=img_size, batch=batch_size, device=devices, project=project, name=name)
    # load the best weights
    path_weights = project + '/' + name + '/weights/best.pt'
    # load the best weights
    model = YOLO(path_weights)
    # test the model # ./data/AiForMankind/DS/data.yaml
    path_test_images = data_config.replace('data.yaml', 'images/test')
    results_test = model(path_test_images)
    # save the results
    bounding_boxes = []
    for result in results_test:
        for box in result.boxes:
            box_info = {
                'image': result.path,  # Ruta de la imagen
                'xmin': box.xyxy[0].item(),  # Coordenada X mínima
                'ymin': box.xyxy[1].item(),  # Coordenada Y mínima
                'xmax': box.xyxy[2].item(),  # Coordenada X máxima
                'ymax': box.xyxy[3].item(),  # Coordenada Y máxima
                'confidence': box.conf.item(),  # Probabilidad
                'class': box.cls.item()  # Clase predicha
            }
            bounding_boxes.append(box_info)

    # Guardar las bounding boxes y probabilidades en un archivo JSON
    output_json_path = os.path.join(project, name, 'test_bounding_boxes.json')
    with open(output_json_path, 'w') as json_file:
        json.dump(bounding_boxes, json_file, indent=4)

    print(f"Bounding boxes y probabilidades guardadas en: {output_json_path}")


    



    # Finalizar la sesión de W&B
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
    # parser.add_argument('--name', type=str, default='exp', help='Experiment name to save results.')

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
    )


