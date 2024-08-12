import argparse
import wandb
from ultralytics import YOLO

# Define una función para la búsqueda de hiperparámetros
def train_yolo(config=None):
    with wandb.init(config=config):
        config = wandb.config
        
        # Cargar el modelo preentrenado
        model = YOLO(config.model_weights)  # Load a pretrained model

        # Entrenar el modelo
        results = model.train(
            data=config.data_config,
            epochs=config.epochs,
            imgsz=config.img_size,
            batch=config.batch_size,
            device=config.devices,
            project=config.project,
            name=config.name,
            optimizer=config.optimizer,
            lr0=config.lr0,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            multi_scale=config.multi_scale,
            rect=config.rect,
            single_cls=config.single_cls,
            freeze=config.freeze,
            amp=config.amp
        )

        return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train YOLOv8 using a pretrained model.')
    parser.add_argument('--model_weights', type=str, default='yolov8s.pt', help='Path to the pretrained model weights.')
    parser.add_argument('--data_config', type=str, required=True, help='Path to dataset YAML config file.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--img_size', type=int, default=640, help='Image size for training.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for training.')
    parser.add_argument('--devices', type=str, default=None, help='GPUs to use for training (e.g., "0", "0,1").')
    parser.add_argument('--project', type=str, default='models/ForestFireDetection', help='Project directory to save results.')
    parser.add_argument('--name', type=str, default='fire_detection_exp', help='Experiment name to save results.')

    args = parser.parse_args()

    # Convertir devices a lista de enteros si se proporcionan
    devices = [int(d) for d in args.devices.split(',')] if args.devices else None

    # Configuración inicial para W&B
    sweep_config = {
        'method': 'bayes',  # Optimización bayesiana
        'metric': {'name': 'val_loss', 'goal': 'minimize'},
        'parameters': {
            'model_weights': {'value': args.model_weights},
            'data_config': {'value': args.data_config},
            'epochs': {'value': args.epochs},  # Fijar el número de épocas en 50
            'img_size': {'values': [640, 512, 320]},  # Tamaños que podrían capturar bien los detalles
            'batch_size': {'values': [16, 32, 64]},  # Ajustes en el tamaño del batch
            'devices': {'value': devices},
            'project': {'value': args.project},
            'name': {'value': args.name},
            'optimizer': {'values': ['SGD', 'Adam', 'AdamW']},
            'lr0': {'min': 0.0001, 'max': 0.01},  # Rango de tasas de aprendizaje
            'momentum': {'min': 0.85, 'max': 0.99},
            'weight_decay': {'min': 0.0, 'max': 0.0005},
            'multi_scale': {'values': [True, False]},
            'rect': {'values': [True, False]},
            'single_cls': {'values': [True, False]},  # Podría ser útil si sólo tienes una clase (incendio)
            'freeze': {'values': [None, 10]},
            'amp': {'values': [True, False]},  # Entrenamiento con precisión mixta
        }
    }

    # Crear y ejecutar el sweep
    sweep_id = wandb.sweep(sweep_config, project=args.project)
    wandb.agent(sweep_id, function=train_yolo, count=20)  # Ejecutar 20 experimentos


