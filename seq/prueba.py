
# Define the transformations
from torchvision import transforms
import random
import torch
import numpy as np

resize = transforms.Resize((112, 112))
horizontal_flip = transforms.RandomHorizontalFlip(p=0.5)
rotation = transforms.RandomRotation(degrees=10)
color_jitter = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
random_crop = transforms.RandomResizedCrop(size=(112, 112), scale=(0.9, 1.0), ratio=(0.9, 1.1))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])



# Function to apply the transformations using the generated parameters
def apply_transform_list(imgs, is_train=True):
    # Seed the random number generators
    seed = np.random.randint(2147483647)
    random.seed(seed)
    torch.manual_seed(seed)

    # Generate random transformation parameters
    params = {
        'horizontal_flip': random.random(),
        'rotation': random.uniform(-10, 10),
        'brightness': random.uniform(0.9, 1.1),
        'contrast': random.uniform(0.9, 1.1),
        'saturation': random.uniform(0.9, 1.1),
        'hue': random.uniform(-0.1, 0.1),
        'crop_params': random_crop.get_params(resize(imgs[0]), scale=(0.9, 1.0), ratio=(0.9, 1.1))
    }

    new_imgs = []

    for img in imgs:
        img = resize(img)
        if is_train:
            if params['horizontal_flip'] < 0.5:
                img = transforms.functional.hflip(img)

            img = transforms.functional.rotate(img, params['rotation'])

            img = transforms.functional.adjust_brightness(img, params['brightness'])
            img = transforms.functional.adjust_contrast(img, params['contrast'])
            img = transforms.functional.adjust_saturation(img, params['saturation'])
            img = transforms.functional.adjust_hue(img, params['hue'])
            img = transforms.functional.resized_crop(img, *params['crop_params'], size=(112, 112))
        img = to_tensor(img)
        img = normalize(img)

        new_imgs.append(img)

    return new_imgs



import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import glob
import random
import numpy as np
from torchmetrics import Accuracy, Precision, Recall
import wandb
from pytorch_lightning.loggers import WandbLogger
import torchvision.transforms as T

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'

class FireSeriesDataset(Dataset):
    def __init__(self, root_dir, img_size=112, transform=None, is_train=True):
        self.transform = transform
        self.sets = glob.glob(f"{root_dir}/**/*")
        self.img_size=img_size
        self.is_train = is_train
        random.shuffle(self.sets)

    def __len__(self):
        return len(self.sets)

    def __getitem__(self, idx):
        img_folder = self.sets[idx]
        img_list = glob.glob(f"{img_folder}/*.jpg")

        labels = []
        for file in img_list:
            label_file = file.replace("images", "labels").replace(".jpg", ".txt")
            with open(label_file, "r") as f:
                lines = f.readlines()

            labels.append(np.array(lines[0].split(" ")[1:5]).astype("float"))

        labels = np.array(labels)
        xc = np.median(labels[:, 0])
        yc = np.median(labels[:, 1])
        wb = np.max(labels[:, 2])
        hb = np.max(labels[:, 3])

        # Load all images first
        images = [Image.open(file) for file in img_list]
        w, h = images[0].size

        crop_size = max(wb*h, hb*h)
        if crop_size < self.img_size:
            crop_size = self.img_size

        x0 = int(xc * w - crop_size / 2)
        y0 = int(yc * h - crop_size / 2)
        x1 = int(xc * w  + crop_size / 2)
        y1 = int(yc * h + crop_size / 2)

        img_list = []

        for im in images:
            cropped_image = im.crop(
                (x0, y0, x1,y1))

            cropped_image = cropped_image.resize((self.img_size, self.img_size))
            img_list.append(cropped_image)

        tensor_list = apply_transform_list(img_list, is_train=self.is_train)
        # txc, w,h to t, x,c,w,h
        tensor_list = [tensor.unsqueeze(0) for tensor in tensor_list]

        # Concatena a lo largo de una nueva dimensión al principio, formando un tensor de (T, C, W, H)
        combined_tensor = torch.cat(tensor_list, dim=0)




        return torch.cat(tensor_list), int(img_folder.split("/")[-2])


class FireDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=16, img_size=112, num_workers=12):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.img_size = img_size
        self.num_workers = num_workers



    def setup(self, stage=None):
        self.train_dataset = FireSeriesDataset(
            os.path.join(self.data_dir, "train"), self.img_size, is_train=True
        )
        self.val_dataset = FireSeriesDataset(
            os.path.join(self.data_dir, "val"), self.img_size, is_train=False
        )

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

class FireClassifier(pl.LightningModule):
    def __init__(self, learning_rate=1e-4):
        super(FireClassifier, self).__init__()
        self.save_hyperparameters()

        # Usamos una ResNet como extractor de características.
        # Pretrained sobre ImageNet, usualmente se carga con 3 canales.
        resnet = models.resnet50(pretrained=True)
        # Removemos la capa final para usarla como extractor de características.
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])

        # LSTM que procesará las características extraídas.
        # Número de características de la salida del último bloque conv de ResNet.
        num_features = resnet.fc.in_features
        self.lstm = nn.LSTM(input_size=32768, hidden_size=256, batch_first=True, num_layers=1)

        # Capa de clasificación.
        self.classifier = nn.Linear(256, 1)  # Salida binaria

        # Dropout para regularización
        self.dropout = nn.Dropout(0.2)

        # Métricas
        self.train_accuracy = Accuracy(task="binary")
        self.val_accuracy = Accuracy(task="binary")
        self.train_precision = Precision(task="binary")
        self.val_precision = Precision(task="binary")
        self.train_recall = Recall(task="binary")
        self.val_recall = Recall(task="binary")

    def forward(self, x):
        # x shape: [batch_size, seq_length, channels, height, width]
        # Procesa cada imagen de la secuencia a través del extractor de características.
        batch_size, seq_length, C, H, W = x.size()
        x = x.view(batch_size * seq_length, C, H, W)
        x = self.feature_extractor(x)

        # Reformatear salida para la LSTM
        # Deberás asegurarte que las dimensiones coincidan con lo que espera la LSTM.
        x = x.view(batch_size, seq_length, -1)

        # Pasar las características por la LSTM
        x, _ = self.lstm(x)


        # Solo nos interesa la última salida de la secuencia para la clasificación
        x = x[:, -1, :]
        x = self.dropout(x)
        x = self.classifier(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch

        y_hat = self(x).squeeze()

        loss = F.binary_cross_entropy_with_logits(y_hat, y.float())
        acc = self.train_accuracy(torch.sigmoid(y_hat), y.int())
        precision = self.train_precision(torch.sigmoid(y_hat), y.int())
        recall = self.train_recall(torch.sigmoid(y_hat), y.int())
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        self.log("train_precision", precision)
        self.log("train_recall", recall)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x).squeeze()


        loss = F.binary_cross_entropy_with_logits(y_hat, y.float())
        acc = self.val_accuracy(torch.sigmoid(y_hat), y.int())
        precision = self.val_precision(torch.sigmoid(y_hat), y.int())
        recall = self.val_recall(torch.sigmoid(y_hat), y.int())
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        self.log("val_precision", precision)
        self.log("val_recall", recall)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams['learning_rate'], weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
        return {
            'optimizer': optimizer,
            'lr_scheduler': scheduler
        }

# Initialize the DataModule
data_dir = "/data/nisla/temporal_ds/images"
data_module = FireDataModule(data_dir)
# set
data_module.setup()
# Función para obtener la imagen y etiqueta por índice de lote y posición
def get_image_by_index(loader, batch_index, img_index):
    for i, (x, y) in enumerate(loader):
        # guardar el índice del lote
        if i == batch_index:
            return x[img_index], y[img_index]


train_loader = data_module.train_dataloader()
# Extrae la imagen y etiqueta específicas
images, label = get_image_by_index(train_loader, 1, 4) 

# save the image
for i in range(images.size(0)):
    T.ToPILImage()(images[i].cpu()).save(f"image_{i}.png")