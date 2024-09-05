import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from utils import apply_transform_list
import random

class FireSeriesDataset(Dataset):
    def __init__(self, root_dir, img_size=112, transform=None, is_train=True):
        self.transform = transform
        self.sets = glob.glob(f"{root_dir}/**/*")
        self.img_size=img_size
        random.shuffle(self.sets)
        self.train = is_train

    def __len__(self):
        return len(self.sets)

    def __getitem__(self, idx):
        img_folder = self.sets[idx]
        img_list = glob.glob(f"{img_folder}/*.jpg")

       
        

        tensor_list = apply_transform_list(img_list, self.train)
        label = int(img_folder.split("/")[-2])
        print(label)

        return torch.cat(tensor_list, dim=0), int(img_folder.split("/")[-2])

class FireDataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size=16, img_size=112, num_workers=4):
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
    
# Initialize the DataModule
data_dir = "/data/nisla/temporal_ds/images"
data_module = FireDataModule(data_dir)
data_module.setup()
# print size of the datasets

print(f"Number of training samples: {len(data_module.train_dataset)}")
print(f"Number of validation samples: {len(data_module.val_dataset)}")
def print_batch_shape(dataloader):
    # Fetch one batch from the dataloader
    for batch in dataloader:
        x, y = batch
        print(f"Batch x shape: {x.shape}")  # Shape of the images
        print(f"Batch y shape: {y.shape}")  # Shape of the n
        break  # We only need to print the shape of one batch

# Call this function with the training dataloader
print_batch_shape(data_module.train_dataloader())

import torch
import torch.nn as nn
import torchvision.models as models
from pytorch_lightning import LightningModule

class ResNetLSTM(LightningModule):
    def __init__(self, hidden_dim, num_layers, bidirectional=False, num_classes=2):
        super().__init__()
        self.resnet = models.resnet18(pretrained=True)  # Using a pretrained ResNet18
        self.resnet.fc = nn.Identity()  # Remove the final fully connected layer

        # Assuming the output of ResNet18 is 512 features
        self.lstm = nn.LSTM(input_size=512, hidden_size=hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=bidirectional)

        # Classifier layer
        multiplier = 2 if bidirectional else 1
        self.classifier = nn.Linear(hidden_dim * multiplier, num_classes)

    def forward(self, x):
        batch_size, timesteps, C, H, W = x.size()
        x = x.view(batch_size * timesteps, C, H, W)
        x = self.resnet(x)  # Apply ResNet to each image
        x = x.view(batch_size, timesteps, -1)
        x, (h_n, c_n) = self.lstm(x)
        # Use the last hidden state for classification
        x = self.classifier(x[:, -1, :])
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = nn.functional.cross_entropy(logits, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = nn.functional.cross_entropy(logits, y)
        self.log('val_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        return optimizer

model = ResNetLSTM(hidden_dim=256, num_layers=1)