import os
import glob
import torch
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from utils import apply_transform_list

class FireSeriesDataset(Dataset):
    def __init__(self, root_dir, img_size=112, transform=None, is_train=True):
        self.root_dir = root_dir
        self.img_size = img_size
        self.transform = transform
        self.is_train = is_train
        self.images = glob.glob(os.path.join(root_dir, "*.jpg"))
        print(f"Loading images from: {self.root_dir}")
        print(f"Found {len(self.images)} images")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img_list = glob.glob(f"{os.path.dirname(img_path)}/*.jpg")
        tensor_list = apply_transform_list(img_list, self.is_train)
        return torch.cat(tensor_list, dim=0), int(os.path.basename(img_path).split("_")[0])

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
    
# Initialize the DataModule
data_dir = "/data/nisla/temporal_ds/images"
data_module = FireDataModule(data_dir)
data_module.setup()
# print size of the datasets

print(f"Number of training samples: {len(data_module.train_dataset)}")
print(f"Number of validation samples: {len(data_module.val_dataset)}")

