from torchvision import transforms
import random
import torch
import numpy as np

# Define the transformations
resize = transforms.Resize((112, 112))
horizontal_flip = transforms.RandomHorizontalFlip(p=0.5)
rotation = transforms.RandomRotation(degrees=10)
color_jitter = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)
random_crop = transforms.RandomResizedCrop(size=(112, 112), scale=(0.9, 1.0), ratio=(0.9, 1.1))
to_tensor = transforms.ToTensor()
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

# Function to apply the transformations using the generated parameters
def apply_transform_list(imgs, train=True):
    if not imgs:
        return []  # Early return if the list is empty

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
        if train:
            if params['horizontal_flip'] < 0.5:
                img = transforms.functional.hflip(img)
            
            img = transforms.functional.rotate(img, params['rotation'])
            img = transforms.functional.adjust_brightness(img, params['brightness'])
            img = transforms.functional.adjust_contrast(img, params['contrast'])
            img = transforms.functional.adjust_saturation(img, params['saturation'])
            img = transforms.functional.adjust_hue(img, params['hue'])
        img = to_tensor(img)
        img = normalize(img)
        new_imgs.append(img)

    return new_imgs