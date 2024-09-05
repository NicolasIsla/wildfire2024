from PIL import Image
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import random

# # Function to apply the transformations using the generated parameters
# def apply_transform_list(img_paths, train=True):
#     if not img_paths:
#         return []  # Early return if the list is empty

#     # Set up basic transforms
#     resize = transforms.Resize((112, 112))  # Resize to a fixed size
#     to_tensor = transforms.ToTensor()
#     normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],  # Standard normalization for pre-trained models
#                                      std=[0.229, 0.224, 0.225])

#     new_imgs = []
#     for img_path in img_paths:
#         img = Image.open(img_path).convert('RGB')  # Load the image and ensure it is in RGB format

#         # Apply deterministic resize
#         img = resize(img)

#         # Generate random transformation parameters only if training
#         if train:
#             params = {
#                 'horizontal_flip': random.random() < 0.5,  # 50% chance to flip
#                 'rotation': random.uniform(-10, 10),
#                 'brightness': random.uniform(0.9, 1.1),
#                 'contrast': random.uniform(0.9, 1.1),
#                 'saturation': random.uniform(0.9, 1.1),
#                 'hue': random.uniform(-0.1, 0.1)
#             }
            
#             # Apply transformations
#             if params['horizontal_flip']:
#                 img = F.hflip(img)
#             img = F.rotate(img, params['rotation'])
#             img = F.adjust_brightness(img, params['brightness'])
#             img = F.adjust_contrast(img, params['contrast'])
#             img = F.adjust_saturation(img, params['saturation'])
#             img = F.adjust_hue(img, params['hue'])

#         # Convert to tensor and normalize
#         img = to_tensor(img)
#         img = normalize(img)
#         new_imgs.append(img)

#     return new_imgs

def apply_transform_list(img_paths, train=True):
    if not img_paths:
        return []  # Early return if the list is empty

    # Base transformations that are always applied
    base_transforms = [
        transforms.Resize((112, 112)),  # Resize to a fixed size
        transforms.ToTensor(),  # Convert image to tensor
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]

    # Additional transformations for training to augment data
    train_transforms = [
        transforms.RandomHorizontalFlip(),  # 50% chance to flip horizontally
        transforms.RandomRotation((-10, 10)),  # Random rotation between -10 and 10 degrees
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1)  # Random color jitter
    ]

    # Compose the transformations
    if train:
        # If training, apply both training and base transformations
        all_transforms = transforms.Compose(train_transforms + base_transforms)
    else:
        # If not training, apply only the base transformations
        all_transforms = transforms.Compose(base_transforms)

    # Process each image path
    new_imgs = []
    for img_path in img_paths:
        img = Image.open(img_path).convert('RGB')  # Load the image and convert it to RGB
        img = all_transforms(img)  # Apply the composed transformations
        new_imgs.append(img)

    return new_imgs