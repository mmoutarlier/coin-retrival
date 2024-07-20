import torch
import torch.nn as nn
# import shutil
import os
# import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
from torchvision.models import EfficientNet_V2_S_Weights
from tqdm import tqdm

import pandas as pd
import json
from sklearn.model_selection import train_test_split
import cv2 as cv
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def read_image(image_path: str) -> np.ndarray:
    img = cv.imread(image_path, cv.IMREAD_COLOR)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    return img


def save_image(image: np.ndarray, image_path: str):
    """save image in .png"""
    cv.imwrite(image_path, cv.cvtColor(image, cv.COLOR_RGB2BGR))


USE_SIZE_REF = (1200, 800)


def resize_image(image: np.ndarray, size: tuple) -> np.ndarray:
    #check the scale in conserved
    assert size[0] / image.shape[1] == size[1] / image.shape[0]

    return cv.resize(image, size)


#model_ft = models.resnet18(weights='IMAGENET1K_V1')

def divide_equally(n1, n2):
    # Calculate the quotient and remainder
    quotient = n1 // n2
    remainder = n1 % n2

    # Distribute shares
    shares = [quotient] * n2

    # Adjust shares to achieve equality
    for i in range(remainder):
        shares[i] += 1

    np.random.shuffle(shares)

    return shares


def create_train_valid_folders(label_file):
    labels_dict = json.load(open(label_file))
    labels_df = pd.Series(labels_dict).sort_index().to_frame('label').reset_index().rename(
        columns={'index': 'image_path'})
    #Train-valid split 80-20, class balanced

    X_train, X_val, y_train, y_val = train_test_split(
        labels_df['image_path'], labels_df['label'], test_size=0.2, stratify=labels_df['label'], random_state=42
    )

    train_df = pd.DataFrame({'path': X_train, 'label': y_train})
    val_df = pd.DataFrame({'path': X_val, 'label': y_val})

    return train_df, val_df


def augment_coin(img, nb_augmentations):
    #rotation angle for each augmentation
    #avge of 10 border pixels for filling
    avg_pixel = img[0:10, 0:10, :].mean(axis=(0, 1))
    avg_pixel = (int(avg_pixel[0]), int(avg_pixel[1]), int(avg_pixel[2]))

    angles = np.random.choice(np.arange(0, 360), size=nb_augmentations, replace=False)
    blurring = [np.random.choice([0, 1]) for _ in range(nb_augmentations)]
    pil_img = Image.fromarray(img)
    augmentated_imgs = []
    for angle, blur in zip(angles, blurring):
        augmented_img = transforms.functional.rotate(img=pil_img, angle=float(angle), fill=avg_pixel)
        if blur:
            augmented_img = transforms.GaussianBlur([7, 7], sigma=(0.1, 2.0))(augmented_img)
        augmentated_imgs.append(augmented_img)
    return pil_img, augmentated_imgs


def augment_data(path_label_df, output_dir, nb_per_class, central_croping_size=750, size=(256, 256)):
    label_counts = path_label_df['label'].value_counts()

    nbs_augment = {label: divide_equally(nb_per_class - count, count) for label, count in label_counts.items()}

    for label in path_label_df['label'].unique():
        os.makedirs(f'{output_dir}/{label}', exist_ok=True)

    img_paths_per_label = path_label_df.groupby('label')['path'].apply(list)

    for label, nb_s_to_augment in nbs_augment.items():
        for img_path, nb_to_augment in tqdm(zip(img_paths_per_label[label], nb_s_to_augment),
                                            total=len(img_paths_per_label[label]), desc=f'Augmenting {label}'):
            img = read_image(img_path[8:])
            parts = img_path.split('/')
            bg = parts[-3]
            id = parts[-2]
            local_id = parts[-1].split('_')[1][0]

            save_path = f'{output_dir}/{label}/{bg}_{id}_local_{local_id}'
            pil_img, augmentated_imgs = augment_coin(img, nb_to_augment)

            croped_img = transforms.functional.center_crop(pil_img, central_croping_size)
            array_img = np.array(croped_img)
            resized_img = resize_image(array_img, size)
            save_image(resized_img, f'{save_path}.jpg')

            for i, augmentated_img in enumerate(augmentated_imgs):
                croped_augmnted_img = transforms.functional.center_crop(augmentated_img, central_croping_size)
                array_augmentated_img = np.array(croped_augmnted_img)
                resized_augmnted_img = resize_image(array_augmentated_img, size)

                save_image(resized_augmnted_img, f'{save_path}_aug_{i}.jpg')


def get_data_loaders(data_dir, batch_size=32):
    # Define transforms for the training and validation sets
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(root=f'{data_dir}/{x}', transform=data_transforms[x])
                      for x in ['train', 'val']}
    json.dump(image_datasets['train'].class_to_idx, open('models/class_to_idx.json', 'w'))
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True, num_workers=4)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    return dataloaders, dataset_sizes, class_names


def get_test_data_loader(data_dir, batch_size=32):
    data_transforms = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    sub_dirs = list(sorted(filter(lambda x: x[0] != '.', os.listdir(data_dir))))
    image_datasets = datasets.ImageFolder(root=data_dir, transform=data_transforms, target_transform=lambda x: sub_dirs[x])
    dataloader = DataLoader(image_datasets, batch_size=batch_size, shuffle=False, num_workers=4)
    return dataloader


def get_model(num_classes):
    model = models.efficientnet_v2_s(weights=EfficientNet_V2_S_Weights.IMAGENET1K_V1)
    model.classifier = nn.Linear(1280, num_classes)
    return model


def train_model(model, dataloaders, dataset_sizes, device, criterion, optimizer, num_epochs=25, scheduler=None):
    best_model_wts = model.state_dict()
    best_acc = 0.0

    acc_hist = {'train': [], 'val': []}
    model = model.to(device)
    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in tqdm(dataloaders[phase], total=len(dataloaders[phase]),
                                       desc=f'{phase} epoch {epoch}'):
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        if scheduler is not None:
                            scheduler.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            acc_hist[phase].append(epoch_acc.detach().cpu().item())

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()
                print("New best model")
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
    return best_model_wts, acc_hist


