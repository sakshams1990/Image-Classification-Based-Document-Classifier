import os

from torch.utils.data import DataLoader
from torchvision import transforms, datasets

from get_values_from_config_file import image_resize, image_std, image_mean, rot_degree, shuffle_images, batch_size
from get_values_from_config_file import num_workers, dataset_folder_path, dataset_split_images_folder


# Transform images of train folder
def train_image_transforms():
    train_transform = transforms.Compose([
        transforms.Resize(size=image_resize),
        transforms.RandomRotation(degrees=rot_degree),
        transforms.ColorJitter(),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(image_mean, image_std)
    ])

    return train_transform


def val_test_image_transforms():
    val_test_transform = transforms.Compose([
        transforms.Resize(size=image_resize),
        transforms.ToTensor(),
        transforms.Normalize(image_mean, image_std)
    ])
    return val_test_transform


def train_dataloader(train_folder, train_transform):
    train_ds = datasets.ImageFolder(train_folder, transform=train_transform)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=shuffle_images, num_workers=num_workers)
    return train_ds, train_dl


def val_test_dataloader(val_folder, test_folder, val_test_transform):
    val_ds = datasets.ImageFolder(val_folder, transform=val_test_transform)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=shuffle_images, num_workers=num_workers)

    test_ds = datasets.ImageFolder(test_folder, transform=val_test_transform)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=shuffle_images, num_workers=num_workers)

    return val_ds, val_dl, test_ds, test_dl


def get_datasets_dataloaders_for_model():
    train_dir = os.path.join(dataset_folder_path, dataset_split_images_folder, 'train')
    val_dir = os.path.join(dataset_folder_path, dataset_split_images_folder, 'val')
    test_dir = os.path.join(dataset_folder_path, dataset_split_images_folder, 'test')
    train_transform = train_image_transforms()
    val_test_transform = val_test_image_transforms()
    train_ds, train_dl = train_dataloader(train_dir, train_transform)
    val_ds, val_dl, test_ds, test_dl = val_test_dataloader(val_dir, test_dir, val_test_transform)
    return train_ds, train_dl, val_ds, val_dl, test_ds, test_dl


if __name__ == '__main__':
    pass
