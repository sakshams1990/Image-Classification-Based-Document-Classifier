import os

import numpy as np
import pandas as pd
import torch
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from torch.utils.data import DataLoader
from torchvision import datasets
import seaborn as sns
from get_values_from_config_file import device, dataset_folder_path, batch_size, shuffle_images, num_workers, \
    final_images_folder, result_folder
from dataset_loader import val_test_image_transforms


def get_datasets_dataloaders_complete_dataset():
    image_dir = os.path.join(dataset_folder_path, final_images_folder)
    transform = val_test_image_transforms()
    complete_ds = datasets.ImageFolder(image_dir, transform=transform)
    complete_dl = DataLoader(complete_ds, batch_size=batch_size, shuffle=shuffle_images, num_workers=num_workers)
    return complete_ds, complete_dl


def get_predictions(model, data_loader):
    model = model.eval()
    predictions = []
    real_values = []
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            predictions.extend(preds)
            real_values.extend(labels)
    predictions = torch.as_tensor(predictions).cpu()
    real_values = torch.as_tensor(real_values).cpu()
    return predictions, real_values


def show_confusion_matrix(confusion_matrix, class_names, path):
    cm = confusion_matrix.copy()

    cell_counts = cm.flatten()

    cm_row_norm = cm / cm.sum(axis=1)[:, np.newaxis]

    row_percentages = ["{0:.2f}".format(value) for value in cm_row_norm.flatten()]

    cell_labels = [f"{cnt}\n{per}" for cnt, per in zip(cell_counts, row_percentages)]
    cell_labels = np.asarray(cell_labels).reshape(cm.shape[0], cm.shape[1])

    df_cm = pd.DataFrame(cm_row_norm, index=class_names, columns=class_names)

    hmap = sns.heatmap(df_cm, annot=cell_labels, fmt="", cmap="Blues")
    hmap.yaxis.set_ticklabels(hmap.yaxis.get_ticklabels(), rotation=0, ha='right')
    hmap.xaxis.set_ticklabels(hmap.xaxis.get_ticklabels(), rotation=30, ha='right')
    plt.ylabel('Real Class')
    plt.xlabel('Predicted Class')
    plt.savefig(f'{path}/Confusion Matrix Complete.jpg', bbox_inches='tight')


if __name__ == '__main__':
    pass
