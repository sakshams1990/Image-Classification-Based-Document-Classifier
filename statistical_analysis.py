import os
from get_values_from_config_file import dataset_split_images_folder, dataset_folder_path, result_folder
import pandas as pd
from collections import Counter


def perform_statistical_analysis(train_dataset, val_dataset, test_dataset):
    class_to_idx = train_dataset.class_to_idx
    idx_to_class = {
        idx: class_
        for class_, idx in class_to_idx.items()
    }

    train_count = Counter([idx_to_class[x] for x in train_dataset.targets])
    val_count = Counter([idx_to_class[x] for x in val_dataset.targets])
    test_count = Counter([idx_to_class[x] for x in test_dataset.targets])

    train_count = pd.DataFrame({'class': list(train_count.keys()), 'train count': list(train_count.values())})
    val_count = pd.DataFrame({'class': list(val_count.keys()), 'val count': list(val_count.values())})
    test_count = pd.DataFrame({'class': list(test_count.keys()), 'test count': list(test_count.values())})

    cnt_df = pd.merge(train_count, val_count, on='class', how='left').merge(test_count, on='class', how='left')
    if not os.path.exists(result_folder):
        os.makedirs(result_folder)
    cnt_df.to_csv(f'{result_folder}/Statistical Analysis.csv', index=False)

    categories = []
    train_dir = os.path.join(dataset_folder_path, dataset_split_images_folder, 'train')
    for cat in os.listdir(train_dir):
        categories.append(cat)
    n_classes = len(categories)
    return idx_to_class, n_classes, categories


if __name__ == '__main__':
    pass
