import splitfolders
from get_values_from_config_file import dataset_split_images_folder, dataset_folder_path, final_images_folder
from get_values_from_config_file import train, val, test
import os


def split_images_into_train_test_val():
    input_path = os.path.join(dataset_folder_path, final_images_folder)
    output_path = os.path.join(dataset_folder_path, dataset_split_images_folder)
    splitfolders.ratio(input_path, output=output_path, seed=101, ratio=(train, val, test))


if __name__ == '__main__':
    split_images_into_train_test_val()
