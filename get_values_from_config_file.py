import ast
import os

import Utils_Configurations

config_file_path = os.path.join(os.getcwd(), "Config.ini")
config_file = Utils_Configurations.Configuration(config_file_path)


# Read raw input folder paths from config file
def get_raw_input_file_path_from_config():
    dataset_path = config_file.read_configuration_options('RAW INPUT PATH', 'DATASET_PATH')
    return dataset_path


# Get PDF to Image and Train, Val and Test folder from config file
def get_image_folder_names_from_config():
    final_image_folder = config_file.read_configuration_options('IMAGE FOLDER', 'FINAL_IMAGE_FOLDER', 'str')
    dataset_split_folder = config_file.read_configuration_options('IMAGE FOLDER', 'DATASET_SPLIT_FOLDER', 'str')
    raw_folder = config_file.read_configuration_options('IMAGE FOLDER', 'RAW_INPUT_FOLDER', 'str')
    return final_image_folder, dataset_split_folder, raw_folder


# Get image preprocessing parameters from config file
def get_image_dataloader_values_from_config():
    image_resize = config_file.read_configuration_options('IMAGE DATALOADER', 'IMAGE_RESIZE', 'int')
    mean = config_file.read_configuration_options('IMAGE DATALOADER', 'MEAN', 'str')
    image_mean = ast.literal_eval(mean)
    std = config_file.read_configuration_options('IMAGE DATALOADER', 'STD', 'str')
    image_std = ast.literal_eval(std)
    shuffle = config_file.read_configuration_options('IMAGE DATALOADER', 'SHUFFLE', 'bool')
    degree = config_file.read_configuration_options('IMAGE DATALOADER', 'ROTATION_DEGREE', 'int')
    num_workers = config_file.read_configuration_options('IMAGE DATALOADER', 'NUM_WORKERS', 'int')
    batch_size = config_file.read_configuration_options('IMAGE DATALOADER', 'BATCH_SIZE', 'int')
    return image_resize, image_mean, image_std, shuffle, degree, num_workers, batch_size


# Get train,val,test ratio split from config file
def get_train_val_test_split_ratio():
    train = config_file.read_configuration_options('TRAIN VAL TEST SPLIT RATIO', 'train', 'float')
    val = config_file.read_configuration_options('TRAIN VAL TEST SPLIT RATIO', 'val', 'float')
    test = config_file.read_configuration_options('TRAIN VAL TEST SPLIT RATIO', 'test', 'float')
    return train, val, test


# Get the supported image extensions from config file
def get_supported_image_extensions_from_config():
    image_ext = config_file.read_configuration_options('SUPPORTED IMAGES EXTENSIONS', 'IMAGE_EXTENSION', 'str')
    image_ext = ast.literal_eval(image_ext)
    return image_ext


# Get the algorithm parameters from config file for model training
def get_model_parameters_from_config():
    num_epochs = config_file.read_configuration_options('MODEL PARAMETERS', 'EPOCHS', 'int')
    learning_rate = config_file.read_configuration_options('MODEL PARAMETERS', 'LEARNING_RATE', 'float')
    momentum = config_file.read_configuration_options('MODEL PARAMETERS', 'MOMENTUM', 'float')
    gamma = config_file.read_configuration_options('MODEL PARAMETERS', 'GAMMA', 'float')
    device = config_file.read_configuration_options('MODEL PARAMETERS', 'DEVICE', 'str')
    step_size = config_file.read_configuration_options('MODEL PARAMETERS', 'STEP_SIZE', 'int')
    beta1 = config_file.read_configuration_options('MODEL PARAMETERS', 'BETA1', 'float')
    beta2 = config_file.read_configuration_options('MODEL PARAMETERS', 'BETA2', 'float')
    eps = config_file.read_configuration_options('MODEL PARAMETERS', 'EPSILON', 'float')
    return num_epochs, learning_rate, momentum, gamma, device, step_size, beta1, beta2, eps


def get_result_folders_from_config():
    checkpoint_folder = config_file.read_configuration_options('RESULT FOLDER', 'CHECKPOINT_FOLDER', 'str')
    model_folder = config_file.read_configuration_options('RESULT FOLDER', 'MODEL_FOLDER', 'str')
    result_folder = config_file.read_configuration_options('RESULT FOLDER', 'RESULT_FOLDER', 'str')
    return checkpoint_folder, model_folder, result_folder


def get_poppler_path_from_config():
    poppler_path = config_file.read_configuration_options('POPPLER PATH', 'poppler_path', 'str')
    return poppler_path


dataset_folder_path = get_raw_input_file_path_from_config()
final_images_folder, dataset_split_images_folder, raw_files_folder = get_image_folder_names_from_config()
image_resize, image_mean, image_std, shuffle_images, rot_degree, num_workers, batch_size = \
    get_image_dataloader_values_from_config()
train, val, test = get_train_val_test_split_ratio()
image_extension = get_supported_image_extensions_from_config()
num_epochs, learning_rate, momentum, gamma, device, step_size, beta1, beta2, eps = \
    get_model_parameters_from_config()
checkpoint_folder, model_folder, result_folder = get_result_folders_from_config()
poppler_path = get_poppler_path_from_config()

if __name__ == '__main__':
    pass
