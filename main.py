from sklearn.metrics import confusion_matrix

from get_values_from_config_file import result_folder
from model_building import run_training
from pdf_to_image import move_files_from_raw_to_input_folder
from Split_images_train_val_test import split_images_into_train_test_val
from dataset_loader import get_datasets_dataloaders_for_model
from model_predictions_for_test_images import get_predictions, show_confusion_matrix

from statistical_analysis import perform_statistical_analysis


def model_training_testing():
    # Moving files from raw folder to Images Folder
    move_files_from_raw_to_input_folder()

    # Splitting Images into train, val and test
    split_images_into_train_test_val()

    # Dataset and dataloaders created for pytorch model
    train_dataset, train_dataload, val_dataset, val_dataload, \
        test_dataset, test_dataload = get_datasets_dataloaders_for_model()

    # Class Index, number of classes and class names are returned
    index_to_class, num_classes, category_names = perform_statistical_analysis(train_dataset, val_dataset, test_dataset)

    # CNN Model is created
    base_model = run_training(num_classes, train_dataset, train_dataload, val_dataset, val_dataload)
    y_pred, y_test = get_predictions(base_model, test_dataload)
    cm = confusion_matrix(y_test, y_pred)
    # Create confusion matrix and save the image
    show_confusion_matrix(cm, category_names, result_folder)


if __name__ == '__main__':
    model_training_testing()
