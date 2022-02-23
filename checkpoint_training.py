import copy
import os
import time
from collections import defaultdict

import torch

from torch import optim
from torch.optim import lr_scheduler

from Utils_results import plot_losses_history, plot_accuracy_history
from dataset_loader import get_datasets_dataloaders_for_model
from get_values_from_config_file import num_epochs, step_size, learning_rate, beta1, beta2, eps, gamma, \
    checkpoint_folder, model_folder, result_folder
from model_building import create_model, train_epoch, eval_model, save_checkpoint
from statistical_analysis import perform_statistical_analysis


def load_checkpoint(path, num_classes):
    checkpoint = torch.load(path)
    model = create_model(num_classes)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(beta1, beta2), eps=eps)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch']
    criterion = checkpoint['loss']
    return model, optimizer, start_epoch, criterion


def resume_training_from_checkpoint(checkpoint_path):
    # Dataset and dataloaders created for pytorch model
    train_dataset, train_dataload, val_dataset, val_dataload, \
        test_dataset, test_dataload = get_datasets_dataloaders_for_model()

    # Class Index, number of classes and class names are returned
    index_to_class, num_classes, category_names = perform_statistical_analysis(train_dataset, val_dataset, test_dataset)

    model, optimizer, start_epoch, criterion = load_checkpoint(checkpoint_path, num_classes)
    start_epoch = start_epoch + 2
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    history = defaultdict(list)
    best_accuracy = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    print(f'\nResuming Model Training from epoch {start_epoch}..')
    start_time = time.time()
    for epoch in range(start_epoch, num_epochs + 1):
        print(f'\nEpoch {epoch}/{num_epochs}:')
        print('-' * 10)
        train_accuracy, train_loss = train_epoch(model, train_dataload, criterion, optimizer, scheduler,
                                                 len(train_dataset))
        print(f'Training Loss:{train_loss:.3f}\t\t Training Accuracy:{train_accuracy:.3f}')
        validation_accuracy, validation_loss = eval_model(model, val_dataload, criterion, len(val_dataset))
        print(f'Validation Loss:{validation_loss:.3f}\t Validation Accuracy:{validation_accuracy:.3f}')

        history['training_accuracy'].append(train_accuracy)
        history['training_loss'].append(train_loss)
        history['validation_accuracy'].append(validation_accuracy)
        history['validation_loss'].append(validation_loss)

        if validation_accuracy > best_accuracy:
            best_model_wts = copy.deepcopy(model.state_dict())
            best_accuracy = validation_accuracy

        # Saving model checkpoints at checkpoint folder
        checkpoint_dir = checkpoint_folder
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        checkpoint_filename = 'model_epoch_' + f'{epoch}'
        checkpoint_path = f"{checkpoint_dir}/{checkpoint_filename}.pth"
        if epoch % 2 == 0:
            save_checkpoint(model, epoch, optimizer, criterion, checkpoint_path)

    total_time = time.time() - start_time

    print(f'\nBest validation accuracy:{best_accuracy:.3f}')
    model.load_state_dict(best_model_wts)

    model_dir = model_folder
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    print(f"\nSaving best model to {model_dir} folder..")
    model_filename = 'best_model_state.pt'
    model_path = f"{model_dir}/{model_filename}"
    torch.save(model, model_path)

    plot_losses_history(history, result_folder, 'training_loss', 'validation_loss')
    plot_accuracy_history(history, result_folder, 'training_accuracy', 'validation_accuracy')

    print('Total Time taken for training:', round(total_time, 3))
    return model


if __name__ == '__main__':
    final_model = resume_training_from_checkpoint('Checkpoint/model_epoch_2.pth')
