import copy
import os
import time
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import models

from Utils_results import plot_losses_history, plot_accuracy_history
from get_values_from_config_file import checkpoint_folder, model_folder, result_folder
from get_values_from_config_file import num_epochs, step_size, learning_rate, gamma, device, beta1, beta2, eps


def create_model(num_classes):
    model = models.resnet50(pretrained=True)
    n_features = model.fc.in_features
    model.fc = nn.Linear(n_features, num_classes)
    return model.to(device)


def train_epoch(model, data_loader, loss_function, optimizer, scheduler, n_examples):
    model = model.train()
    losses = []
    correct_predictions = 0
    for inputs, labels in data_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        _, preds = torch.max(outputs, dim=1)
        loss = loss_function(outputs, labels)
        correct_predictions += torch.sum(preds == labels)
        losses.append(loss.item())
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    scheduler.step()
    return correct_predictions.double() / n_examples, np.mean(losses)


def eval_model(model, data_loader, loss_function, n_examples):
    model = model.eval()
    losses = []
    correct_predictions = 0

    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, dim=1)
            loss = loss_function(outputs, labels)
            correct_predictions += torch.sum(preds == labels)
            losses.append(loss.item())
    return correct_predictions.double() / n_examples, np.mean(losses)


def train_model(model, train_dataset, train_dataload, val_dataset, val_dataload):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(beta1, beta2), eps=eps)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    loss_fn = nn.CrossEntropyLoss().to(device)
    history = defaultdict(list)
    best_accuracy = 0.0
    best_model_wts = copy.deepcopy(model.state_dict())
    print('\nStarting Model Training..')
    start_time = time.time()
    for epoch in range(1, num_epochs + 1):
        print(f'\nEpoch {epoch}/{num_epochs}:')
        print('-' * 10)
        train_accuracy, train_loss = train_epoch(model, train_dataload, loss_fn, optimizer, scheduler,
                                                 len(train_dataset))
        print(f'Training Loss:{train_loss:.3f}\t\t Training Accuracy:{train_accuracy:.3f}')
        validation_accuracy, validation_loss = eval_model(model, val_dataload, loss_fn, len(val_dataset))
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
            save_checkpoint(model, epoch, optimizer, loss_fn, checkpoint_path)

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

    print('Total Time taken for training:', round(total_time, 3), 'seconds')
    return model


# Saving a checkpoint
def save_checkpoint(model, epoch, optimizer, loss, path):
    """
    :param model: model to be saved
    :param epoch: epoch at which the model gets saved
    :param optimizer: optimizer to compute the gradients
    :param loss: loss function for the model
    :param path: path for the checkpoint to be saved
    :return: None
    """

    torch.save({'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
                }, path)


def run_training(num_classes, train_dataset, train_dataload, val_dataset, val_dataload):
    base_model = create_model(num_classes)
    final_model = train_model(base_model, train_dataset, train_dataload, val_dataset, val_dataload)
    print("\nModel Training Completed!!")
    return final_model


if __name__ == '__main__':
    pass
