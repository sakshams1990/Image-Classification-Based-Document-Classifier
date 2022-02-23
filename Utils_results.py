from matplotlib import pyplot as plt


def plot_losses_history(history, path, train_loss, valid_loss):
    plt.figure(figsize=(10, 10))
    for c in [train_loss, valid_loss]:
        plt.plot(history[c], label=c)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Average Losses')
    plt.title('Training and Validation Losses')
    plt.savefig(f"{path}/Epoch vs Loss.jpg")
    plt.close()
    print("Training Loss vs Validation Loss graph plotted ")


def plot_accuracy_history(history, path, train_accuracy, valid_accuracy):
    plt.figure(figsize=(10, 10))
    for c in [train_accuracy, valid_accuracy]:
        plt.plot(history[c], label=c)
    plt.legend()
    plt.xlabel('Epoch')
    plt.ylabel('Average Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.savefig(f"{path}/Epoch vs Accuracy.jpg")
    plt.close()
    print("Training Accuracy vs Validation Accuracy graph plotted ")


if __name__ == '__main__':
    pass
