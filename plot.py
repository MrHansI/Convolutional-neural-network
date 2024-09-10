import matplotlib as plt

def train_n_validation_plot(train_loss, test_loss, model_name, close=True):
    if close:
        plt.close('all')
    plt.figure(figsize=[10, 5])
    plt.title(model_name)
    epochs = [k for k in range(1, len(train_loss) + 1)]

    plt.xlabel("Epoch")
    plt.ylabel("Loss value")

    plt.grid()

    plt.plot(epochs, train_loss, 'g-', label='Train loss')
    plt.plot(epochs, test_loss, 'r--', label='Test loss')
    plt.ion()
    plt.legend()
    plt.show()
    plt.pause(1.0)