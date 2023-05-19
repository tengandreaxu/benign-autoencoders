"""
Simple script to get a ~99% accuracy discriminator for MNIST
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from generator_quality.DatasetsTorch import DatasetsTorch
from generator_quality.torch_utils import accuracy, get_device, set_seed
from generator_quality.fmnist.FMNIST import Discriminator
from plotting.Plotter import Plotter


def train_classifier(
    x_train,
    y_train,
    x_test,
    y_test,
    batch_size,
    epochs,
    criterion,
    run_output,
    f_parameter_file,
):
    device = get_device()
    netD = Discriminator().to(device)
    netD.to(device)

    # Initialize Optimizers
    learning_rate = 0.005
    optimizerD = torch.optim.RMSprop(netD.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizerD, "min", patience=5, cooldown=2
    )
    # optimizerD = optim.Adam(netD.parameters(), lr=0.001)

    # Test Loss to save best result
    test_best_classifier_accuracy = 0
    model_parameters = None

    test_classifier_accuracies = []
    train_classifier_accuracies = []
    for epoch in range(1, epochs + 1):
        netD.train()
        train_accuracies = []
        batches = x_train.shape[0] // batch_size
        for i in range(batches):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size

            # Discriminator error on Real images
            optimizerD.zero_grad()
            real_images = x_train[start_idx:end_idx]
            real_images = real_images[:, None, :, :]

            output = netD(real_images)

            label = y_train[start_idx:end_idx]
            errD_real = criterion(output, label)
            errD_real.backward()

            optimizerD.step()
            train_accuracies.append(accuracy(output, label))
            if not (i % 100):
                scheduler.step(errD_real)

        netD.eval()
        with torch.no_grad():
            output = netD(x_test[:, None, :, :])
            test_accuracy = accuracy(output, y_test)
        train_accuracy = np.mean(train_accuracies)
        print(
            "====> Epoch: {} Train Accuracy: {:.4f} \t Test Accuracy: {:.4f}".format(
                epoch,
                train_accuracy,
                test_accuracy,
            )
        )

        if test_accuracy > test_best_classifier_accuracy:
            test_best_classifier_accuracy = test_accuracy
            model_parameters = netD.state_dict()
        test_classifier_accuracies.append(test_accuracy)
        train_classifier_accuracies.append(train_accuracy)

    torch.save(model_parameters, f_parameter_file)

    df = pd.DataFrame(
        {
            "test_accuracy": test_classifier_accuracies,
            "train_accuracy": train_classifier_accuracies,
        }
    )
    df.to_csv(os.path.join(run_output, f"history.csv"), index=False)
    plotter = Plotter()

    max_accuracy = np.max(test_classifier_accuracies)

    plotter.plot_multiple_curves_lists(
        xs=range(1, epochs + 1),
        ys=[
            (train_classifier_accuracies, "Train Accuracy"),
            (test_classifier_accuracies, f"Test Accuracy, {max_accuracy*100:.2f}%"),
        ],
        xlabel="# Epochs",
        ylabel="Accuracy (%)",
        file_name=os.path.join(run_output, f"accuracy.png"),
        title="",
    )

    return model_parameters


if __name__ == "__main__":
    """
    W should be a non-convex function capable
    to discriminate between classes
    """

    device = get_device()
    set_seed(1234)
    dm = DatasetsTorch()
    dataset = "fmnist"
    # W should be trained without noise.
    x_train, y_train, x_test, y_test, _, _ = dm.get_dataset(dataset, noise_factor=0)
    batch_size = 120
    criterion = nn.CrossEntropyLoss()
    epochs = 100
    output_dir = f"results/ushape_classification/{dataset}/Ws"
    os.makedirs(output_dir, exist_ok=True)
    f_parameter_file = os.path.join(output_dir, f"best_f_{epochs}.pt")

    # Train Gan and fix f with full latent space
    f_parameters = train_classifier(
        x_train,
        y_train,
        x_test,
        y_test,
        batch_size,
        criterion=criterion,
        run_output=output_dir,
        epochs=epochs,
        f_parameter_file=f_parameter_file,
    )
