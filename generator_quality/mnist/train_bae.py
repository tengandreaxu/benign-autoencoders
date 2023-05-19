import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pandas as pd
from torchvision.utils import save_image
from models.variational_autoencoders.MNISTVAE import VAE
from generator_quality.DatasetsTorch import DatasetsTorch
from plotting.Plotter import Plotter
from utils.torch_utils import accuracy, get_device, set_seed
from models.discriminators.MNISTTorch import Discriminator
import logging

logging.basicConfig(level=logging.INFO)


Z_DIMS = [
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    15,
    20,
    30,
    50,
    60,
    70,
    100,
    200,
    400,
    600,
    784,
    1000,
    1500,
    2000,
]


def train_bvae(
    x_train,
    y_train,
    x_test,
    y_test,
    x_train_noisy,
    x_test_noisy,
    batch_size,
    z_dim,
    epochs,
    criterion,
    output_dir,
    f_parameter_file,
):
    run_output = os.path.join(output_dir, f"z={z_dim}")
    os.makedirs(run_output, exist_ok=True)
    # Models
    vae = VAE(x_dim=784, h_dim1=512, h_dim2=256, z_dim=z_dim)
    vae.to(device)
    netD = Discriminator().to(device)
    netD.to(device)
    netD.load_state_dict(
        torch.load(f_parameter_file, map_location=torch.device(device))
    )

    # Initialize Optimizers
    optimizerVAE = optim.Adam(vae.parameters(), lr=0.001, betas=(0.5, 0.999))
    test_rec_losses = []

    netD.eval()
    train_classifier_accuracies = []
    test_classifier_accuracies = []
    for epoch in range(1, epochs + 1):
        vae.train()

        train_rec_loss = 0
        train_accuracies = []
        batches = x_train.shape[0] // batch_size
        for i in range(batches):
            start_idx = i * batch_size
            end_idx = (i + 1) * batch_size

            # Train encoder on adversarial loss
            # log(f(x)) + (1 - log(f(D(E(x)))))
            optimizerVAE.zero_grad()
            real_images = x_train[start_idx:end_idx]
            real_images = real_images[:, None, :, :]

            noisy_images = x_train_noisy[start_idx:end_idx]
            noisy_images = noisy_images[:, None, :, :]

            # Fake
            fake_images, mu, log_var = vae(noisy_images)
            label = y_train[start_idx:end_idx]
            output = netD(fake_images.view(batch_size, 1, 28, 28))
            errD_fake = criterion(output, label)

            rec_loss = F.binary_cross_entropy(
                fake_images, real_images.view(-1, 784), reduction="sum"
            )

            tot_error = errD_fake + rec_loss
            tot_error.backward()
            optimizerVAE.step()

            train_rec_loss += rec_loss.item()

            acc = accuracy(output, label)
            train_accuracies.append(acc)

        train_accuracy = np.mean(train_accuracies)
        with torch.no_grad():
            fake_images, mu, log_var = vae(x_test_noisy[:, None, :, :])
            output = netD(
                fake_images.view(x_test.shape[0], 1, 28, 28)
            )  # .view(-1, 784))
            test_accuracy = accuracy(output, y_test)

            test_rec_loss = F.binary_cross_entropy(
                fake_images, x_test.view(-1, 784), reduction="sum"
            )
        train_classifier_accuracies.append(train_accuracy)
        test_classifier_accuracies.append(test_accuracy)
        test_rec_losses.append(test_rec_loss.item() / x_test.shape[0])
        print(
            "====> Epoch: {} Train Reconstruction loss: {:.4f} \t Test Rec Loss: {:.4f} \t Train Accuracy: {:.4f} \t Test Accuracy: {:.4f}".format(
                epoch,
                train_rec_loss / x_train.shape[0],
                test_rec_loss / x_test.shape[0],
                train_accuracy,
                test_accuracy,
            )
        )

        # Plot
        vae.eval()
        with torch.no_grad():
            fake_images, mu, log_var = vae(x_test[:64])

            save_image(
                fake_images.view(64, 1, 28, 28),
                os.path.join(run_output, f"generated_image={epoch}.png"),
            )
    save_image(
        x_test[:64][:, None, :, :],
        os.path.join(run_output, f"true_x_test.png"),
    )
    save_image(
        x_test_noisy[:64][:, None, :, :],
        os.path.join(run_output, f"noised_x_test.png"),
    )
    with torch.no_grad():
        output = netD(x_test_noisy[:, None, :, :])
        test_accuracy = accuracy(output, y_test)

        logging.info(f"W test error on noisy data: {test_accuracy:.4f}")
    df = pd.DataFrame(
        {
            "test_rec_loss": test_rec_losses,
            "test_accuracy": test_classifier_accuracies,
            "train_accuracy": train_classifier_accuracies,
        }
    )
    df.to_csv(os.path.join(run_output, "history.csv"), index=False)
    plotter = Plotter()
    max_test_accuracy = np.max(test_classifier_accuracies)
    min_rec_loss = np.min(test_rec_losses)
    plotter.plot_multiple_curves_lists(
        xs=range(1, epochs + 1),
        ys=[
            (train_classifier_accuracies, "Train Accuracy"),
            (
                test_classifier_accuracies,
                f"Test Accuracy, {max_test_accuracy*100:.2f}%",
            ),
        ],
        xlabel="# Epochs",
        ylabel="Accuracy (%)",
        file_name=os.path.join(run_output, "accuracy.png"),
        title="",
    )

    plotter.plot_single_curve(
        x=range(1, epochs + 1),
        y=test_rec_losses,
        xlabel="# Epochs",
        ylabel="Reconstruction Loss",
        file_name=os.path.join(run_output, "test_rec_loss.png"),
        label=f"min loss ${min_rec_loss:.2f}$",
        title="",
    )


if __name__ == "__main__":
    device = get_device()
    set_seed(1234)
    dm = DatasetsTorch()
    dataset = "mnist"
    for noise_factor in [0.5, 1]:
        x_train, y_train, x_test, y_test, x_train_noisy, x_test_noisy = dm.get_dataset(
            dataset, noise_factor=noise_factor
        )
        batch_size = 32
        criterion = nn.CrossEntropyLoss()
        epochs = 20

        output_dir = f"results/ushape_classification/{dataset}/joint_noise={noise_factor}_epochs={epochs}"
        os.makedirs(output_dir, exist_ok=True)

        f_parameter_file = os.path.join(
            f"results/ushape_classification/{dataset}/Ws", f"best_f_{epochs}.pt"
        )
        # We need a W.
        if not os.path.exists(f_parameter_file):
            logging.error("W not found, train it with experiments.train_Ws")
            exit(1)

        for z_dim in Z_DIMS:
            train_bvae(
                x_train,
                y_train,
                x_test,
                y_test,
                x_train_noisy,
                x_test_noisy,
                batch_size,
                z_dim,
                epochs,
                criterion,
                output_dir,
                f_parameter_file,
            )
