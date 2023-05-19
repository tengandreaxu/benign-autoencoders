import torch
from torchvision import datasets, transforms
import logging

logging.basicConfig(level=logging.INFO)


class DatasetsTorch:
    def __init__(self):
        self.support_datasets = ["mnist", "fmnist"]
        self.logger = logging.getLogger("DatasetManager")

    def get_dataset(self, dataset_name: str, noise_factor: float):
        if dataset_name == "mnist":
            self.logger.info("Loading MNIST")
            trainset = datasets.MNIST(
                root="data",
                train=True,
                download=True,
                transform=transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize(
                            0.5,
                            0.5,
                        ),
                    ]
                ),
            )
            testset = datasets.MNIST(
                root="data",
                train=False,
                download=True,
                transform=transforms.Compose(
                    [
                        transforms.ToTensor(),
                        transforms.Normalize(
                            0.5,
                            0.5,
                        ),
                    ]
                ),
            )

        elif dataset_name == "fmnist":
            self.logger.info("Loading FMNIST")
            trainset = datasets.FashionMNIST(
                root="data",
                train=True,
                download=True,
                transform=transforms.Compose(
                    [
                        transforms.ToTensor(),
                    ]
                ),
            )
            testset = datasets.FashionMNIST(
                root="data",
                train=False,
                download=True,
                transform=transforms.Compose(
                    [
                        transforms.ToTensor(),
                    ]
                ),
            )

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        if torch.backends.mps.is_available() and not torch.cuda.is_available():
            device = "mps"
        x_train, y_train = torch.Tensor(trainset.data), torch.Tensor(trainset.targets)
        x_test, y_test = torch.Tensor(testset.data), torch.Tensor(testset.targets)

        x_train = x_train / 255.0
        x_test = x_test / 255.0
        x_train_noisy = x_train + (noise_factor * torch.randn(x_train.shape))
        x_test_noisy = x_test + (noise_factor * torch.randn(x_test.shape))

        x_train = x_train.to(torch.float32)
        x_test = x_test.to(torch.float32)
        y_train = y_train.type(torch.LongTensor)
        y_test = y_test.type(torch.LongTensor)
        x_train_noisy = x_train_noisy.to(torch.float32)
        x_test_noisy = x_test_noisy.to(torch.float32)
        self.logger.info("*" * 47)
        self.logger.info(f"Moving data to: {device}")
        self.logger.info("*" * 47)
        return (
            x_train.to(device),
            y_train.to(device),
            x_test.to(device),
            y_test.to(device),
            x_train_noisy.to(device),
            x_test_noisy.to(device),
        )
