import os

import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm
from torch.utils.data import DataLoader


from context_encoders.datasets import ImageDataset
from context_encoders.models import Generator
import torch


if __name__ == "__main__":
    mask_size = 64
    img_size = 128
    save_model_every = 10
    # Dataset loader
    transforms_ = [
        transforms.Resize((img_size, img_size), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]

    test_dataloader = DataLoader(
        ImageDataset("data/celebA", transforms_=transforms_, mode="val"),
        batch_size=10,
        shuffle=False,
        num_workers=1,
    )

    in_painting_grid = torch.Tensor().to("cuda:0")
    for latent_dim in [1, 10, 50, 100, 500, 1000, 4000]:
        output_dir = f"results/context_encoders/z={latent_dim}"
        os.makedirs(output_dir, exist_ok=True)
        models_dir = os.path.join(output_dir, "models")

        generator = Generator(channels=3, nz=latent_dim)
        generator.load_state_dict(torch.load(os.path.join(models_dir, "150_netG.pt")))
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        generator.to(device)
        generator.eval()

        with torch.no_grad():
            samples, masked_samples, i = next(iter(test_dataloader))
            samples = samples.to(device)
            masked_samples = masked_samples.to(device)
            i = i[0].item()  # Upper-left coordinate of mask
            # Generate inpainted image
            gen_mask = generator(masked_samples)
            filled_samples = masked_samples.clone()
            filled_samples[:, :, i : i + mask_size, i : i + mask_size] = gen_mask

            if in_painting_grid.size(0) == 0:
                in_painting_grid = torch.cat([in_painting_grid, masked_samples])
            in_painting_grid = torch.cat([in_painting_grid, filled_samples], 0)
            if latent_dim == 4000:
                in_painting_grid = torch.cat([in_painting_grid, samples], 0)
    save_image(in_painting_grid, "grid_context.pdf", nrow=10, normalize=True)
    save_image(in_painting_grid, "grid_context.png", nrow=10, normalize=True)
