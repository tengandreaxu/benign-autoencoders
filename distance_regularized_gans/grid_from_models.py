import os
import random
import torch
import torch.nn.parallel
import torch.utils.data

from torchvision.utils import save_image
from Generator import Generator
from Encoder import Encoder

from Autoencoder import Autoencoder


if __name__ == "__main__":
    """
    This guy reproduces the grid shown in the paper
    """
    manualSeed = 999
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    nc = 3
    ngf = 64
    ndf = 64
    num_epochs = 100

    ngpu = 1
    number_of_images = 10

    final_plot = torch.Tensor().cuda()
    for nz in [1, 10, 50, 100, 500, 1000]:
        # Decide which device we want to run on
        device = torch.device(
            "cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu"
        )

        netG = Generator(ngpu, ngf, nz, nc).to(device)
        encoder = Encoder(ngpu, ngf, nz, nc).to(device)

        autoencoder = Autoencoder(encoder=encoder, decoder=netG)

        output_dir = f"results/distance_regularized_gan_d1/z_dim={nz}"

        decoder_dict = os.path.join(output_dir, "models", "100_netG.pt")

        netG.load_state_dict(torch.load(decoder_dict))

        generated_images = os.path.join(output_dir, "generated")
        os.makedirs(generated_images, exist_ok=True)

        netG.eval()
        number = 0
        with torch.no_grad():
            noise = torch.randn(number_of_images, nz, 1, 1, device=device)
            generated = netG(noise)

            final_plot = torch.cat([final_plot, generated], 0)

    save_image(final_plot, "grid.png", nrow=10, normalize=True)
    save_image(final_plot, "grid.pdf", nrow=10, normalize=True)
