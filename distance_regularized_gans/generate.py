import os
import random
import torch
import torch.nn.parallel
import torch.utils.data

from torchvision.utils import save_image
from tqdm import tqdm

from Generator import Generator
from Encoder import Encoder

from Autoencoder import Autoencoder


if __name__ == "__main__":
    manualSeed = 999
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    batch_size = 100
    nc = 3
    ngf = 64
    ndf = 64
    num_epochs = 100

    ngpu = 1
    # Generate 30k images
    number_of_images = 30000
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

        batches = number_of_images // batch_size

        generated_images = os.path.join(output_dir, "generated")
        os.makedirs(generated_images, exist_ok=True)

        netG.eval()
        number = 0
        with torch.no_grad():
            for i in tqdm(range(batches)):
                noise = torch.randn(batch_size, nz, 1, 1, device=device)
                generated = netG(noise)

                for image in generated:
                    save_image(image, os.path.join(generated_images, f"{number}.jpg"))
                    number += 1
