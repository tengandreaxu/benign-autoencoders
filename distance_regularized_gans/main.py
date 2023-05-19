"""
Code source: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html
We study the importance of the latent dimension z_dim
"""
import pandas as pd
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms

from torchvision.utils import save_image

import matplotlib.pyplot as plt

from Generator import Generator
from Discriminator import Discriminator
from Encoder import Encoder
from Autoencoder import Autoencoder


# custom weights initialization called on ``netG`` and ``netD``
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


if __name__ == "__main__":
    manualSeed = 999
    print("Random Seed: ", manualSeed)
    random.seed(manualSeed)
    torch.manual_seed(manualSeed)

    dataroot = "data/resized_celebA"
    workers = 2
    # Batch size during training
    batch_size = 128
    image_size = 64
    nc = 3
    for nz in [1, 10, 50, 100, 500, 1000]:
        ngf = 64
        ndf = 64
        num_epochs = 101
        lr = 0.0002
        beta1 = 0.5
        ngpu = 1

        dataset = dset.ImageFolder(
            root=dataroot,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ]
            ),
        )
        # Create the dataloader
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, num_workers=workers
        )

        # Decide which device we want to run on
        device = torch.device(
            "cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu"
        )

        netG = Generator(ngpu, ngf, nz, nc).to(device)
        netD1 = Discriminator(ngpu, ndf, nc).to(device)
        encoder = Encoder(ngpu, ngf, nz, nc).to(device)
        netG.apply(weights_init)
        netD1.apply(weights_init)
        encoder.apply(weights_init)
        autoencoder = Autoencoder(encoder=encoder, decoder=netG)

        criterion = nn.BCELoss()
        l2_loss = nn.MSELoss()

        fixed_noise = torch.randn(64, nz, 1, 1, device=device)

        real_label = 1.0
        fake_label = 0.0

        # Setup Adam optimizers for both G and D
        optimizerD1 = optim.Adam(netD1.parameters(), lr=lr, betas=(beta1, 0.999))
        optimizerA = optim.Adam(autoencoder.parameters(), lr=lr, betas=(beta1, 0.999))

        # Lists to keep track of progress
        img_list = []
        G_losses = []
        D1_losses = []
        E_losses = []

        rec_losses = []

        print("Starting Training Loop...")
        # For each epoch
        output_dir = f"results/distance_regularized_gan_d1/z_dim={nz}"
        fake_dir = os.path.join(output_dir, "fixed_noise_evolution")
        rec_dir = os.path.join(output_dir, "reconstruction_evolution")
        history_dir = os.path.join(output_dir, "histories")
        model_dir = os.path.join(output_dir, "models")
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(fake_dir, exist_ok=True)
        os.makedirs(rec_dir, exist_ok=True)
        os.makedirs(history_dir, exist_ok=True)
        os.makedirs(model_dir, exist_ok=True)
        steps = []
        step = 0
        for epoch in range(num_epochs):
            for i, data in enumerate(dataloader, 0):
                ############################
                # (1) Update D1 network: maximize log(D(x)) + log(1 - D(G(E(x)))
                ###########################

                ## D1
                netD1.zero_grad()
                # Real
                real_cpu = data[0].to(device)
                b_size = real_cpu.size(0)
                label = torch.full(
                    (b_size,), real_label, dtype=torch.float, device=device
                )
                output = netD1(real_cpu).view(-1)
                errD_real = criterion(output, label)
                errD_real.backward()
                D_x = output.mean().item()

                # Fake
                noise = autoencoder.encode(real_cpu)
                fake = autoencoder.generate(noise.detach())

                label.fill_(fake_label)
                output = netD1(fake.detach()).view(-1)
                errD_fake = criterion(output, label)
                errD_fake.backward()

                D_G_z1 = output.mean().item()
                errD = errD_real + errD_fake
                optimizerD1.step()

                ############################
                # (2) Update G and E network: maximize log(D(G(E(x)))) + l2(x, fake)
                ###########################
                autoencoder.zero_grad()
                label.fill_(real_label)
                # fake = G(E(x))
                output = netD1(fake).view(-1)
                errG_E_fake = criterion(output, label)
                # L2 loss

                rec_loss = l2_loss(fake.view(-1), real_cpu.view(-1))

                errG_E = errG_E_fake + rec_loss
                errG_E.backward()

                D_G_z2 = output.mean().item()
                optimizerA.step()

                step += 1
                # Output training stats
                if i % 50 == 0:
                    print(
                        "[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(E(x))): %.4f / %.4f\tl2(x,G(E(x))): %.4f"
                        % (
                            epoch,
                            num_epochs,
                            i,
                            len(dataloader),
                            errD.item(),
                            errG_E.item(),
                            D_x,
                            D_G_z1,
                            D_G_z2,
                            rec_loss.item(),
                        )
                    )

                # Save Losses for plotting later
                G_losses.append(errG_E_fake.item())
                D1_losses.append(errD.item())
                rec_losses.append(rec_loss.item())

                steps.append(step)

            fake_images = netG(fixed_noise)
            for i, data in enumerate(dataloader, 0):
                real = data[0].to(device)
                break
            save_image(
                fake_images,
                os.path.join(fake_dir, f"fake_epoch={epoch}.png"),
            )
            autoencoder.eval()
            with torch.no_grad():
                reconstructed = autoencoder.generate(autoencoder.encode(real))

            whole = torch.cat([real, reconstructed], 0)
            save_image(
                whole, os.path.join(rec_dir, f"reconstruction_epooch={epoch}.png")
            )

            if epoch % 100 == 0:
                df = pd.DataFrame(
                    {
                        "steps": steps,
                        "G_losses": G_losses,
                        "D1": D1_losses,
                        "rec": rec_losses,
                    }
                )
                df.to_csv(
                    os.path.join(history_dir, f"{epoch}_history.csv"), index=False
                )
                torch.save(
                    netD1.state_dict(), os.path.join(model_dir, f"{epoch}_netD1.pt")
                )
                torch.save(
                    netG.state_dict(), os.path.join(model_dir, f"{epoch}_netG.pt")
                )
                torch.save(
                    encoder.state_dict(), os.path.join(model_dir, f"{epoch}_encoder.pt")
                )

        plt.figure(figsize=(10, 5))
        plt.title("Generator and Discriminator Loss During Training")
        plt.plot(G_losses, label="G")
        plt.plot(D1_losses, label="D")
        plt.xlabel("iterations")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(os.path.join(output_dir, "training.png"))
