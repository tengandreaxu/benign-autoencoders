"""
Code credits: https://github.com/eriklindernoren/PyTorch-GAN
"""

import argparse
import os
import time
import pandas as pd
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image

from torch.utils.data import DataLoader
from torch.autograd import Variable

from context_encoders.datasets import ImageDataset
from context_encoders.models import Generator, Discriminator
from plotting.Plotter import Plotter
import torch.nn as nn
import torch.nn.functional as F
import torch


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_epochs", type=int, default=201, help="number of epochs of training"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="size of the batches"
    )

    parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
    parser.add_argument(
        "--b1",
        type=float,
        default=0.5,
        help="adam: decay of first order momentum of gradient",
    )
    parser.add_argument(
        "--b2",
        type=float,
        default=0.999,
        help="adam: decay of first order momentum of gradient",
    )
    parser.add_argument(
        "--n_cpu",
        type=int,
        default=4,
        help="number of cpu threads to use during batch generation",
    )
    parser.add_argument(
        "--latent_dim", type=int, default=100, help="dimensionality of the latent space"
    )
    parser.add_argument(
        "--img_size", type=int, default=128, help="size of each image dimension"
    )
    parser.add_argument("--mask_size", type=int, default=64, help="size of random mask")
    parser.add_argument(
        "--channels", type=int, default=3, help="number of image channels"
    )
    parser.add_argument(
        "--sample_interval",
        type=int,
        default=500,
        help="interval between image sampling",
    )
    opt = parser.parse_args()
    print(opt)

    cuda = True if torch.cuda.is_available() else False
    save_model_every = 10
    output_dir = f"results/context_encoders/z={opt.latent_dim}"
    os.makedirs(output_dir, exist_ok=True)

    models_dir = os.path.join(output_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    loss_dir = os.path.join(output_dir, "losses")
    os.makedirs(loss_dir, exist_ok=True)

    test_dir = os.path.join(output_dir, "reconstructed")
    os.makedirs(test_dir, exist_ok=True)

    # Calculate output of image discriminator (PatchGAN)
    patch_h, patch_w = int(opt.mask_size / 2**3), int(opt.mask_size / 2**3)
    patch = (1, patch_h, patch_w)

    # Loss function
    adversarial_loss = torch.nn.MSELoss()
    pixelwise_loss = torch.nn.L1Loss()

    # Initialize generator and discriminator
    generator = Generator(channels=opt.channels, nz=opt.latent_dim)
    discriminator = Discriminator(channels=opt.channels)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    adversarial_loss.to(device)
    pixelwise_loss.to(device)
    generator.to(device)
    discriminator.to(device)

    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

    # Dataset loader
    transforms_ = [
        transforms.Resize((opt.img_size, opt.img_size), Image.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
    dataloader = DataLoader(
        ImageDataset("data/original_celebA/celebA", transforms_=transforms_),
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
    )
    test_dataloader = DataLoader(
        ImageDataset(
            "data/original_celebA/celebA", transforms_=transforms_, mode="val"
        ),
        batch_size=12,
        shuffle=True,
        num_workers=1,
    )

    # Optimizers
    optimizer_G = torch.optim.Adam(
        generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2)
    )
    optimizer_D = torch.optim.Adam(
        discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2)
    )

    def save_sample(batches_done):
        samples, masked_samples, i = next(iter(test_dataloader))
        samples = samples.to(device)
        masked_samples = masked_samples.to(device)
        i = i[0].item()  # Upper-left coordinate of mask
        # Generate inpainted image
        gen_mask = generator(masked_samples)
        filled_samples = masked_samples.clone()
        filled_samples[:, :, i : i + opt.mask_size, i : i + opt.mask_size] = gen_mask
        # Save sample

        sample = torch.cat((masked_samples.data, filled_samples.data, samples.data), -2)
        save_image(
            sample,
            os.path.join(test_dir, "%d.png" % batches_done),
            nrow=6,
            normalize=True,
        )

    # ----------
    #  Training
    # ----------
    D_losses = []
    G_adv_losses = []
    pixel_losses = []
    plotter = Plotter()
    for epoch in range(opt.n_epochs):
        d_loss_epoch = 0
        g_adv_loss_epoch = 0
        pix_loss_epoch = 0

        start = time.monotonic()
        for i, (imgs, masked_imgs, masked_parts) in enumerate(dataloader):
            # Adversarial ground truths

            valid = Variable(
                torch.Tensor(imgs.shape[0], *patch).fill_(1.0), requires_grad=False
            ).to(device)
            fake = Variable(
                torch.Tensor(imgs.shape[0], *patch).fill_(0.0), requires_grad=False
            ).to(device)

            # Configure input
            imgs = imgs.to(device)
            masked_imgs = masked_imgs.to(device)
            masked_parts = masked_parts.to(device)

            # -----------------
            #  Train Generator
            # -----------------

            optimizer_G.zero_grad()

            # Generate a batch of images
            gen_parts = generator(masked_imgs)

            # Adversarial and pixelwise loss
            g_adv = adversarial_loss(discriminator(gen_parts), valid)
            g_pixel = pixelwise_loss(gen_parts, masked_parts)
            # Total loss
            g_loss = 0.001 * g_adv + 0.999 * g_pixel

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()

            # Measure discriminator's ability to classify real from generated samples
            real_loss = adversarial_loss(discriminator(masked_parts), valid)
            fake_loss = adversarial_loss(discriminator(gen_parts.detach()), fake)
            d_loss = 0.5 * (real_loss + fake_loss)

            d_loss.backward()
            optimizer_D.step()

            d_lo = d_loss.item()
            g_ad = g_adv.item()
            pixel = g_pixel.item()
            d_loss_epoch += d_lo
            g_adv_loss_epoch += g_ad
            pix_loss_epoch += pixel

            # Generate sample at sample interval
            batches_done = epoch * len(dataloader) + i
            if batches_done % opt.sample_interval == 0:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G adv: %f, pixel: %f]"
                    % (
                        epoch,
                        opt.n_epochs,
                        i,
                        len(dataloader),
                        d_loss_epoch / ((i + 1) * opt.batch_size),
                        g_adv_loss_epoch / ((i + 1) * opt.batch_size),
                        pix_loss_epoch / ((i + 1) * opt.batch_size),
                    )
                )

                save_sample(batches_done)
        end = time.monotonic()
        print(f"====> Epoch time: {end-start:.4f}")
        D_losses.append(d_loss_epoch / ((i + 1) * opt.batch_size))
        G_adv_losses.append(g_adv_loss_epoch / ((i + 1) * opt.batch_size))
        pixel_losses.append(pix_loss_epoch / ((i + 1) * opt.batch_size))
        if epoch % save_model_every == 0:
            torch.save(
                generator.state_dict(), os.path.join(models_dir, f"{epoch}_netG.pt")
            )
            torch.save(
                discriminator.state_dict(), os.path.join(models_dir, f"{epoch}_netD.pt")
            )
            df = {
                "d_loss": D_losses,
                "g_adv_loss": G_adv_losses,
                "pixel_loss": pixel_losses,
            }
            df = pd.DataFrame(df)
            df.to_csv(os.path.join(loss_dir, f"{epoch}_history.csv"), index=False)

            plotter.plot_single_curve(
                x=range(len(D_losses)),
                y=D_losses,
                xlabel="Training Step",
                ylabel="D Loss",
                file_name=os.path.join(loss_dir, f"{epoch}_d_loss.png"),
                title="",
                label="",
                grid=False,
            )
            plotter.plot_single_curve(
                x=range(len(G_adv_losses)),
                y=G_adv_losses,
                xlabel="Training Step",
                ylabel="G adv Loss",
                file_name=os.path.join(loss_dir, f"{epoch}_g_adv_loss.png"),
                title="",
                label="",
                grid=False,
            )
            plotter.plot_single_curve(
                x=range(len(pixel_losses)),
                y=pixel_losses,
                xlabel="Training Step",
                ylabel="Pixel Losses",
                file_name=os.path.join(loss_dir, f"{epoch}_pixel_losses.png"),
                title="",
                label="",
                grid=False,
            )
