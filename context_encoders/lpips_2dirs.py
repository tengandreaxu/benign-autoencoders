import argparse
import os
import lpips
import pandas as pd

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("-v", "--version", type=str, default="0.1")
parser.add_argument("--use_gpu", action="store_true", help="turn on flag to use GPU")

opt = parser.parse_args()


for latent_dim in [1, 10, 50, 100, 500, 1000, 4000]:
    output_dir = f"results/context_encoders/z={latent_dim}"
    os.makedirs(output_dir, exist_ok=True)
    models_dir = os.path.join(output_dir, "models")

    test_dir = os.path.join(output_dir, "tests")
    in_painted = os.path.join(output_dir, "in_painted")

    ## Initializing the model
    loss_fn = lpips.LPIPS(net="alex", version=opt.version)
    if opt.use_gpu:
        loss_fn.cuda()

    files = os.listdir(test_dir)

    df = {"file": [], "lpips": []}
    for file in files:
        if os.path.exists(os.path.join(in_painted, file)):
            # Load images
            img0 = lpips.im2tensor(
                lpips.load_image(os.path.join(test_dir, file))
            )  # RGB image from [-1,1]
            img1 = lpips.im2tensor(lpips.load_image(os.path.join(in_painted, file)))

            if opt.use_gpu:
                img0 = img0.cuda()
                img1 = img1.cuda()

            # Compute distance
            dist01 = loss_fn.forward(img0, img1)
            df["file"].append(file)
            df["lpips"].append(dist01.item())
    df = pd.DataFrame(df)
    df.to_csv(os.path.join(output_dir, "lpips.csv"), index=False)
    print(f"z:\t{latent_dim}\tlpips:\t{df.lpips.mean()}")
