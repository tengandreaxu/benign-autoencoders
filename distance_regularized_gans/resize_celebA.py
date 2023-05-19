from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import cv2

root = "data/celebA/"
save_root = "data/resized_celebA/celebA"
resize_size = 64


os.makedirs(save_root)
img_list = os.listdir(root)

for i in tqdm(range(len(img_list))):
    img = plt.imread(os.path.join(root, img_list[i]))

    img = cv2.resize(img, (resize_size, resize_size))
    plt.imsave(fname=os.path.join(save_root, img_list[i]), arr=img)
