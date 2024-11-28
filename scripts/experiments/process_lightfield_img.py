import os
import numpy as np
from PIL import Image
from skimage.transform import resize

img_name = "bunny"

folder_path = f"./data/light_field/{img_name}"
grid_size = 17
image_files = sorted(os.listdir(folder_path))

images = []
for file in image_files:
    image_path = os.path.join(folder_path, file)
    img = Image.open(image_path).convert("RGB")
    images.append(np.array(img))

images = np.stack(images)
tensor = images.reshape(grid_size, grid_size, 1024, 1024, 3).transpose(2, 3, 4, 0, 1)
tensor = tensor[:, :, :, ::2, ::2]  # reducing tensor size


# Resize pics from 1024x1024 to 60x40 as in Paper
h, w, c, u, v = tensor.shape
Z = np.zeros((40, 60, c, u, v), dtype=tensor.dtype)

for i in range(u):
    for j in range(v):
        Z[:, :, :, i, j] = resize(tensor[:, :, :, i, j], (40, 60), anti_aliasing=True, preserve_range=True)

selected_view = Z[:, :, :, 0, 0]
# Image.fromarray(selected_view.astype(np.uint8)).save("./resized_view.png")
np.save(f"./data/light_field/tensors/{img_name}.npy", Z)