'converts the base path consisting of .tif images into structured dataset of train images and masks'

import os
import tifffile as tiff
from PIL import Image
import numpy as np


image_stacks = ['24-images.tif', '50-images.tif']
label_stacks = ['24-semantic.tif', '50-semantic.tif']
base_path = 'platelet-em'

output_image_dir = 'input/train_images'
output_mask_dirs = {
    'alpha': 'input/train_masks/alpha',
    'cell': 'input/train_masks/cell',
    'mito': 'input/train_masks/mito',
    'vessels': 'input/train_masks/vessels',
}


label_map = {
    'cell': 1,
    'mito': 2,
    'alpha': 3,
    'vessels': 4
}


os.makedirs(output_image_dir, exist_ok=True)
for folder in output_mask_dirs.values():
    os.makedirs(folder, exist_ok=True)


slice_idx = 0
for img_file, lbl_file in zip(image_stacks, label_stacks):
    img_stack = tiff.imread(os.path.join(base_path, 'images', img_file))
    lbl_stack = tiff.imread(os.path.join(base_path, 'labels-semantic', lbl_file))

    for i in range(len(img_stack)):
        image_path = os.path.join(output_image_dir, f'slice_{slice_idx:03d}.png')
        img = img_stack[i].astype(np.float32)
        img -= img.min()
        img /= img.max() if img.max() > 0 else 1
        img *= 255.0
        img = img.astype(np.uint8)
        Image.fromarray(img).save(image_path)

       
        for class_name, class_id in label_map.items():
            mask = (lbl_stack[i] == class_id).astype(np.uint8) * 255
            mask_path = os.path.join(output_mask_dirs[class_name], f'slice_{slice_idx:03d}.png')
            Image.fromarray(mask).save(mask_path)

        slice_idx += 1
