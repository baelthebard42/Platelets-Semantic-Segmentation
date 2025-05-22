import glob
import json
import os
import cv2

category_ids = {
    "Alpha":1,
    "Cells":2,
    "Mito":3,
    "Vessels":4
}

MASK_EXTEN = 'png'
ORIGINAL_EXTEN = "png"
image_id = 0
annotation_id = 0

