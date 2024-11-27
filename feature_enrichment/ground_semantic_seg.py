import numpy as np
from transformers import Mask2FormerImageProcessor, Mask2FormerForUniversalSegmentation
from PIL import Image
import torch
import os
from tqdm import tqdm

# we define the color palette for the model

color_palette = [
    (196, 128, 128),  # Bird - Light Red
    (196, 128, 128),  # Ground Animal - Light Red
    (0, 0, 255),      # Curb - Blue
    (0, 128, 128),    # Fence - Teal
    (0, 196, 128),    # Guard Rail - Medium Green
    (32, 32, 128),    # Barrier -
    (32, 32, 128),    # Wall -
    (0, 0, 0),        # Bike Lane - Black
    (0, 0, 0),        # Crosswalk - Plain - Black
    (0, 0, 255),      # Curb Cut - Blue
    (0, 0, 0),        # Parking - Black
    (255, 255, 196),  # Pedestrian Area - Light Yellow
    (64, 64, 0),      # Rail Track - Olive
    (0, 0, 0),        # Road - Black
    (0, 0, 0),        # Service Lane - Black
    (255, 255, 0),    # Sidewalk - Yellow
    (0, 64, 0),       # Bridge - Dark Green
    (128, 128, 128),  # Building - Gray
    (0, 128, 0),      # Tunnel - Green
    (128, 0, 128),    # Person - Purple
    (128, 0, 128),    # Bicyclist - Purple
    (128, 0, 128),    # Motorcyclist - Purple
    (128, 0, 128),    # Other Rider - Purple
    (255, 255, 255),  # Lane Marking - Crosswalk - White
    (255, 255, 255),  # Lane Marking - General - White
    (128, 64, 0),     # Mountain - Brown
    (240, 230, 140),  # Sand - Khaki
    (135, 206, 235),  # Sky - Sky Blue
    (221, 242, 249),  # Snow - Light Blue
    (128, 128, 0),    # Terrain - Olive
    (0, 255, 0),      # Vegetation - Bright Green
    (0, 128, 255),    # Water - Medium Blue
    (128, 0, 0),      # Banner - Maroon
    (255, 192, 203),  # Bench - Pink
    (255, 192, 203),  # Bike Rack - Pink
    (128, 0, 0),      # Billboard - Maroon
    (196, 196, 0),    # Catch Basin - Yellow-Green
    (255, 192, 203),  # CCTV Camera - Pink
    (255, 192, 203),  # Fire Hydrant - Pink
    (255, 192, 203),  # Junction Box - Pink
    (255, 192, 203),  # Mailbox - Pink
    (196, 196, 0),    # Manhole - Yellow-Green
    (255, 192, 203),  # Phone Booth - Pink
    (0, 0, 0),        # Pothole - Black
    (255, 0, 255),    # Street Light - Fuchsia
    (255, 0, 255),    # Pole - Fuchsia
    (255, 0, 255),    # Traffic Sign Frame - Fuchsia
    (255, 0, 255),    # Utility Pole - Fuchsia
    (255, 0, 255),    # Traffic Light - Fuchsia
    (255, 0, 255),    # Traffic Sign (Back) - Fuchsia
    (255, 0, 255),    # Traffic Sign (Front) - Fuchsia
    (255, 192, 203),  # Trash Can - Pink
    (255, 128, 0),    # Bicycle - Orange
    (196, 64, 0),     # Boat - Dark Orange
    (255, 0, 0),      # Bus - Red
    (255, 0, 0),      # Car - Red
    (255, 0, 0),      # Caravan - Red
    (255, 128, 0),    # Motorcycle - Orange
    (196, 0, 64),     # On Rails - Dark Red
    (255, 0, 0),      # Other Vehicle - Red
    (255, 0, 0),      # Trailer - Red
    (255, 0, 0),      # Truck - Red
    (255, 0, 0),      # Wheeled Slow - Red
    (255, 0, 0),      # Car Mount - Red
    (255, 0, 0)       # Ego Vehicle - Red
]
color_palette = np.array(color_palette)

def create_model(name):
    processor = Mask2FormerImageProcessor.from_pretrained(name)
    model = Mask2FormerForUniversalSegmentation.from_pretrained(name)

    processor.do_resize = False
    processor.do_rescale = True

    return processor, model

processor, model = create_model("facebook/mask2former-swin-large-mapillary-vistas-semantic")

def obtain_segmentation_map(model, img_path, sigma = 1.0):
    assert sigma <= 1 and sigma >= 0

    # read image and pre-process
    image = Image.open(img_path)
    inputs = processor(images=image, return_tensors="pt")

    # apply model
    with torch.no_grad():
      outputs = model(**inputs)

    # post-process
    seg = processor.post_process_semantic_segmentation(outputs, target_sizes=[image.size[::-1]])[0]

    # take the maximum logit for each pixel
    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    for label, color in enumerate(color_palette):
        color_seg[seg == label, :] = color
    color_seg = color_seg[..., ::1]

    # apply as a mask or as a whole image
    semantic_img = color_seg * sigma + np.array(image)*(1-sigma)
    semantic_img = semantic_img.astype(np.uint8)

    return semantic_img

# here we iterate the defined function over each image in the dataset and save them

target_dir_path = "../../Data/CVUSA_subset/streetview_segmentation_new"
src_dir_path = "../../Data/CVUSA_subset/streetview"

if not os.path.exists(target_dir_path):
    os.makedirs(target_dir_path)

image_names = os.listdir(src_dir_path)
image_paths = []
for image_name in image_names:
    image_path = os.path.join(src_dir_path, image_name)
    image_paths.append(image_path)
total_num = len(image_paths)


segmentation_paths = []
for image_name in image_names:
    image_path = os.path.join(target_dir_path, image_name)
    segmentation_paths.append(image_path)


for i in tqdm(range(0, total_num)):
    if not os.path.exists(segmentation_paths[i]):
        semantic_img = Image.fromarray(obtain_segmentation_map(model, image_paths[i], sigma = 1.0))
        semantic_img.save(segmentation_paths[i])