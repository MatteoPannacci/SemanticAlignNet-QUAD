import torch
from depth_anything.dpt import DepthAnything
from torchvision.transforms import Compose
from depth_anything.util.transform import Resize, NormalizeImage, PrepareForNet
import cv2
import spaces
from PIL import Image
import numpy as np
import torch.nn.functional as F
import os
from tqdm import tqdm
import gc

device = 'cuda' if torch.cuda.is_available() else 'cpu'
encoder = 'vitl'
model = DepthAnything.from_pretrained(f"LiheYoung/depth_anything_{encoder}14").to(device).eval()

transform = Compose([
        Resize(
            width=518,
            height=518,
            resize_target=False,
            keep_aspect_ratio=True,
            ensure_multiple_of=14,
            resize_method='lower_bound',
            image_interpolation_method=cv2.INTER_CUBIC,
        ),
        NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        PrepareForNet(),
])

@spaces.GPU
@torch.no_grad()
def predict_depth(model, image):
    return model(image)

def obtain_depth_map(model, transform, img_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # load the image
    image = Image.open(img_path)
    image = np.array(image)
    original_image = image.copy()
    h, w = image.shape[:2]

    # pre-process the image
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
    image = transform({'image': image})['image']
    image = torch.from_numpy(image).unsqueeze(0).to(device)

    # apply the model
    depth = predict_depth(model, image)
    depth = F.interpolate(depth[None], (h, w), mode='bilinear', align_corners=False)[0, 0]

    # post-process the image
    depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
    depth = depth.cpu().numpy().astype(np.uint8)
    colored_depth = cv2.applyColorMap(depth, cv2.COLORMAP_INFERNO)[:, :, ::-1]

    return colored_depth

# here we iterate the defined function over each image in the dataset and save them

src_dir_path = "../../../Data/CVUSA_subset/streetview"
target_dir_path = "../../../Data/CVUSA_subset/streetview_depth_new"

if not os.path.exists(target_dir_path):
    os.makedirs(target_dir_path)

image_names = os.listdir(src_dir_path)
image_paths = []
for image_name in image_names:
    image_path = os.path.join(src_dir_path, image_name)
    image_paths.append(image_path)

depth_paths = []
for image_name in image_names:
    image_path = os.path.join(target_dir_path, image_name)
    depth_paths.append(image_path)

for i in tqdm(range(len(image_paths))):
    if i % 200 == 0:
        gc.collect()
        torch.cuda.empty_cache()
    if not os.path.exists(depth_paths[i]):
        colored_depth = obtain_depth_map(model, transform, image_paths[i])
        depth_image = Image.fromarray(colored_depth)
        depth_image.save(depth_paths[i])