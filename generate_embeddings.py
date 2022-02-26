import os
import json

import numpy as np
import torch
import tqdm
from PIL import Image

from .models import create_and_load_from_hub

IMAGE_SIZE_OUTPUT = 'small'
base_folder = './datasets/unsplash'

def list_files(dir, image_type='jpg'):
  files = []
  for parent_path, _, filenames in os.walk(dir):
    for f in filenames:
      if f'.{image_type}' not in f:
        continue
      files.append(os.path.join(parent_path, f))
  return files


def load_images(files):
  images = {file: Image.open(file) for file in files}
  return images


def calculate_image_embeddings(model, images, device='cpu'):
  images_input = torch.stack([model.image_transform(img).to(device) for img in images])

  with torch.cuda.amp.autocast():
    with torch.no_grad():
      images_embeddings = model.vision_encoder(images_input).float().to(device)

  return images_embeddings


def generate_unsplash_embeddings(input_folder, output_folder, model=None, device='cpu'):
  if not model:
    model = create_and_load_from_hub()

  all_embeddings = []
  urls = []
  # Input files
  files = list_files(input_folder)
  for file in tqdm.tqdm(files):
    image = Image.open(file)
    with open(file.replace('.jpg', '.json'), 'r') as f:
      image_data = json.load(f)
      urls.append(image_data['urls'][IMAGE_SIZE_OUTPUT])

    embeddings = calculate_image_embeddings(model, [image], device=device)
    all_embeddings.append(embeddings[0].cpu().numpy())

  # Save embeddings
  if not os.path.exists(output_folder):
    os.makedirs(output_folder)
  np.save(f'{output_folder}/embeddings.npy', all_embeddings)
  with open(f'{output_folder}/urls.json', 'w') as f:
    json.dump(urls, f, indent=4)


if __name__ == '__main__':
  device = 'cpu' # hpu, cuda, or cpu

  # Create model
  model = create_and_load_from_hub()
  model.to(device)

  # Generate embeddings
  generate_unsplash_embeddings(
    input_folder=f'{base_folder}/images',
    output_folder=f'{base_folder}/embeddings',
    model = model,
    device = device,
  )
