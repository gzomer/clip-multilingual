import requests
import os
import json

import numpy as np
import tqdm
from PIL import Image

from .utils import cache

API_KEY = os.getenv('UNSPLASH_API_KEY')
IMAGE_SIZE_DOWNLOAD = 'small'
IMAGE_SIZE_OUTPUT = 'small'

# Get queries
queries = []
base_folder = './datasets/unsplash'
with open(f'{base_folder}/queries.txt') as f:
  queries = [item.strip() for item in f.readlines()]


# Search photos
@cache(f'{base_folder}/search_results')
def search_photos(query):
    url = f'https://api.unsplash.com/search/photos?query={query}&per_page=30'
    payload = {'client_id': API_KEY}
    r = requests.get(url, params=payload)
    data = r.json()
    return data


def download_images_by_queries(queries, folder):
    for query in tqdm.tqdm(queries):
        data = search_photos(query)
        download_images(data, folder)


def download_images(data, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    for photo in data['results']:
        url = photo['urls'][IMAGE_SIZE_DOWNLOAD]

        if os.path.exists(f'{folder}/{photo["id"]}.jpg'):
            continue

        r = requests.get(url)
        with open(f'{folder}/{photo["id"]}.jpg', 'wb') as f:
            f.write(r.content)
        with open(f'{folder}/{photo["id"]}.json', 'w') as f:
            f.write(json.dumps(photo))


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


if __name__ == '__main__':
  # Download images
  download_images_by_queries(queries, f'{base_folder}/images')