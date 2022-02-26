import hashlib
import os
import warnings
import json
import random

import pandas as pd
import torch
import torchvision.datasets as dset
from torch.utils.data import Dataset
from PIL import Image


class PairedImageTextDataset(Dataset):
    def __init__(self, segments, image_transform=None, text_transform=None):
        self.segments = segments
        self.image_transform = image_transform
        self.text_transform = text_transform

    def __len__(self):
        return len(self.segments)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load image
        image_path = os.path.join(self.segments[idx]['image_path'])
        image = Image.open(image_path)

        if self.image_transform:
            image_item = self.image_transform(image)
        else:
            image_item = image

        # Load text
        text = self.segments[idx]['text']

        if self.text_transform:
            text_item = self.text_transform(text)
        else:
            text_item = text
        return image_item, text_item


def get_image_path(base_folder, url):
    filename = hashlib.sha1(url.encode('utf-8')).hexdigest()+ '.jpg'
    return f'{base_folder}{filename}'


def load_googlecc(dataset_base_dir, image_transform, tokenizer):

    all_items = pd.read_csv(f'{dataset_base_dir}/googlecc.tsv', delimiter='\t')
    all_items = list(all_items.T.to_dict().values())
    with open(f'{dataset_base_dir}/processed_map.json', 'r') as f:
        processed_map = json.load(f)

    valid_items = [item for item in all_items if processed_map.get(item['url'], None) == 'success']

    dataset_files = [
        {
            'image_path': get_image_path(f'{dataset_base_dir}/images/', item['url']),
            'text': item['caption']
        }
        for item in valid_items
    ]

    dataset_files = [
        item
        for item in dataset_files
        if os.path.exists(item['image_path']) and os.path.getsize(item['image_path']) > 0
    ]

    def check_image(path):
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                image = Image.open(path)
                if not image:
                    return False
                else:
                    return image.format
        except Exception as e:
            return False

    dataset_valid_images = [
        item
        for item in dataset_files
        if os.path.exists(item['image_path']) and os.path.getsize(item['image_path']) > 0
        and check_image(item['image_path'])
    ]

    return PairedImageTextDataset(dataset_valid_images, image_transform, tokenizer)


def load_coco(dataset_base_dir, image_transform, tokenizer):
    target_transform = lambda x: tokenizer(random.choice(x))
    return dset.CocoCaptions(
        root = f'{dataset_base_dir}/train2014/',
        annFile = f'{dataset_base_dir}/annotations/captions_train2014.json',
        transform=image_transform,
        target_transform=target_transform,
    )