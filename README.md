# MultiLingual CLIP

# Training

## Download dataset

## Training params


## Habana Gaudi - 8 accelerators

## Habana Gaudi - 16 accelerators (multi-server training)

## Other devices
If you don't have access to a Habana Gaudi accelerator yet, you can also train on CPU/GPU, although it will be way slower.

To train on CPU, just pass `--train-device=cpu` and on GPU `--train-device=cuda` to the `train.py` script.

# Inference

## Loading pre-trained model from Hugging Face HUB
```
from models import MultiLingualCLIP

model = MultiLingualCLIP(num_layers=3)
load_from_hub(model)
```


## Loading model from local checkpoint
```
from models import MultiLingualCLIP

text_checkpoint_path = '/path/to/text model checkpoint'
vision_checkpoint_path = '/path/to/vision model checkpoint'

model = MultiLingualCLIP(num_layers=3)
load_model(model, vision_checkpoint_path, text_checkpoint_path)
```

## Generate embeddings for your dataset


## Instantiate search

```
import numpy as np
from search import MultiLingualSearch


images_embeddings = np.load('/path/to/images_embeddings')
images_data = [...] # List of image info for each row of the embeddings. For instance, it could be a list of urls, filepaths, ids. They will be returned when calling the search function
semantic_search = MultiLingualSearch(model, images_embeddings, images_data)

results = semantic_search.search('people smiling')

results = [
    {'image': 'path1,}
]
