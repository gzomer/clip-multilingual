# MultiLingual CLIP

Multilingual CLIP is a pre-trained model which can be used for multilingual semantic search and zero-shot image classification in 100 languages.


# Model Architecture
Multilingual CLIP was built using [OpenAI CLIP](https://github.com/openai/CLIP) model. I have used the same Vision encoder (ResNet 50x4), but instead I replaced their text encoder (Transformer) with a Mulilingual Text Encoder ([XLM-Roberta](https://huggingface.co/xlm-roberta-large)) and a configurable number of projection heads, as seen below:

![Model Architecture](https://challengepost-s3-challengepost.netdna-ssl.com/photos/production/software_photos/001/858/046/datas/gallery.jpg)

The model was trained in a distributed fashion on 16 Habana Gaudi Accelerators and with mixed Precision in two phases (using COCO Dataset for phase 1 and Google Conceptual Captions for phase 2). The training pipeline was built using PyTorch, PyTorch Lightning, and Distributed Data Parallel.


# Datasets

Three datasets have been used for building the model. COCO captions was used for training phase 1 and Google Conceptual Captions was used for training phase 2. Unsplash dataset was used for testing and inference.

## COCO Captions

COCO (Common Objects in Context) is a large-scale object detection, segmentation, and captioning dataset. The COCO captions dataset has around ~85000 images and captions pairs.

Run the following to download the dataset:

```bash
./download_coco.sh
```

This dataset was used for the first pre-training phase.

## Google Conceptual Captions

Conceptual Captions is a dataset consisting of ~3.3 million images annotated with captions. In contrast with the curated style of other image caption annotations, Conceptual Caption images and their raw descriptions are harvested from the web, and therefore represent a wider variety of styles.

Download the datasets urls/captions from [here](https://storage.cloud.google.com/gcc-data/Train/GCC-training.tsv?_ga=2.191230122.-1896153081.1529438250) as save it to `datasets/googlecc/googlecc.tsv`. The full dataset has over 3 million images, but you can select a subset by loading the `googlecc.tsv` file and saving only the number of rows you want (I have used 1 million images for training).

Then run the following commands to download each image on the `googlecc.tsv` file:

```bash
npm install
node download_build_googlecc.js
```

This dataset was used for the second pre-training phase.

## Unplash

This dataset was used as the test set during inference.

Run `python3.8 download_unsplash.py` to download the dataset.

# Training

![Training phase 1](https://challengepost-s3-challengepost.netdna-ssl.com/photos/production/software_photos/001/858/047/datas/gallery.jpg)

![Training phase 2](https://challengepost-s3-challengepost.netdna-ssl.com/photos/production/software_photos/001/858/048/datas/gallery.jpg)

## Setup

Create two Habana instances ([AWS EC2 DL1](https://aws.amazon.com/ec2/instance-types/dl1/)) using [Habana® Deep Learning Base AMI (Ubuntu 20.04)](https://aws.amazon.com/marketplace/pp/prodview-fw46rwuxrtfse)


Create the PyTorch docker container running:

```bash
docker run --name pytorch -td --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --net=host --ipc=host vault.habana.ai/gaudi-docker/1.2.0/ubuntu20.04/habanalabs/pytorch-installer-1.10.0:1.2.0-585
```

Enter the docker image by running:

```
docker exec -it pytorch  /bin/bash
```

#### Setup password-less ssh between all connected servers

1. Configure password-less ssh between all nodes:

   Do the following in all the nodes' docker sessions:
   ```bash
   mkdir ~/.ssh
   cd ~/.ssh
   ssh-keygen -t rsa -b 4096
   ```
   Copy id_rsa.pub contents from every node's docker to every other node's docker's ~/.ssh/authorized_keys (all public keys need to be in all hosts' authorized_keys):
   ```bash
   cat id_rsa.pub > authorized_keys
   vi authorized_keys
   ```
   Copy the contents from inside to other systems.
   Paste all hosts' public keys in all hosts' “authorized_keys” file.

2. On each system:
   Add all hosts (including itself) to known_hosts. The IP addresses used below are just for illustration:
   ```bash
   ssh-keyscan -p 3022 -H $IP1 >> ~/.ssh/known_hosts
   ssh-keyscan -p 3022 -H $IP2 >> ~/.ssh/known_hosts
   ```

3. Change Docker SSH port to 3022
    ```bash
    sed -i 's/#Port 22/Port 3022/g' /etc/ssh/sshd_config
    sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config
    service ssh restart
    ```

[Allow all TCP](https://docs.aws.amazon.com/vpc/latest/userguide/VPC_SecurityGroups.html) traffic between the nodes on AWS

Clone the git repo:

```bash
git clone https://github.com/gzomer/clip-multilingual
```

Create environment:

```bash
python3.8 -m venv .env
```

Install requirements:

```bash
python3.8 -r requirements.txt
```

Activate environment

```bash
source .env/bin/activate
```

## Training params

Learning rate: 1e-3

Batch size: 64

Phase 1 - Epochs: 100

Phase 2 - Epochs: 15

## Train script arguments

```
--dataset-num-workers       Number of workers (default: 8)
--dataset-type      Dataset type (coco or googlecc) (default: coco)
--dataset-dir       Dataset dir (default: ./datasets/coco/)
--dataset-subset-size       Load only a subset of the dataset (useful for debugging)
--dataset-train-split       Dataset train split (default: 0.8)
--train-device      Type of device to use (default: hpu)
--distributed-num-nodes         Number of nodes (machines) (default: 2)
--distributed-parallel-devices      Number of parallel devices per node (default: 8)
--distributed-master-address        Master node IP address
--distributed-master-port       Master node port (default: 12345)
--distributed-bucket-cap-mb         DDP bucket cap MB (default: 200)
--checkpoint-dir        Model checkpoint dir (default: ./models)
--checkpoint-save-every-n       Save every n epochs (default: 1)
--checkpoint-load-vision-path       Load vision encoder checkpoint
--checkpoint-load-text-path         Load text encoder checkpoint
--model-visual-name         Which visual model to use (default: RN50x4)
--model-textual-name        Which textual model to use (default: xlm-roberta-base)
--hyperparam-num-layers         Number of layers (default: 3)
--hyperparam-lr         Model learning rate (default: 0.001)
--hyperparam-epochs         Max epochs (default: 100)
--hyperparam-precision      Precision (default: 16)
--hyperparam-batch-size         Batch size (default: 64)
--wandb-project         W&B project name (default: clip)
--wandb-enabled         W&B is enabled? (default: True)
```

## Habana Gaudi - 8 accelerators

### Phase 1 training

```bash
python3.8 train.py --train-device hpu --distributed-parallel-devices 8 --distributed-num-nodes 1
```

### Phase 2 training
```bash
python3.8 train.py --train-device hpu --distributed-parallel-devices 8 --distributed-num-nodes 1 --hyperparam-epochs 15 --checkpoint-load-text-path /home/models/text-last.ckpt --checkpoint-load-vision-path /home/models/vision-last.ckpt --checkpoint-dir ./models_phase2
```

## Habana Gaudi - 16 accelerators (multi-server training)

Change the master IP address based on your instances (use local IP, not public IP).

### Phase 1 training

```bash
NODE_RANK=0 python3.8 train.py --distributed-master-address 172.31.86.231 --train-device hpu --distributed-parallel-devices 8 --distributed-num-nodes 2
```

```bash
NODE_RANK=1 python3.8 train.py --distributed-master-address 172.31.86.231 --train-device hpu --distributed-parallel-devices 8 --distributed-num-nodes 2
```

### Phase 2 training

```bash
NODE_RANK=0 python3.8 train.py --distributed-master-address 172.31.86.231 --train-device hpu --distributed-parallel-devices 8 --distributed-num-nodes 2 --hyperparam-epochs 10 --checkpoint-load-text-path /home/models/text-last.ckpt --checkpoint-load-vision-path /home/models/vision-last.ckpt --checkpoint-dir ./models_phase2
```

```bash
NODE_RANK=1 python3.8 train.py --distributed-master-address 172.31.86.231 --train-device hpu --distributed-parallel-devices 8 --distributed-num-nodes 2 --hyperparam-epochs 15 --checkpoint-load-text-path /home/models/text-last.ckpt --checkpoint-load-vision-path /home/models/vision-last.ckpt --checkpoint-dir ./models_phase2
```

## Other devices
If you don't have access to a Habana Gaudi accelerator yet, you can also train on CPU/GPU, although it will be way slower.

To train on CPU, just pass `--train-device=cpu` and on GPU `--train-device=cuda` to the `train.py` script.

# Inference

## Loading pre-trained model from Hugging Face HUB
```python
from models import create_and_load_from_hub

model = create_and_load_from_hub()
```

## Loading model from local checkpoint
```python
from models import MultiLingualCLIP, load_model

text_checkpoint_path = '/path/to/text model checkpoint'
vision_checkpoint_path = '/path/to/vision model checkpoint'

model = MultiLingualCLIP(num_layers=3)
load_model(model, vision_checkpoint_path, text_checkpoint_path)
```

## Generate embeddings

Run the following (after downloading Unplash dataset):

`python3.8 ./generate_embeddings.py`

## Searching images

```python
import numpy as np
from search import MultiLingualSearch

images_embeddings = np.load('/path/to/images_embeddings')
images_data = [...] # List of image info for each row of the embeddings. For instance, it could be a list of urls, filepaths, ids. They will be returned when calling the search function
semantic_search = MultiLingualSearch(model, images_embeddings, images_data)

results = semantic_search.search('विद्यालय में') # Means at school
print(results)
```
```json
[{"image": "https://images.unsplash.com/photo-1557804506-669a67965ba0?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=MnwyNDg3OTV8MHwxfHNlYXJjaHwxM3x8bWVldGluZ3N8ZW58MHx8fHwxNjQ1NjA2MjQz&ixlib=rb-1.2.1&q=80&w=400",
  "prob": 0.2461608648300171},
 {"image": "https://images.unsplash.com/photo-1558403194-611308249627?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=MnwyNDg3OTV8MHwxfHNlYXJjaHwyMXx8cGVvcGxlJTIwd29ya2luZ3xlbnwwfHx8fDE2NDU2MDMyMjE&ixlib=rb-1.2.1&q=80&w=400",
  "prob": 0.16881239414215088},
 {"image": "https://images.unsplash.com/photo-1531497865144-0464ef8fb9a9?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=MnwyNDg3OTV8MHwxfHNlYXJjaHw4Nnx8cGVvcGxlJTIwd29ya2luZ3xlbnwwfHx8fDE2NDU2MDY5ODc&ixlib=rb-1.2.1&q=80&w=400",
  "prob": 0.14744874835014343},
 {"image": "https://images.unsplash.com/photo-1561089489-f13d5e730d72?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=MnwyNDg3OTV8MHwxfHNlYXJjaHw5MHx8ZWR1Y2F0aW9ufGVufDB8fHx8MTY0NTYwNjk1Nw&ixlib=rb-1.2.1&q=80&w=400",
  "prob": 0.095176100730896},
 {"image": "https://images.unsplash.com/photo-1580582932707-520aed937b7b?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=MnwyNDg3OTV8MHwxfHNlYXJjaHwxMnx8ZWR1Y2F0aW9ufGVufDB8fHx8MTY0NTYwMzIwMA&ixlib=rb-1.2.1&q=80&w=400",
  "prob": 0.05218643322587013}]
```
