import os
import random
import argparse
from typing import Tuple, List
from copy import deepcopy

import clip
import torch
import torchvision.datasets as dset
import torch.nn.functional as F
import pytorch_lightning as pl
import wandb
from torch.utils.data import random_split, DataLoader
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers import WandbLogger
from transformers import AutoTokenizer

from .datasets import load_googlecc, load_coco
from .models import Tokenizer, MultiLingualCLIP, AutoTokenizer, load_model
from .checkpoint import PeriodicCheckpoint
from .habana_utils import permute_params, adjust_tensors_for_save, change_state_dict_device


# Default params
DEFAULT_TRAIN_DEVICE = 'hpu'
DEFAULT_DATASET_SPLIT = 0.8
DEFAULT_DATASET_BASE_DIR = './datasets/coco/'
DEFAULT_DATASET_TYPE = 'coco'
DEFAULT_DATASET_NUM_WORKERS = 8
DEFAULT_WANDB_ENABLED = True
DEFAULT_DISTRIBUTED_DEVICES_PER_NODE = 8
DEFAULT_DISTRIBUTED_NUM_NODES = 2
DEFAULT_DISTRIBUTED_BUCKET_MB = 200
DEFAULT_HYPERPARAM_BATCH_SIZE = 64
DEFAULT_HYPERPARAM_EPOCHS = 100
DEFAULT_HYPERPARAM_NUM_LAYERS = 3
DEFAULT_HYPERPARAM_LR = 1e-3
DEFAULT_HYPERPARAM_PRECISION = 16
DEFAULT_CHECKPOINT_DIR='./models'
DEFAULT_CHECKPOINT_SAVE_EVERY = 1
DEFAULT_WANDB_PROJECT_NAME='clip-multilingual'
DEFAULT_VISUAL_MODEL_NAME = 'RN50x4'
DEFAULT_TEXTUAL_MODEL_NAME = 'xlm-roberta-base'

# Habana
## Setup multi server training params (change here or pass the params as command line args)
def setup_distributed_training(config):
    os.environ['PL_TORCH_DISTRIBUTED_BACKEND'] = 'hccl'
    if config.distributed_num_nodes > 1:
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        os.environ['HCCL_OVER_TCP'] = '1'
        os.environ['HCCL_SOCKET_IFNAME'] = 'ens32'
        os.environ['SOCKET_NTHREADS'] = '2'
        os.environ['NSOCK_PERTHREAD'] = '3'
        os.environ['HCCL_COMM_ID'] = f'{config.distributed_master_address}:9696'
        os.environ['MASTER_ADDR'] = config.distributed_master_address
        os.environ['MASTER_PORT'] = config.distributed_master_port

def is_habana(config):
    return config.train_device == 'hpu'

def init_habana(model):
    from habana_frameworks.torch.utils.library_loader import load_habana_module
    load_habana_module()

    import habana_frameworks.torch.core
    import habana_frameworks.torch.core.hccl


# Data preparation
def create_data_loaders(
    dataset_type,
    image_transform,
    tokenizer,
    dataset_base_dir: str,
    batch_size: int,
    num_workers: int,
    train_split: float,
    subset_size = None
    ):

    if dataset_type == 'googlecc':
        dataset = load_googlecc(dataset_base_dir, image_transform, tokenizer)
    else:
        dataset = load_coco(dataset_base_dir, image_transform, tokenizer)

    if subset_size:
        dataset = torch.utils.data.Subset(dataset, list(range(subset_size)))

    # Split dataset into train and validation
    train_len = int(train_split*len(dataset))
    train_data, valid_data = random_split(dataset, [train_len, len(dataset) - train_len])

    train_dataloader = DataLoader(
        train_data,
        batch_size,
        pin_memory=True,
        num_workers=num_workers,
        drop_last=True,
    )
    valid_dataloader = DataLoader(
        valid_data,
        batch_size,
        pin_memory=True,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    return train_dataloader, valid_dataloader

# Evaluation metrics
def contrastive_loss(logits, dim):
    neg_ce = torch.diag(F.log_softmax(logits, dim=dim))
    return -neg_ce.mean()

def clip_loss(similarity: torch.Tensor) -> torch.Tensor:
    text_loss = contrastive_loss(similarity, dim=0)
    image_loss = contrastive_loss(similarity, dim=1)
    return (text_loss + image_loss) / 2.0

def metrics(similarity: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    y = torch.arange(len(similarity)).to(similarity.device)
    img2cap_match_idx = similarity.argmax(dim=1)
    cap2img_match_idx = similarity.argmax(dim=0)

    img_acc = (img2cap_match_idx == y).float().mean()
    text_acc = (cap2img_match_idx == y).float().mean()

    return img_acc, text_acc

# Define PyTorch Lighting train pipeline

class Model(MultiLingualCLIP):
    def __init__(self, config, *args, **kwargs):
      self.config = config
      super().__init__(*args, **kwargs)

    def common_step(self, batch: Tuple[torch.Tensor, List[str]]) -> torch.Tensor:
        images, text = batch
        text = {k: torch.squeeze(v, 1).to(self.device) for k, v in text.items()}

        image_embed = self.vision_encoder(images)
        caption_embed = self.text_encoder(text)
        similarity = caption_embed @ image_embed.T

        loss = clip_loss(similarity)
        img_acc, text_acc = metrics(similarity)
        return loss, img_acc, text_acc

    def training_step(
        self, batch: Tuple[torch.Tensor, List[str]], *args: list
    ) -> torch.Tensor:
        loss, img_acc, text_acc = self.common_step(batch)
        self.log('training_loss', loss, on_step=True)
        self.log('training_img_acc', img_acc, on_step=True, prog_bar=True)
        self.log('training_text_acc', text_acc, on_step=True, prog_bar=True)
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, List[str]], *args: list
    ) -> torch.Tensor:
        loss, img_acc, text_acc = self.common_step(batch)
        self.log('validation_loss', loss, on_step=True)
        self.log('validation_img_acc', img_acc, on_step=True, prog_bar=True)
        self.log('validation_text_acc', text_acc, on_step=True, prog_bar=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        vision_params = {'params': self.vision_encoder.projection.parameters(), 'lr': self.config.hyperparam_lr}
        text_params = {'params': self.text_encoder.projection.parameters() , 'lr': self.config.hyperparam_lr}

        if self.device == 'hpu':
            from habana_frameworks.torch.hpex.optimizers import FusedAdamW
            print('[Train] Using FusedAdamW optimizer')
            optimizer = FusedAdamW([vision_params, text_params])
        else:
            optimizer = torch.optim.Adam([vision_params, text_params])
        return optimizer

    def on_save_checkpoint(self, checkpoint):
        if self.device != 'hpu':
            return

        state_dict = checkpoint['state_dict']
        optimizer_states = checkpoint['optimizer_states']
        optimizer_state_dict = optimizer_states[0]['state']

        for  k, v in checkpoint['callbacks'].items():
            if isinstance(v, dict):
                for k1, v1 in v.items():
                    if isinstance(v1, torch.Tensor):
                        v[k1] = v1.to('cpu')

        adjust_tensors_for_save(
            state_dict,
            optimizer_state_dict,
            to_device='cpu',
            to_filters_last=False,
            permute=False
        )


class CustomTrainer(pl.Trainer):
    def save_checkpoint(self, filepath, weights_only=False):
        if self.is_global_zero:
            dirpath, filename = os.path.split(filepath)
            os.makedirs(dirpath, exist_ok=True)

            # Check if model has been wrapped by DistributedDataParallel
            model = self.model
            if hasattr(self.model, 'module'):
                model = self.model.module.module

            torch.save(
                deepcopy(change_state_dict_device(model.vision_encoder.projection.state_dict(), 'cpu')),
                os.path.join(dirpath, 'vision-' + filename),
            )
            torch.save(
                deepcopy(change_state_dict_device(model.text_encoder.projection.state_dict(), 'cpu')),
                os.path.join(dirpath, 'text-' + filename),
            )


def main(config):
    if config.wandb_enabled:
        wandb.login()
        logger = WandbLogger(project=config.wandb_project)

    # Instantiate base CLIP and tokenizer
    clip_model, image_transform = clip.load(config.model_visual_name)
    tokenizer = Tokenizer(AutoTokenizer.from_pretrained(config.model_textual_name))

    # Instantiate model
    model = Model(
        config,
        clip_model,
        image_transform,
        tokenizer,
        config.model_textual_name,
        config.hyperparam_num_layers,
    )

    # Resume from checkpoint (either when resuming training or for training with different dataset)
    if config.checkpoint_load_vision_path and config.checkpoint_load_text_path:
        load_model(model, config.checkpoint_load_vision_path, config.checkpoint_load_text_path)

    # Create data loaders
    train_dataloader, valid_dataloader = create_data_loaders(
        config.dataset_type,
        image_transform,
        tokenizer,
        dataset_base_dir = config.dataset_dir,
        batch_size = config.hyperparam_batch_size,
        num_workers = config.dataset_num_workers,
        train_split = config.dataset_train_split,
        subset_size = config.dataset_subset_size,
    )

    # Setup Habana (move to device and permute params)
    if is_habana(config):
        init_habana(model)
        # Setup multi server training params if needed
        setup_distributed_training(config)
        # Move to HPU
        model.to(config.train_device)
        # Permute weights of vision model
        permute_params(model.vision_encoder, to_filters_last=True)
    else:
        model.to(config.train_device)

    # Setup number of parallel devices
    parallel_devices = []
    if is_habana(config):
        parallel_devices = [torch.device(config.train_device)] * config.distributed_parallel_devices
    else:
        parallel_devices = config.distributed_parallel_devices

    # Setup distributed strategy if configured
    strategy = DDPPlugin(
        parallel_devices=parallel_devices,
        bucket_cap_mb=config.distributed_bucket_cap_mb,
        gradient_as_bucket_view=True,
        static_graph=True,
        broadcast_buffers = False,
        find_unused_parameters=False
    ) if config.distributed_parallel_devices > 1 else None

    callbacks = []
    # Create checkpoints
    if config.checkpoint_save_every_n > 0:
        model_ckpt = PeriodicCheckpoint(
            monitor='validation_loss',
            dirpath=config.checkpoint_dir,
            mode='min',
            save_last=True,
            filename='clip-{epoch}',
            every_n=config.checkpoint_save_every_n,
            pl_module=model)

        callbacks.append(model_ckpt)

    # Create trainer
    if is_habana(config):
      trainer = CustomTrainer(
          max_epochs=config.hyperparam_epochs,
          precision=config.hyperparam_precision,
          hpus=parallel_devices,
          num_nodes=config.distributed_num_nodes,
          strategy=strategy,
          callbacks=callbacks,
      )
    else:
      trainer = CustomTrainer(
          max_epochs=config.hyperparam_epochs,
          precision=config.hyperparam_precision,
          gpus=parallel_devices,
          num_nodes=config.distributed_num_nodes,
          strategy=strategy,
          callbacks=callbacks,
      )

    # Fit model
    trainer.fit(
        model,
        train_dataloader,
        valid_dataloader,
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-num-workers', type=int, default=DEFAULT_DATASET_NUM_WORKERS, help='Number of workers')
    parser.add_argument('--dataset-type', type=str, default=DEFAULT_DATASET_TYPE, help='Dataset type (coco or googlecc)')
    parser.add_argument('--dataset-dir', type=str, default=DEFAULT_DATASET_BASE_DIR, help='Dataset dir')
    parser.add_argument('--dataset-subset-size', type=int, help='Load only a subset of the dataset (useful for debugging)')
    parser.add_argument('--dataset-train-split', type=str, default=DEFAULT_DATASET_SPLIT, help='Dataset train split')
    parser.add_argument('--train-device', default='hpu', help='Type of device to use')
    parser.add_argument('--distributed-num-nodes', type=int, default=DEFAULT_DISTRIBUTED_NUM_NODES, help='Number of nodes (machines)')
    parser.add_argument('--distributed-parallel-devices', type=int, default=DEFAULT_DISTRIBUTED_DEVICES_PER_NODE, help='Number of parallel devices per node')
    parser.add_argument('--distributed-master-address', type=str, help='Master node IP address')
    parser.add_argument('--distributed-master-port', type=str, default='12345', help='Master node port')
    parser.add_argument('--distributed-bucket-cap-mb', type=int, default=DEFAULT_DISTRIBUTED_BUCKET_MB, help='DDP bucket cap MB')
    parser.add_argument('--checkpoint-dir', type=str, default=DEFAULT_CHECKPOINT_DIR, help='Model checkpoint dir')
    parser.add_argument('--checkpoint-save-every-n', type=int, default=DEFAULT_CHECKPOINT_SAVE_EVERY, help='Save every n epochs')
    parser.add_argument('--checkpoint-load-vision-path', type=str, help='Load vision encoder checkpoint')
    parser.add_argument('--checkpoint-load-text-path', type=str, help='Load text encoder checkpoint')
    parser.add_argument('--model-visual-name', type=str, default=DEFAULT_VISUAL_MODEL_NAME, help='Which visual model to use')
    parser.add_argument('--model-textual-name', type=str, default=DEFAULT_TEXTUAL_MODEL_NAME, help='Which textual model to use')
    parser.add_argument('--hyperparam-num-layers', type=int, default=DEFAULT_HYPERPARAM_NUM_LAYERS, help='Number of layers')
    parser.add_argument('--hyperparam-lr', type=float, default=DEFAULT_HYPERPARAM_LR, help='Model learning rate')
    parser.add_argument('--hyperparam-epochs', type=int, default=DEFAULT_HYPERPARAM_EPOCHS, help='Max epochs')
    parser.add_argument('--hyperparam-precision', type=int, default=DEFAULT_HYPERPARAM_PRECISION, help='Precision')
    parser.add_argument('--hyperparam-batch-size', type=int, default=DEFAULT_HYPERPARAM_BATCH_SIZE, help='Batch size')
    parser.add_argument('--wandb-project', type=str, default='clip', help='W&B project name')
    parser.add_argument('--wandb-enabled', type=bool, default=True, help='W&B is enabled?')

    args = parser.parse_args()
    main(args)