from typing import Dict

import clip
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from transformers import AutoModel, AutoTokenizer, BertTokenizer
from huggingface_hub import cached_download, hf_hub_url

EMBED_DIM = 512
CLIP_EMBED_DIM = 640
TRANSFORMER_EMBED_DIM = 768
MAX_LEN = 64
FREEZE_ENCODERS = True
MODEL_REPO = 'gzomer/clip-multilingual'
DEFAULT_TEXT_MODEL_NAME = 'xlm-roberta-base'
DEFAULT_VISUAL_MODEL_NAME = 'RN50x4'
DEFAULT_NUM_LAYERS = 3

class Projection(nn.Module):
    def __init__(self, d_in: int, d_out: int, p: float=0.5, num_layers=3) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_in, d_out, bias=False)
        self.linear_layers = []
        self.dropouts = []
        for _ in range(num_layers):
            self.linear_layers.append(nn.Linear(d_out, d_out, bias=False))
            self.dropouts.append(nn.Dropout(p))
        self.linear_layers = nn.ModuleList(self.linear_layers)
        self.dropouts = nn.ModuleList(self.dropouts)
        self.layer_norm = nn.LayerNorm(d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        projected_embeddings = self.linear1(x)

        for index in range(len(self.linear_layers)):
          x = F.gelu(projected_embeddings)
          x = self.linear_layers[index](x)
          x = self.dropouts[index](x)
          projected_embeddings = self.layer_norm(projected_embeddings + x)

        return projected_embeddings


class VisionEncoder(nn.Module):
    def __init__(self, vision_model, d_in: int, d_out: int, num_layers=3) -> None:
        super().__init__()
        base = vision_model
        self.base = base
        self.projection = Projection(d_in, d_out, num_layers=num_layers)
        if FREEZE_ENCODERS:
          for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x):
        projected_vec = self.projection(self.base(x))
        projection_len = torch.norm(projected_vec, p=2, dim=-1, keepdim=True)
        return projected_vec / projection_len


class Tokenizer:
    def __init__(self, tokenizer: BertTokenizer) -> None:
        self.tokenizer = tokenizer

    def __call__(self, x: str) -> AutoTokenizer:
        return self.tokenizer(
            x, max_length=MAX_LEN, truncation=True, padding='max_length', return_tensors='pt'
        )

    def decode(self, x: Dict[str, torch.LongTensor]):
        return [self.tokenizer.decode(sentence[:sentence_len]) for sentence, sentence_len in
                zip(x['input_ids'], x['attention_mask'].sum(axis=-1))]


class TextEncoder(nn.Module):
    def __init__(self, encoder_model_name, d_out: int, num_layers=3) -> None:
        super().__init__()
        self.base = AutoModel.from_pretrained(encoder_model_name)
        self.projection = Projection(TRANSFORMER_EMBED_DIM, d_out, num_layers=num_layers)
        if FREEZE_ENCODERS:
          for p in self.base.parameters():
              p.requires_grad = False

    def forward(self, x):
        out = self.base(**x).last_hidden_state
        out = out[:, 0, :]  # get CLS token vector
        projected_vec = self.projection(out)

        projection_len = torch.norm(projected_vec, p=2, dim=-1, keepdim=True)
        return projected_vec / projection_len


class MultiLingualCLIP(pl.LightningModule):
    def __init__(self,
                 clip_model,
                 text_model_name,
                 num_layers = 3,
                 clip_embed_dim = CLIP_EMBED_DIM,
                 embed_dim = EMBED_DIM,
        ) -> None:
        super().__init__()
        self.vision_encoder = VisionEncoder(
            clip_model.visual,
            clip_embed_dim,
            embed_dim,
            num_layers=num_layers,
        )
        self.text_encoder = TextEncoder(text_model_name, embed_dim, num_layers=num_layers)


def create_tokenizer(model_name=DEFAULT_TEXT_MODEL_NAME):
    return Tokenizer(AutoTokenizer.from_pretrained(model_name))


def load_model(model, vision_checkpoint_path, text_checkpoint_path, device='cpu'):
    model.vision_encoder.projection.load_state_dict(torch.load(vision_checkpoint_path, map_location=device))
    model.text_encoder.projection.load_state_dict(torch.load(text_checkpoint_path, map_location=device))


def load_from_hub(model, device='cpu'):
    clip_text_projection_url = hf_hub_url(repo_id=MODEL_REPO, filename='clip-text-projection.ckpt')
    clip_visual_projection_url = hf_hub_url(repo_id=MODEL_REPO, filename='clip-visual-projection.ckpt')

    text_model_checkpoint = cached_download(url=clip_text_projection_url)
    visual_model_checkpoint = cached_download(url=clip_visual_projection_url)

    load_model(model, visual_model_checkpoint, text_model_checkpoint, device=device)


def create_default_model():
    clip_model, compose = clip.load(DEFAULT_VISUAL_MODEL_NAME)
    return MultiLingualCLIP(
        clip_model=clip_model,
        text_model_name=DEFAULT_TEXT_MODEL_NAME,
        num_layers=DEFAULT_NUM_LAYERS,
        clip_embed_dim=CLIP_EMBED_DIM,
        embed_dim=EMBED_DIM,
    )

def create_and_load_from_hub():
    model = create_default_model()
    load_from_hub(model)
    return model