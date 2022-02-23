import os
from copy import deepcopy
from collections import namedtuple

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from .habana_utils import adjust_tensors_for_save, change_state_dict_device


class PeriodicCheckpoint(ModelCheckpoint):
    def __init__(self,
                filepath  = None,
                monitor  = None,
                verbose=  False,
                save_last = None,
                save_top_k = 1,
                save_weights_only= False,
                mode = 'auto',
                period = 1,
                prefix = '',
                dirpath = None,
                filename = None,
                every_n = 10,
                first_n = 10,
                pl_module = None):
        super().__init__(dirpath=dirpath,
                            filename=filename,
                            monitor=monitor,
                            verbose=verbose,
                            save_last=save_last,
                            save_top_k=save_top_k,
                            save_weights_only=save_weights_only,
                            mode=mode,
                            every_n_epochs=period)
        self.every_n = every_n
        self.first_n = first_n
        self.pl_module = pl_module

    def restore_tensors_for_ckpt(self, pl_module, model, state_dict):
        assert (pl_module.config.train_device == 'hpu')

        model.load_state_dict(state_dict)
        adjust_tensors_for_save(
            model.state_dict(),
            pl_module.optimizers().state,
            to_device='hpu',
            to_filters_last=True,
            permute=False
        )

    def _save_model(self, trainer: 'pl.Trainer', filepath: str) -> None:
        # make paths
        if trainer.should_rank_save_checkpoint:
            self._fs.makedirs(os.path.dirname(filepath), exist_ok=True)

        # delegate the saving to the trainer
        trainer.save_checkpoint(filepath, self.save_weights_only)

    def on_validation_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule'):
        # Save a copy of state_dict and restore after save is finished
        pl_module = self.pl_module
        if pl_module.config.train_device == 'hpu':
            state_dict_vision_encoder = deepcopy(change_state_dict_device(pl_module.vision_encoder.state_dict(), 'cpu'))
            state_dict_text_encoder = deepcopy(change_state_dict_device(pl_module.text_encoder.state_dict(), 'cpu'))

        super(PeriodicCheckpoint, self).on_validation_end(trainer, pl_module)
        if pl_module.config.train_device == 'hpu':
            self.restore_tensors_for_ckpt(pl_module, pl_module.vision_encoder, state_dict_vision_encoder)
            self.restore_tensors_for_ckpt(pl_module, pl_module.text_encoder, state_dict_text_encoder)

    def save_checkpoint(self, trainer: 'pl.Trainer'):
        pl_module = self.pl_module
        if pl_module.config.train_device == 'hpu':
            state_dict_vision_encoder = deepcopy(change_state_dict_device(pl_module.vision_encoder.state_dict(), 'cpu'))
            state_dict_text_encoder = deepcopy(change_state_dict_device(pl_module.text_encoder.state_dict(), 'cpu'))

        super(PeriodicCheckpoint, self).save_checkpoint(trainer)
        if pl_module.config.train_device == 'hpu':
            self.restore_tensors_for_ckpt(pl_module, pl_module.vision_encoder, state_dict_vision_encoder)
            self.restore_tensors_for_ckpt(pl_module, pl_module.text_encoder, state_dict_text_encoder)