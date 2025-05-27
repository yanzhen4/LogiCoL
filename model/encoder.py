from functools import partial

import numpy as np
import torch
import os
import json

from transformers import AutoModel, T5EncoderModel, AutoConfig, get_linear_schedule_with_warmup, get_constant_schedule_with_warmup
import pytorch_lightning as pl

from model.layers import Pooling
from model.losses import LOSS_CLASSES

from utils.tensor_utils import gather_with_var_len, reduce_embeddings, reduce_pos_neg_masks, make_neg_mask, make_strict_neg_mask, apply_label_offset

class LitEncoder(pl.LightningModule):
    def __init__(self, params):
        super().__init__()
        self.save_hyperparameters()  # This will save all arguments / params passed to __init__()

        self.params = params

        # Backbone encoder (from huggingface)
        if "t5" in params.model_name:
            self.encoder = T5EncoderModel.from_pretrained(params.model_name)
        else:
            self.encoder = AutoModel.from_pretrained(params.model_name, trust_remote_code=True)

        self.encoder_config = AutoConfig.from_pretrained(params.model_name, trust_remote_code=True)
        self.encoder_hidden_dim = self.encoder_config.hidden_size
        
        # Pooling
        self.pooler = Pooling()

        # Keep track of validation examples
        self.validation_step_contrastive_loss_outputs = []
        self.validation_step_subset_loss_outputs = []
        self.validation_step_exclusion_loss_outputs = []

        if hasattr(params, "train") and params.train:
            if params.weights_name:
                self.load_checkpoint(params.weights_name)

            self.batch_strategy = params.batch_strategy
            self.loss_type = params.loss_type
            self.subset_loss_weight = params.subset_loss_weight
            self.subset_loss_scale = params.subset_loss_scale
            self.subset_loss_margin = params.subset_loss_margin
            self.exclusion_loss_weight = params.exclusion_loss_weight
            self.exclusion_loss_scale = params.exclusion_loss_scale
            self.exclusion_loss_margin = params.exclusion_loss_margin
            self.normalization = params.normalization
            self.contrastive_loss_weight = params.contrastive_loss_weight #Defualt is 1.0
            self.loss_fn = LOSS_CLASSES[self.loss_type]
            self.subset_loss_fn = LOSS_CLASSES["subset_asymmetric"]
            self.exclusion_loss_fn = LOSS_CLASSES["exclusion"]

        # Number of GPUs
        self.num_gpus = params.gpus

        # self.make_strict_neg_mask = params.make_strict_neg_mask
        self.make_strict_neg_mask = 1

        # self.in_group_negative = params.in_group_negative
        self.in_group_negative = 1

    def forward(self,
                input_ids,
                attention_mask):
        """

        :param encoder_inputs: inputs for encoder
        :param num_prop_per_sentence:
        :param all_prop_mask:
        :param prop_labels:
        :return:
        """

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        output_pooled = self.pooler(outputs.last_hidden_state, attention_mask)

        return output_pooled

    def configure_optimizers(self):

        # TODO(sihaoc): maybe add learning rate scaling
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.params.learning_rate)

        if self.params.lr_scheduler == "constant":
            scheduler = get_constant_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.params.warmup_steps,
            )
        elif self.params.lr_scheduler == "linear":
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.params.warmup_steps,
                num_training_steps=self.trainer.estimated_stepping_batches,
            )
        else:
            raise ValueError(f"Invalid scheduler type: {self.params.scheduler_type}")

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }

    #####
    # Defines how each forward step
    #####
    def _step(self, batch):

        encoded = self.forward(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask
        )

        return {
            "encoded": encoded,
            "pos_mask": batch.pos_mask,
        }

        #Need to record 
    
    def _step_group(self, batch):
        
        encoded = self.forward(
            input_ids=batch.input_ids,
            attention_mask=batch.attention_mask
        )

        return {
            "encoded": encoded,
            "pos_mask": batch.pos_mask,
            'unallowed_neg_mask': batch.unallowed_neg_mask,
            "query_indices": batch.query_indices,
            "document_indices": batch.document_indices,
            "subset_relation": batch.subset_relation,
            "exclusion_relation": batch.exclusion_relation
        }

    def training_step(self, train_batch, batch_idx):
        
        if self.batch_strategy == 'group':
            shard_output = self._step_group(train_batch)
        else:
            shard_output = self._step(train_batch)

        # With DDP, each GPU gets its own mini-batch.
        # Since the labels are created by enumerating instances within batch, i.e. labels ranges from 0 to batch_size
        # within each batch, when merging the batches, we need to differentiate between the labels of each batch
        # This is handled by gather_with_var_len

        if self.num_gpus > 1:
            gather_fn = partial(self.all_gather, sync_grads=True)
            all_output = {
                key: gather_with_var_len(
                    val,
                    base_gather_fn=gather_fn,
                    reduce_fn=(
                    reduce_pos_neg_masks if key in ['pos_mask', 'unallowed_neg_mask', 'subset_relation', 'exclusion_relation']
                    else apply_label_offset if key in ['query_indices', 'document_indices']
                    else reduce_embeddings))
                for key, val in shard_output.items()
            }
        else:
            all_output = shard_output

        if self.make_strict_neg_mask == 1 and self.batch_strategy == 'group':
            neg_mask = make_strict_neg_mask(all_output["pos_mask"], all_output["query_indices"])
        else:
            neg_mask = make_neg_mask(all_output["pos_mask"])

        # For every indices where allowed_neg_mask is 0, turn the corresponding indices in neg_mask to be 0
        if self.in_group_negative == 0 and self.batch_strategy == 'group':
            allowed_neg_mask = 1 - all_output['unallowed_neg_mask']
            assert neg_mask.shape == all_output['unallowed_neg_mask'].shape

            prev_neg_mask = neg_mask

            neg_mask = neg_mask * allowed_neg_mask

        contrastive_loss = self.loss_fn(
            all_output['encoded'], 
            all_output["pos_mask"], 
            neg_mask
        )   

        subset_loss = torch.tensor(0.0, device=self.device)
        exclusion_loss = torch.tensor(0.0, device=self.device)

        if self.batch_strategy == 'group':

            assert len(all_output['encoded']) == len(all_output['query_indices']) + len(all_output['document_indices'])

            assert len(all_output['query_indices']) == len(all_output['subset_relation'])
            assert len(all_output['query_indices']) == len(all_output['exclusion_relation'])

            subset_loss = self.subset_loss_fn(
                all_output['encoded'],
                all_output['query_indices'],
                all_output['document_indices'],
                all_output['subset_relation'],
                scale = self.subset_loss_scale,
                normalization = self.normalization,
                margin = self.subset_loss_margin
            )

            exclusion_loss = self.exclusion_loss_fn(
                all_output['encoded'],
                all_output['query_indices'],
                all_output['document_indices'],
                all_output['exclusion_relation'],
                scale = self.exclusion_loss_scale,
                normalization = self.normalization,
                margin = self.exclusion_loss_margin
            )
            
            loss = self.contrastive_loss_weight * contrastive_loss + self.subset_loss_weight * subset_loss + self.exclusion_loss_weight * exclusion_loss
        else:
            loss = contrastive_loss

        self.log("subset_loss", subset_loss,
                 on_step=True, on_epoch=True, prog_bar=True,
                 batch_size=all_output['encoded'].size(0), sync_dist=True)
        
        self.log("subset_loss weighted", self.subset_loss_weight * subset_loss,
                 on_step=True, on_epoch=True, prog_bar=True,
                 batch_size=all_output['encoded'].size(0), sync_dist=True)

        self.log("exclusion_loss", exclusion_loss,
                    on_step=True, on_epoch=True, prog_bar=True,
                    batch_size=all_output['encoded'].size(0), sync_dist=True)
        
        self.log("exclusion_loss weighted", self.exclusion_loss_weight * exclusion_loss,
                    on_step=True, on_epoch=True, prog_bar=True,
                    batch_size=all_output['encoded'].size(0), sync_dist=True)

        self.log("contrastive_loss", contrastive_loss,
                 on_step=True, on_epoch=True, prog_bar=True,
                 batch_size=all_output['encoded'].size(0), sync_dist=True)

        return loss

    def validation_step(self, val_batch, batch_idx):

        if self.batch_strategy == 'group':
           shard_output = self._step_group(val_batch)
        else:
            shard_output = self._step(val_batch)

        if self.num_gpus > 1:
            gather_fn = partial(self.all_gather, sync_grads=True)
            all_output = {
                key: gather_with_var_len(
                    val,
                    base_gather_fn=gather_fn,
                    reduce_fn=(
                    reduce_pos_neg_masks if key in ['pos_mask', 'unallowed_neg_mask', 'subset_relation', 'exclusion_relation']
                    else apply_label_offset if key in ['query_indices', 'document_indices']
                    else reduce_embeddings))
                for key, val in shard_output.items()
            }
        else:
            all_output = shard_output

        if self.make_strict_neg_mask == 1 and self.batch_strategy == 'group':
            neg_mask = make_strict_neg_mask(all_output["pos_mask"], all_output["query_indices"])
        else:
            neg_mask = make_neg_mask(all_output["pos_mask"])

        if self.in_group_negative == 0 and self.batch_strategy == 'group':
            assert neg_mask.shape == all_output['unallowed_neg_mask'].shape
            neg_mask = neg_mask * all_output['unallowed_neg_mask']

        with torch.no_grad():
            contrastive_loss = self.loss_fn(
                all_output['encoded'], 
                all_output["pos_mask"], 
                neg_mask
            )

            subset_loss = torch.tensor(0.0, device=self.device)
            exclusion_loss = torch.tensor(0.0, device=self.device)

            if self.batch_strategy == 'group':

                assert len(all_output['encoded']) == len(all_output['query_indices']) + len(all_output['document_indices'])

                assert len(all_output['query_indices']) == len(all_output['subset_relation'])
                assert len(all_output['query_indices']) == len(all_output['exclusion_relation'])

                subset_loss = self.subset_loss_fn(
                    all_output['encoded'],
                    all_output['query_indices'],
                    all_output['document_indices'],
                    all_output['subset_relation'],
                    scale = self.subset_loss_scale,
                    normalization = self.normalization,
                    margin = self.subset_loss_margin
                )

                exclusion_loss = self.exclusion_loss_fn(
                    all_output['encoded'],
                    all_output['query_indices'],
                    all_output['document_indices'],
                    all_output['exclusion_relation'],
                    scale = self.exclusion_loss_scale,
                    normalization = self.normalization,
                    margin = self.exclusion_loss_margin
                )
            
                loss = self.contrastive_loss_weight * contrastive_loss + self.subset_loss_weight * subset_loss + self.exclusion_loss_weight * exclusion_loss

            else:
                loss = contrastive_loss

        self.validation_step_contrastive_loss_outputs.append(contrastive_loss.item())
        self.validation_step_subset_loss_outputs.append(subset_loss.item())
        self.validation_step_exclusion_loss_outputs.append(exclusion_loss.item())

        return loss.item()

    def on_validation_epoch_end(self) -> None:
        # take the mean of the loss
        val_contrastive_loss = np.mean(self.validation_step_contrastive_loss_outputs)
        val_subset_loss = np.mean(self.validation_step_subset_loss_outputs)
        val_exclusion_loss = np.mean(self.validation_step_exclusion_loss_outputs)
        self.log("val_contrastive_loss_mean", val_contrastive_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_subset_loss_mean", val_subset_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_subset_loss_mean_weighted", self.subset_loss_weight * val_subset_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_exclusion_loss_mean", val_exclusion_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log("val_exclusion_loss_mean_weighted", self.exclusion_loss_weight * val_exclusion_loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.validation_step_contrastive_loss_outputs.clear()
        self.validation_step_subset_loss_outputs.clear()
        self.validation_step_exclusion_loss_outputs.clear()

    def test_step(self, test_batch, batch_idx):
        return self._step(test_batch)

    def predict_step(self, batch, batch_idx):
                
        if type(batch["input_ids"]) == list:
            input_ids = torch.stack(batch["input_ids"], dim=0)
            attention_mask = torch.stack(batch["attention_mask"], dim=0)
        else:
            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]

        encoded = self(input_ids, attention_mask)
        encoded = encoded / encoded.norm(dim=1, keepdim=True)

        batch["encoded"] = encoded
        return batch
    
    def load_checkpoint(self, checkpoint_path):
        """
        Load model weights from a checkpoint.
        
        :param checkpoint_path: path to the checkpoint file
        """
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.load_state_dict(checkpoint['state_dict'])
            print(f"Loaded model weights from checkpoint: {checkpoint_path}")
        else:
            print(f"Checkpoint path does not exist: {checkpoint_path}")