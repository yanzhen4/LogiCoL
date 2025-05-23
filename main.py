import os

import numpy as np
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.strategies import DDPStrategy

from arguments import create_argparser, make_experiment_id
from datamodule.complex_query_dataset import ComplexQueryDataModule
from model.encoder import LitEncoder


def main(args):

    print("exclusion_loss_weight: ", args.exclusion_loss_weight)
    print("exclusion_loss_margin: ", args.exclusion_loss_margin)
    print("subset_loss_weight: ", args.subset_loss_weight)
    print("subset_loss_margin: ", args.subset_loss_margin)
    
    # weirdness with HuggingFace tokenizer when processing things in parallel
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.multiprocessing.set_sharing_strategy('file_system')

    # Set seed for each worker
    pl.seed_everything(args.random_seed, workers=True)

    # create experiment_dir
    experiment_id = make_experiment_id(args)

    experiment_dir = os.path.join(args.output_dir, experiment_id)
    if not os.path.exists(experiment_dir):
        os.makedirs(experiment_dir)

    # create cache directory to save tokenized dataset, queries, etc
    if not os.path.exists(args.cache_dir):
        os.makedirs(args.cache_dir)

    # Load model and data module for training and evaluation
    model = LitEncoder(args)
    
    dm = ComplexQueryDataModule(
        train_queries_path=args.train_queries,
        val_queries_path=args.val_queries,
        test_queries_path=args.test_queries,
        doc_path=args.documents,
        model_name_or_path=args.model_name,
        cache_dir=args.cache_dir,
        batch_strategy=args.batch_strategy,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.val_batch_size,
        max_seq_length=args.max_seq_length,
        num_workers=args.num_workers,
        alpha=args.alpha,
        sanity=args.sanity,
        query_key=args.query_key,
        doc_key=args.doc_key,
        num_pos=args.num_pos,
        loose_exclusion_loss=args.loose_exclusion_loss,
    )

     # Wandb logger
    logger = WandbLogger(
        project=args.project_name,
        name=f"{experiment_id}",
        save_dir=experiment_dir,
    )

    logger.watch(model)

    lr_monitor = LearningRateMonitor(logging_interval="step")

    # compute validation if needed, otherwise just skip it and save
    # every `period` checkpoints
    limit_val_batches = 0.0
    checkpoint_callback = ModelCheckpoint(
    dirpath=experiment_dir,
    save_on_train_epoch_end=True,
    save_top_k=-1,
    every_n_epochs=5,  # Save every 2 epochs
    save_weights_only=False,
    filename="{epoch}-{step}"
)

    precision = int(args.precision) if args.precision != "bf16" and args.precision != "fp16" else "bf16-mixed" 

    if 'bert' in args.model_name or 'e5' in args.model_name or 'contriever' in args.model_name or 'gte' in args.model_name:
        strategy = DDPStrategy(find_unused_parameters=True)
    else:
        strategy = DDPStrategy(find_unused_parameters=False) # if args.gpus > 1 else None

    trainer = pl.Trainer(
        default_root_dir=experiment_dir,
        max_epochs=args.num_epoch,
        logger=logger,
        enable_checkpointing=True,
        strategy=strategy,
        precision=precision,
        check_val_every_n_epoch=args.validate_every if args.validate else 1,
        callbacks=[lr_monitor, checkpoint_callback],
        accelerator='gpu',
        devices=args.gpus,
        accumulate_grad_batches=args.accumulate_grad_batches,
    )

    if args.train:
        trainer.fit(model, datamodule=dm)

    if args.evaluate:
        trainer.test(model, datamodule=dm)


if __name__ == '__main__':
    args = create_argparser()
    main(args)