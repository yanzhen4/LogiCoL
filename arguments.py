import argparse
from model.losses import LOSS_CLASSES

def create_argparser():

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--project_name", type=str, default="setir",
                        help="Name of a wandb project, checkpoints are saved under this directory")
    parser.add_argument("--experiment_name", type=str, default="train",
                        help="name of the experiment to save the outputs + model checkpoint")
    parser.add_argument("--output_dir", type=str, default="output/",
                        help="Output directory to save model, arguments, and results")
    parser.add_argument("--cache_dir", type=str, default="output/cache/",
                        help="Cache directory to save tokenized results")
    parser.add_argument("--train", action='store_true', default=False,
                        help="Run training")
    parser.add_argument("--validate", action='store_true', default=False,
                        help="validate during training (after epochs)")
    parser.add_argument("--evaluate", action='store_true', default=False,
                        help="Evaluate on the test set")
    parser.add_argument("--validate_every", type=int, default=1,
                        help="Validate every N epochs")
    parser.add_argument("--random_seed", type=int, default=42,
                        help="Random seed for everything")
    parser.add_argument("--sanity", type=int, default=None,
                        help="Subsamples N examples from the dataset, used for debugging")
    parser.add_argument("--limit_val_batches", type=float, default=1.0,
                        help="Percentage of validation set used during training")
    parser.add_argument("--gpus", type=int, default=1,
                        help="Number of GPUs to use for training")
    parser.add_argument("--period", type=int, default=2,
                        help="Periodicity to save checkpoints when not validating")

    ### Training arguments
    parser.add_argument("--model_name", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Name of the huggingface transformer model to use")
    parser.add_argument("--batch_strategy", type=str, default="random", choices=["random", "group"],
                        help="Stategy to sample batches during training")
    parser.add_argument("--lr_scheduler", type=str, choices=["constant", "linear"], default="constant",
                        help="learning rate scheduler to use")
    parser.add_argument("--learning_rate", type=float, default=2e-5,
                        help="Specifies learning rate")
    parser.add_argument("--train_batch_size", type=int, default=32,
                        help="Per GPU batch size")
    parser.add_argument("--val_batch_size", type=int, default=32,
                        help="Per GPU validation batch size")
    parser.add_argument("--loss_type", type=str, default="supcon", choices=list(LOSS_CLASSES.keys()),
                        help="Type of loss / training objective for training the model. Affects how dataloader works.")
    parser.add_argument("--subset_loss_weight", type=float, default=0,
                        help="Weights of the subset regularization loss term")
    parser.add_argument("--subset_loss_margin", type=float, default=0,
                        help="Margin for the subset regularization term")
    parser.add_argument("--subset_loss_scale", default=0.5, type=float)
    parser.add_argument("--exclusion_loss_weight", type=float, default=0,
                        help="Weights of the exclusion regularization loss term")
    parser.add_argument("--exclusion_loss_scale", default=0.5, type=float)
    parser.add_argument("--exclusion_loss_margin", default=0, type=float)
    parser.add_argument("--loose_exclusion_loss", default=0, type=float)
    parser.add_argument("--contrastive_loss_weight", type=float, default=1,
                        help="Weights of the contrastive loss term")
    parser.add_argument("--load_checkpoint", default=False, action='store_true',
                        help="If True, will load the latest checkpoint")
    parser.add_argument("--precision", default="16", type=str,
                        help="Precision of model weights")
    parser.add_argument("--num_workers", type=int, default=1,
                        help="Number of workers to prefetch data")
    parser.add_argument("--num_epoch", type=int, default=20,
                        help="Number of epochs")
    parser.add_argument("--warmup_steps", type=int, default=10,
                        help="Number of warmup steps")
    parser.add_argument("--gradient_checkpointing", default=False, action="store_true",
                        help="If True, activates Gradient Checkpointing")
    parser.add_argument("--discard_empty_query", default=False, action="store_true",
                        help="If True, discard queries with no positives in training set")
    parser.add_argument("--temperature", type=float, default=0.01,
                        help="Temperature to use for SupCon loss")
    parser.add_argument("--save_top_k_ckpts", default=1, type=int,
                        help="Number of checkpoints to save")
    parser.add_argument("--accumulate_grad_batches", default=1, type=int,
                        help="Number of batches to accumulate gradients over")
    # parser.add_argument("--make_strict_neg_mask", default=1, type=int, choices=[0, 1],
    #                 help="whether to normalize query-document similarity scores when computing subset and exclusion loss")
    parser.add_argument("--in_group_negative", default=1, type=int, choices=[0, 1],
                        help="Whether queries are negative examples for each other")
    parser.add_argument("--normalization", default=0, type=int, choices=[0, 1],
                        help="whether to normalize query-document similarity scores when computing subset and exclusion loss")
    
    
    ### Data arguments
    parser.add_argument("--train_queries", type=str, default="data/dbpedia/dbpedia_train.jsonl",
                        help="path to training queries")
    parser.add_argument("--val_queries", type=str, default="data/dbpedia/dbpedia_validation.jsonl",
                        help="path to validation queries")
    parser.add_argument("--test_queries", type=str, default="data/dbpedia/dbpedia_test.jsonl",
                        help="path to test queries")
    parser.add_argument("--documents", type=str, default="data/dbpedia/dbpedia_wiki_text_w_id.jsonl",
                        help="the documents in the retrieval corpus")
    parser.add_argument("--query_key", type=str, default="nl_query",
                        help="key to the query in the jsonl file")
    parser.add_argument("--doc_key", type=str, default="text",
                        help="key to the document in the jsonl file")
    parser.add_argument("--num_pos", type=int, default=1,
                        help="number of positive docs per query")
    parser.add_argument("--alpha", type=float, default=0,
                        help="alpha value for the mix strategy")
                        
    ### Model Hyperparams
    parser.add_argument("--max_seq_length", default=None, type=int,
                        help="Maximum input sequence length of inputs to the encoder model.")
    parser.add_argument("--mlp_hidden_dim", default=None, type=int,
                        help="Dimension of mlp layer after pooling. If None, match the encoder output dim.")
    parser.add_argument("--final_output_dim", default=None, type=int,
                        help="Dimension of mlp layer after pooling. If None, match the encoder output dim.")

    ### Unused arguments
    parser.add_argument("--learning_rate_decay_gamma", type=float, default=0.9,
                        help="Gamma for exponential decay after each epoch.")

    ### For evaluation pipeline
    parser.add_argument("--weights_name", type=str, default=None)

    parser.add_argument("--base_model_name", type=str, default='sentence-transformers/sentence-t5-base', )

    parser.add_argument("--device_id", type=int, default=0, )

    args = parser.parse_args()

    return args


def make_experiment_id(args):
    id_strings = []
    total_bs = args.train_batch_size * args.gpus
    if args.train:
        model_name_compact = args.model_name.split("/")[-1]
        id_strings.append(args.experiment_name)
        id_strings.append(f"{model_name_compact}")
        id_strings.append(f"lr{args.learning_rate}")
        id_strings.append(f"bs{total_bs}")
        id_strings.append(f"len{args.max_seq_length}")
        id_strings.append(f"{args.precision}")

        if args.final_output_dim is not None:
            id_strings.append(f"dim{args.final_output_dim}")

    elif args.evaluate:
        # TODO(sihaoc)
        id_strings.append("eval")

    return "_".join(id_strings)
