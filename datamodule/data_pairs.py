"""
Dataset class for proposition pair
"""
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from utils.data_utils import read_jsonl, convert_example_to_features

from pytorch_lightning import LightningDataModule


class DataPairDataset(LightningDataModule):
    """
    Query and Document Pair dataset
    """

    def __init__(self,
                 train_data_path: str,
                 val_data_path: str,
                 model_name_or_path: str,
                 max_seq_length: int = 512,
                 train_batch_size: int = 32,
                 eval_batch_size: int = 32,
                 random_positive: bool = False,
                 num_workers: int = 64
                 ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.train_data_path = train_data_path
        self.val_data_path = val_data_path

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.random_positive = random_positive
        self.num_workers = num_workers
        self.prepare_data()

    def prepare_data(self):
        self.train_examples = read_jsonl(self.train_data_path)
        self.val_examples = read_jsonl(self.val_data_path)

    def train_dataloader(self):
        return DataLoader(self.train_examples,
                          batch_size=self.train_batch_size,
                          collate_fn=self.convert_to_features,
                          shuffle=False,
                          num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.val_examples,
                          batch_size=self.eval_batch_size,
                          collate_fn=self.convert_to_features,
                          num_workers=self.num_workers)
    
    def convert_to_features(self, batch_examples):
        """

        :param batch_examples:
        :return:
        """
        return convert_example_to_features(batch_examples,
                                           tokenizer=self.tokenizer,
                                           max_seq_len=self.max_seq_length,
                                           generate_labels=True,
                                           random_positive=self.random_positive)