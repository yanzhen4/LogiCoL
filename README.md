# LogiCoL

The official code repo for "LogiCoL: Logically-Informed Contrastive Learning for Set-based Dense Retrieval".

## Installation

This project is implemented using `pytorch_lightning`. Please ensure you have Python 3.8 or higher installed.

To install all required packages, run the following command:

```bash
pip install -r requirements.txt
```

## Training

To train a dense retriever using **LogiCoL**, run the following command:
```bash
./script/train_quest_slurm.sh
```

### Inference
To inference using the trained retriever, run the following command:
```bash
./script/evaluate_quest.sh
```
