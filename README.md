# LogiCoL

The official code repo for "LogiCoL: Logically-Informed Contrastive Learning for Set-based Dense Retrieval".

Our data are available at [this link](https://drive.google.com/drive/folders/1W_8PrA7CibJ4MuI41H4xnS5KAXfQ0Z9D?usp=drive_link).

## Installation

This project is implemented using `pytorch_lightning`. Please ensure you have Python 3.8 or higher installed.

To install all required packages, run the following command:

```bash
pip install -r requirements.txt
```

## Training

To train a dense retriever using **LogiCoL**, run the following command:
```bash
./script/train_quest_LogiCoL.sh
```

To train a dense retriever using the basic supervised constrastive learning (the **SupCon** baseline), run the following command:
```bash
./scripttrain_quest_SupCon.sh
```

### Inference
To inference using the trained retriever, run the following command:
```bash
./script/evaluate_quest.sh
```
