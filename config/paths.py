#####################
###  MODEL paths  ###
#####################

MODEL_BASE_DIR = "/shared/sihaoc/project/set_ir/Set-based-Retrieval/output"

MODEL_PATHS = {
    # GTR-base, finetuned on QUEST
    # https://wandb.ai/cogcomp/setir/runs/o0cp3lix
    "gtr-base-quest-1": "quest_gtr-t5-base_lr0.0001_bs64_32/epoch=epoch=4-step=step=105.ckpt"
}

#####################
###  QUEST paths  ###
#####################
QUEST_TRAIN = "/shared/sihaoc/project/set_ir/Set-based-Retrieval/data/quest/quest_train.jsonl"
QUEST_VAL = "/shared/sihaoc/project/set_ir/Set-based-Retrieval/data/quest/quest_val.jsonl"
QUEST_TEST = "/shared/sihaoc/project/set_ir/Set-based-Retrieval/data/quest/quest_test.jsonl"
QUEST_DOC = "/shared/sihaoc/project/set_ir/Set-based-Retrieval/data/quest/quest_text_w_id.jsonl"
QUEST_DOC_TRAIN = "/shared/sihaoc/project/set_ir/Set-based-Retrieval/data/quest/quest_text_train.jsonl"

#####################
### DBPEDIA paths ###
#####################

DBPEDIA_TITLE2IDX = "/shared/yanzhen4/construct_queries/data/dbpedia_wiki_title2idx.jsonl"
DBPEDIA_TITLE2TEXT = "/shared/yanzhen4/construct_queries/data/dbpedia_wiki_text.jsonl"
DBPEDIA_TEXT = "data/dbpedia/dbpedia_wiki_text_w_id.jsonl"
DBPEDIA_DIR = "data/dbpedia/"

DBPEDIA_QUERY_DIR = "/shared/yanzhen4/construct_queries/data/complex_queries_gt/"
DBPEDIA_QUERY_FILE_PATTERN = "complex_queries_gt_{}.jsonl"
DBPEDIA_QUERY_SPLIT = {
    "train": [
        "album",
        "athlete",
        "building",
        "company",
        "educationalinstitution",
        "meanoftransportation",
        "naturalplace",
        "officeholder",
        "village",
        "writtenwork"
    ],
    "validation": [
        "animal"
    ],
    "test": [
        "film",
        "plant",
    ]
}

