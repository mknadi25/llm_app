# src/data.py
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import ray
from ray.data import Dataset
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer

# Import your own modules
from src.config import STOPWORDS
from src import utils

# Define the default path to your config file
CONFIG_PATH = "config/config.yaml"


# In src/data.py

def load_data(
    config_path: str = CONFIG_PATH, dataset_loc: str = None, num_samples: int = None
) -> Dataset:
    """Load data from the source specified in the config file.

    Args:
        config_path (str): Path to the YAML configuration file.
        dataset_loc (str, optional): Location of the dataset, overrides config. Defaults to None.
        num_samples (int, optional): Number of samples to load, overrides config. Defaults to None.

    Returns:
        Dataset: The loaded dataset represented by a Ray Dataset.
    """
    # Load the configuration from the YAML file
    config = utils.load_config(config_path)

    # Get the dataset URL from the loaded config or use the provided argument
    if not dataset_loc:
        dataset_loc = config["data"]["dataset_loc"]
    if not num_samples:
        num_samples = config["data"].get("num_samples")  # Use .get() for optional keys

    # Read the data into a Ray Dataset
    ds = ray.data.read_csv(dataset_loc)
    ds = ds.random_shuffle(seed=1234)
    ds = ray.data.from_items(ds.take(num_samples)) if num_samples else ds

    print("Data loaded successfully!")
    return ds


def stratify_split(
    ds: Dataset,
    stratify: str,
    test_size: float,
    shuffle: bool = True,
    seed: int = 1234,
) -> Tuple[Dataset, Dataset]:
    """Split a dataset into train and test splits with equal
    amounts of data points from each class in the column we
    want to stratify on.
    
    (This function's internal logic remains unchanged)
    """

    def _add_split(df: pd.DataFrame) -> pd.DataFrame:
        train, test = train_test_split(df, test_size=test_size, shuffle=shuffle, random_state=seed)
        train["_split"] = "train"
        test["_split"] = "test"
        return pd.concat([train, test])

    def _filter_split(df: pd.DataFrame, split: str) -> pd.DataFrame:
        return df[df["_split"] == split].drop("_split", axis=1)

    grouped = ds.groupby(stratify).map_groups(_add_split, batch_format="pandas")
    train_ds = grouped.map_batches(_filter_split, fn_kwargs={"split": "train"}, batch_format="pandas")
    test_ds = grouped.map_batches(_filter_split, fn_kwargs={"split": "test"}, batch_format="pandas")

    train_ds = train_ds.random_shuffle(seed=seed)
    test_ds = test_ds.random_shuffle(seed=seed)

    return train_ds, test_ds


def clean_text(text: str, stopwords: List = STOPWORDS) -> str:
    """Clean raw text string.
    
    (This function's internal logic remains unchanged)
    """
    text = text.lower()
    pattern = re.compile(r"\b(" + r"|".join(stopwords) + r")\b\s*")
    text = pattern.sub(" ", text)
    text = re.sub(r"([!\"'#$%&()*\+,-./:;<=>?@\\\[\]^_`{|}~])", r" \1 ", text)
    text = re.sub("[^A-Za-z0-9]+", " ", text)
    text = re.sub(" +", " ", text)
    text = text.strip()
    text = re.sub(r"http\S+", "", text)
    return text


def tokenize(batch: Dict) -> Dict:
    """Tokenize the text input in our batch using a tokenizer.
    
    (This function's internal logic remains unchanged)
    """
    tokenizer = BertTokenizer.from_pretrained("allenai/scibert_scivocab_uncased", return_dict=False)
    encoded_inputs = tokenizer(batch["text"].tolist(), return_tensors="np", padding="longest")
    return dict(ids=encoded_inputs["input_ids"], masks=encoded_inputs["attention_mask"], targets=np.array(batch["tag"]))


def preprocess(df: pd.DataFrame, class_to_index: Dict) -> Dict:
    """Preprocess the data in our dataframe.
    
    (This function's internal logic remains unchanged)
    """
    df["text"] = df.title + " " + df.description
    df["text"] = df.text.apply(clean_text)
    df = df.drop(columns=["id", "created_on", "title", "description"], errors="ignore")
    df = df[["text", "tag"]]
    df["tag"] = df["tag"].map(class_to_index)
    outputs = tokenize(df)
    return outputs


class CustomPreprocessor:
    """Custom preprocessor class.
    
    (This class's internal logic remains unchanged)
    """
    def __init__(self, class_to_index={}):
        self.class_to_index = class_to_index or {}
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}

    def fit(self, ds):
        tags = ds.unique(column="tag")
        self.class_to_index = {tag: i for i, tag in enumerate(tags)}
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}
        return self

    def transform(self, ds):
        return ds.map_batches(preprocess, fn_kwargs={"class_to_index": self.class_to_index}, batch_format="pandas")