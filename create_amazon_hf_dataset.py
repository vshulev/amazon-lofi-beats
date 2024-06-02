from datasets import Dataset, DatasetDict
import numpy as np
import pandas as pd


data_path = "amazon_data.tsv"

df = pd.read_csv(data_path, sep="\t")
df["embeddings"] = df["embeddings"].apply(lambda x: np.array(list(map(float, x[1:-1].split()))))

dataset = Dataset.from_pandas(df)
dataset = dataset.remove_columns("Unnamed: 0")

train_test_split = dataset.train_test_split(test_size=0.2)
train_dataset = train_test_split["train"]
test_dataset = train_test_split["test"]
validation_test_split = test_dataset.train_test_split(test_size=0.5)
validation_dataset = validation_test_split["train"]
test_dataset = validation_test_split["test"]

dataset_dict = DatasetDict({
    'train': train_dataset,
    'validation': validation_dataset,
    'test': test_dataset
})

dataset_dict.push_to_hub("LofiAmazon/BOLD-Embeddings-Amazon")
