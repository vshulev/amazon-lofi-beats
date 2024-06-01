import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from datasets import Dataset as HFDataset
from transformers import PreTrainedTokenizerFast, BertForMaskedLM
from tqdm import tqdm

torch.set_grad_enabled(False)

device = "mps"
batch_size = 32
max_len = 660

df = pd.read_csv("~/Downloads/BOLD_embeddings_seed.tsv", sep="\t")
# Truncate nucraw to max 660 characters
df["nucraw"] = df["nucraw"].apply(lambda x: x[:max_len])
# Add a space at every 4 characters in the sequence
df["nucraw"] = df["nucraw"].apply(lambda x: " ".join([x[i:i+4] for i in range(0, len(x), 4)]))

model = BertForMaskedLM.from_pretrained("LofiAmazon/BarcodeBERT-Entire-BOLD")
model.to(device)
model.eval()

tokenizer = PreTrainedTokenizerFast.from_pretrained("LofiAmazon/BarcodeBERT-Entire-BOLD")
tokenizer.add_special_tokens({"pad_token": "<UNK>"})

class VectorDataset(Dataset):
    def __init__(self, df: pd.DataFrame):
        self.df: pd.DataFrame = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return row["nucraw"], row["processid"]

dataset = VectorDataset(df)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

dataset_dict = {
    "processid": [],
    "embeddings": []
}

count = 0
for nucleotides, processids in tqdm(dataloader):
    inputs = tokenizer(nucleotides, return_tensors="pt", padding=True)

    if count < 643:
        count += 1
        continue

    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model(**inputs).hidden_states[-1]
    outputs = outputs.mean(1).cpu().numpy()

    dataset_dict["processid"].extend(processids)
    dataset_dict["embeddings"].extend([outputs[i, :] for i in range(outputs.shape[0])])

    if count % 100 == 0:
        hf_dataset = HFDataset.from_dict(dataset_dict)
        hf_dataset.push_to_hub("LofiAmazon/BOLD-Embeddings")

    count += 1
