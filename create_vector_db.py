import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizerFast, BertForMaskedLM
from tqdm import tqdm

torch.set_grad_enabled(False)

device = "mps"

df = pd.read_csv("~/Downloads/BOLD_embeddings_seed.tsv", sep="\t")
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
dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

for nucleotides, processids in tqdm(dataloader):
    inputs = tokenizer(nucleotides, return_tensors="pt", padding=True)

    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model(**inputs).hidden_states[-1]
    outputs = outputs.mean(1).squeeze().cpu().numpy()
