import os

import numpy as np
import pandas as pd
from pinecone import Pinecone 
import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizerFast, BertForMaskedLM

device = "mps"
max_len = 660
dataset_path = "~/Downloads/BOLD_Public.19-Apr-2024.Amazon_Countries.tsv"

taxonomy_columns = ["species", "genus", "family", "order", "class",
                    "phylum", "kingdom"]

torch.set_grad_enabled(False)

pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index("amazon")

df = pd.read_csv(dataset_path, sep="\t")
# Keep only COI markers
df = df[df["marker_code"] == "COI-5P"]
# Remove any rows witohut nucraw
df = df.dropna(subset=["nucraw"])
# Replace all symbols in nucraw which are not A, C, G, T with N
df["nucraw"] = df["nucraw"].str.replace("[^ACGT]", "N", regex=True)
# Truncate trailing Ns from nucraw
df["nucraw"] = df["nucraw"].str.replace("N+$", "", regex=True)
# Truncate nucraw to max 660 characters
df["nucraw"] = df["nucraw"].apply(lambda x: x[:max_len])
# Add a space at every 4 characters in the sequence
df["nucraw"] = df["nucraw"].apply(lambda x: " ".join([x[i:i+4] for i in range(0, len(x), 4)]))

print(f"Total of {len(df)} entries.")

model = BertForMaskedLM.from_pretrained("LofiAmazon/BarcodeBERT-Entire-BOLD")
model.to(device)
model.eval()

tokenizer = PreTrainedTokenizerFast.from_pretrained("LofiAmazon/BarcodeBERT-Entire-BOLD")

embeddings = []
for nucleotides in tqdm(df["nucraw"]):
    inputs = tokenizer(nucleotides, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    outputs = model(**inputs).hidden_states[-1]
    outputs = outputs.mean(1).squeeze()

    embeddings.append(outputs.cpu().numpy())

# Add embeddings as a column to the dataframe
df["embeddings"] = embeddings

df.to_csv("amazon_data.tsv", sep="\t", index=False)

def mean_embeddings(group):
    return np.mean(group.tolist(), axis=0)

# Generate grouped dfs
df_by_species = df[df["species"].notna()].groupby("species")["embeddings"].apply(mean_embeddings).reset_index()
df_by_genus = df[df["genus"].notna()].groupby("genus")["embeddings"].apply(mean_embeddings).reset_index()

vectors_all = [{
    "id": df["processid"].iloc[i],
    "values": df["embeddings"].iloc[i],
    "metadata": {col: (df[col].iloc[i] if pd.notna(df[col].iloc[i]) else "") for col in taxonomy_columns},
} for i in range(len(df))]

vectors_by_species = [{
    "id": df_by_species["species"].iloc[i],
    "values": df_by_species["embeddings"].iloc[i],
} for i in range(len(df_by_species))]

vectors_by_genus = [{
    "id": df_by_genus["genus"].iloc[i],
    "values": df_by_genus["embeddings"].iloc[i],
} for i in range(len(df_by_genus))]

def upsert_paginated(index, vectors, namespace, page_size=100):
    for i in tqdm(range(0, len(vectors), page_size)):
        index.upsert(
            vectors=vectors[i:i+page_size],
            namespace=namespace,
        )

upsert_paginated(index, vectors_all, "all")

upsert_paginated(index, vectors_by_species, "by_species")

upsert_paginated(index, vectors_by_genus, "by_genus")
