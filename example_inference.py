import torch
from transformers import PreTrainedTokenizerFast, BertForMaskedLM

model = BertForMaskedLM.from_pretrained("LofiAmazon/BarcodeBERT-Entire-BOLD")
model.eval()

tokenizer = PreTrainedTokenizerFast.from_pretrained("LofiAmazon/BarcodeBERT-Entire-BOLD")

inputs = tokenizer("ACTG GTCA GGAA", return_tensors="pt")

with torch.no_grad():
    outputs = model(**inputs).hidden_states[-1]
    outputs = outputs.mean(1).squeeze()

print(outputs)
