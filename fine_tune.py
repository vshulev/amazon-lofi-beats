import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import PreTrainedTokenizerFast, BertForMaskedLM
from datasets import load_dataset
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import wandb
from huggingface_hub import PyTorchModelHubMixin


# Ecolayer feautres
ecolayers = [
    "median_elevation_1km",
    "human_footprint",
    "population_density_1km",
    "annual_precipitation",
    "precipitation_seasonality",
    "annual_mean_air_temp",
    "temp_seasonality",
]


class DNAEnvDataset:
    def __init__(self, dataset, ecolayers):
        self.dataset = dataset
        self.ecolayers = torch.from_numpy(ecolayers).to(torch.float32)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        dna_seq = sample["nucraw"]
        env_data = self.ecolayers[idx]
        label = sample["label"]

        return {
            "dna_seq": dna_seq,
            "env_data": env_data,
            "label": label,
        }

# Define the model architecture
class DNASeqClassifier(nn.Module, PyTorchModelHubMixin):
    def __init__(self, bert_model, env_dim, num_classes):
        super(DNASeqClassifier, self).__init__()
        self.bert = bert_model
        self.env_dim = env_dim
        self.num_classes = num_classes
        self.fc = nn.Linear(768 + env_dim, num_classes)

    def forward(self, bert_inputs, env_data):
        outputs = self.bert(**bert_inputs)
        dna_embeddings = outputs.hidden_states[-1].mean(1)
        combined = torch.cat((dna_embeddings, env_data), dim=1)
        logits = self.fc(combined)

        return logits


# Training loop
def train(
    model,
    tokenizer,
    device,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    epochs=3
):
    model.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        correct_predictions = 0
        total_predictions = 0

        for batch in tqdm(train_loader):
            optimizer.zero_grad()

            bert_inputs = tokenizer(batch["dna_seq"], padding=True, return_tensors="pt")
            bert_inputs = {key: val.to(device) for key, val in bert_inputs.items()}
            env_data = batch["env_data"].to(device)
            labels = batch["label"].to(device)

            logits = model(bert_inputs, env_data)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, preds = torch.max(logits, dim=1)
            correct_predictions += torch.sum(preds == labels)
            total_predictions += labels.size(0)

            wandb.log({ "batch_loss": loss.item() })

        train_accuracy = correct_predictions / total_predictions
        print(f"Epoch {epoch+1}, Training loss: {train_loss / len(train_loader)}, Training accuracy: {train_accuracy}")

        # Validation
        model.eval()
        val_loss = 0
        correct_predictions = 0
        total_predictions = 0

        with torch.no_grad():
            for batch in val_loader:
                bert_inputs = tokenizer(batch["dna_seq"], padding=True, return_tensors="pt")
                bert_inputs = {key: val.to(device) for key, val in bert_inputs.items()}
                env_data = batch["env_data"].to(device)
                labels = batch["label"].to(device)

                logits = model(bert_inputs, env_data)
                loss = criterion(logits, labels)

                val_loss += loss.item()
                _, preds = torch.max(logits, dim=1)
                correct_predictions += torch.sum(preds == labels)
                total_predictions += labels.size(0)

        val_accuracy = correct_predictions / total_predictions
        print(f"Epoch {epoch+1}, Validation loss: {val_loss / len(val_loader)}, Validation accuracy: {val_accuracy}")

        wandb.log({
            "epoch": epoch + 1,
            "train_loss": train_loss / len(train_loader),
            "train_accuracy": train_accuracy,
            "val_loss": val_loss / len(val_loader),
            "val_accuracy": val_accuracy
        })

def main(
        model: str,
        dataset: str,
        epochs: int,
        device: str,
        batch_size: int,
        lr: float,
        repo: str,
):
    # Load the dataset
    ds = load_dataset(dataset)
    train_dataset = ds["train"]
    val_dataset = ds["validation"]

    # Filter the dataset to remove sparse species genus
    train_dataset = train_dataset.filter(lambda x: x["genus"] is not None)
    val_dataset = val_dataset.filter(lambda x: x["genus"] is not None)
    genera = set(train_dataset["genus"])
    val_dataset = val_dataset.filter(lambda x: x["genus"] in genera)
    val_dataset = val_dataset.shuffle(seed=42).select(range(int(0.01 * len(val_dataset))))

    # Create labels
    label_map = {genus: i for i, genus in enumerate(genera)}
    train_dataset = train_dataset.map(lambda x: {"label": label_map[x["genus"]]})
    val_dataset = val_dataset.map(lambda x: {"label": label_map[x["genus"]]})

    # Load the tokenizer and model
    tokenizer = PreTrainedTokenizerFast.from_pretrained(model)
    tokenizer.add_special_tokens({"pad_token": "<UNK>"})

    bert_model = BertForMaskedLM.from_pretrained(model)

    # Normalize the environmental data
    def filter_columns(ds):
        return {key: ds[key] for key in ecolayers}
    scaler = StandardScaler()
    train_env_data = scaler.fit_transform(train_dataset.map(filter_columns, remove_columns=[col for col in train_dataset.column_names if col not in ecolayers]).to_pandas())
    val_env_data = scaler.transform(val_dataset.map(filter_columns, remove_columns=[col for col in val_dataset.column_names if col not in ecolayers]).to_pandas())

    train_dataset = DNAEnvDataset(train_dataset, train_env_data)
    val_dataset = DNAEnvDataset(val_dataset, val_env_data)

    num_classes = len(genera)
    clf_model = DNASeqClassifier(bert_model, env_dim=len(ecolayers), num_classes=num_classes)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(clf_model.parameters(), lr=lr)

    wandb.init(project="dna-env-finetune")

    train(clf_model, tokenizer, device, train_loader, val_loader, criterion, optimizer, epochs)

    # Save the model
    clf_model.push_to_hub(repo)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model", type=str, default="LofiAmazon/BarcodeBERT-Entire-BOLD",
    )
    parser.add_argument(
        "--dataset", type=str, default="LofiAmazon/BOLD-Embeddings-Ecolayers-Amazon",
    )
    parser.add_argument(
        "--epochs", type=int, default=3,
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
    )
    parser.add_argument(
        "--lr", type=float, default=2e-5,
    )
    parser.add_argument(
        "--repo", type=str, default="LofiAmazon/BarcodeBERT-Finetuned-Amazon",
    )

    args = parser.parse_args()

    main(**vars(args))
