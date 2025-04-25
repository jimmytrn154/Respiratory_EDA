import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from sklearn.model_selection import train_test_split
from multiprocessing import freeze_support
from Audio_Module import get_dataloader, SpectrogramEmbedding

class RespiratoryTransformer(nn.Module):
    def __init__(self, n_mels=80, d_model=128, nhead=8, num_layers=3, num_classes=2):
        super().__init__()
        self.embed = SpectrogramEmbedding(n_mels, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=0.1,
            activation="relu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, specs):
        x = self.embed(specs)
        x = x.permute(1, 0, 2)
        x = self.transformer(x)
        x = x.mean(dim=0)
        return self.classifier(x)


def load_data_and_labels(wav_dir: Path, excel_path: Path):
    # collect files
    all_files = sorted(str(p) for p in wav_dir.rglob("*.wav"))

    # load sheet
    df = pd.read_excel(excel_path, dtype=str)
    id_to_label_str = dict(zip(df["Patient ID"], df["Diagnosis"]))

    # map to string labels
    labels_str = []
    for f in all_files:
        pid = Path(f).stem[:4]
        if pid not in id_to_label_str:
            raise KeyError(f"Missing label for Patient ID '{pid}' in {excel_path.name}")
        labels_str.append(id_to_label_str[pid])

    # encode string labels to ints
    unique_labels = sorted(set(labels_str))
    label2idx = {lbl: i for i, lbl in enumerate(unique_labels)}
    labels = [label2idx[s] for s in labels_str]
    num_classes = len(unique_labels)

    # split
    train_files, val_files, train_labels, val_labels = train_test_split(
        all_files, labels, test_size=0.2, random_state=42
    )
    return train_files, val_files, train_labels, val_labels, num_classes

if __name__ == "__main__":
    freeze_support()

    wav_dir    = Path(r"C:\Users\jimmy\Documents\Tài liệu\vital docs\Side Project\Resp\RespiratoryDatabase@TR")
    excel_path = Path(r"C:\Users\jimmy\Documents\Tài liệu\vital docs\Side Project\Resp\COPD.xlsx")

    # load & split
    train_files, val_files, train_labels, val_labels, num_classes = load_data_and_labels(wav_dir, excel_path)

    # dataloaders
    dataset_kwargs = dict(sample_rate=16000, n_mels=80, segment_length=5.0)
    train_loader = get_dataloader(
        train_files, train_labels, batch_size=16, shuffle=True, num_workers=4, **dataset_kwargs
    )
    val_loader = get_dataloader(
        val_files, val_labels, batch_size=16, shuffle=False, num_workers=4, **dataset_kwargs
    )

    # model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RespiratoryTransformer(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # training loop
    num_epochs = 10
    for epoch in range(1, num_epochs + 1):
        model.train()
        train_loss = 0.0
        for specs, labels in train_loader:
            specs, labels = specs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(specs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * specs.size(0)
        avg_train = train_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.no_grad():
            for specs, labels in val_loader:
                specs, labels = specs.to(device), labels.to(device)
                outputs = model(specs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * specs.size(0)
                preds = outputs.argmax(dim=1)
                correct += (preds == labels).sum().item()
        avg_val = val_loss / len(val_loader.dataset)
        val_acc = correct / len(val_loader.dataset)

        print(
            f"Epoch {epoch}/{num_epochs} | Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | Val Acc: {val_acc:.4f}"
        )