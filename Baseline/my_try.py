import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
from sklearn.model_selection import train_test_split
import pandas as pd
from pathlib import Path
import torch.nn as nn
import torch.optim as optim
from multiprocessing import freeze_support

# Fix random seed for reproducibility
torch.manual_seed(42)

class AudioWaveformDataset(Dataset):
    def __init__(self, file_list, labels, sample_rate=16000, segment_length=5.0):
        self.file_list = file_list
        self.labels = labels
        self.sample_rate = sample_rate
        self.segment_samples = int(sample_rate * segment_length)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path = self.file_list[idx]
        label = self.labels[idx]
        waveform, sr = torchaudio.load(path)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)
        waveform = waveform.mean(dim=0)  # to mono

        # split into fixed-length segments
        segments = []
        total = waveform.size(0)
        for start in range(0, total - self.segment_samples + 1, self.segment_samples):
            segment = waveform[start:start + self.segment_samples]
            segments.append(segment)
        segments = torch.stack(segments)  # (num_segments, samples)
        return segments, label

def audio_collate_fn(batch):
    waveforms, labels = [], []
    for segments, lbl in batch:
        for seg in segments:
            waveforms.append(seg.numpy())
            labels.append(lbl)
    # process into input_values (and maybe attention_mask)
    inputs = processor(
        waveforms,
        sampling_rate=processor.feature_extractor.sampling_rate,
        return_tensors="pt",
        padding=True
    )
    # if attention_mask wasn’t created, make a full-ones mask
    if "attention_mask" not in inputs:
        inputs["attention_mask"] = torch.ones_like(inputs["input_values"])
    return inputs, torch.tensor(labels)

def create_dataloader(files, labels, batch_size=8, shuffle=False, num_workers=0, resample=False):
    ds = AudioWaveformDataset(files, labels, sample_rate=processor.feature_extractor.sampling_rate)
    if resample:
        counts = torch.tensor([labels.count(i) for i in sorted(set(labels))], dtype=torch.float)
        weights = 1.0 / counts
        sample_weights = torch.tensor([weights[l] for l in labels])
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        return DataLoader(ds, batch_size=batch_size, sampler=sampler,
                          num_workers=num_workers, collate_fn=audio_collate_fn)
    else:
        return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                          num_workers=num_workers, collate_fn=audio_collate_fn)

if __name__ == "__main__":
    freeze_support()

    # === Data & Labels ===
    wav_dir   = Path(r"C:\Users\jimmy\Documents\Tài liệu\vital docs\Side Project\Resp\RespiratoryDatabase@TR")
    excel_path= Path(r"C:\Users\jimmy\Documents\Tài liệu\vital docs\Side Project\Resp\COPD.xlsx")

    all_files = sorted(str(p) for p in wav_dir.rglob("*.wav"))
    df        = pd.read_excel(excel_path, dtype=str)
    id2lbl    = dict(zip(df["Patient ID"], df["Diagnosis"]))
    labels    = [id2lbl[Path(f).stem[:4]] for f in all_files]
    uniq      = sorted(set(labels))
    lbl2idx   = {l: i for i, l in enumerate(uniq)}
    labels    = [lbl2idx[l] for l in labels]

    train_files, val_files, train_lbls, val_lbls = train_test_split(
        all_files, labels, test_size=0.2, random_state=42
    )

    # === Model & Processor ===
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model     = Wav2Vec2ForSequenceClassification.from_pretrained(
                    "facebook/wav2vec2-base-960h",
                    num_labels=len(uniq)
                )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # === DataLoaders ===
    train_loader = create_dataloader(train_files, train_lbls, batch_size=8,
                                     shuffle=False, num_workers=0, resample=True)
    val_loader   = create_dataloader(val_files,   val_lbls,   batch_size=8,
                                     shuffle=False, num_workers=0, resample=False)

    # === Optimizer & Loss ===
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                     factor=0.5, patience=2)
    criterion = nn.CrossEntropyLoss()

    # === Training Loop ===
    for epoch in range(1, 11):
        model.train()
        train_loss = 0.0
        for inputs, lbls in train_loader:
            iv = inputs["input_values"].to(device)
            am = inputs["attention_mask"].to(device)
            lbls = lbls.to(device)

            optimizer.zero_grad()
            outputs = model(iv, attention_mask=am, labels=lbls)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * lbls.size(0)
        avg_train = train_loss / len(train_loader.dataset)

        model.eval()
        val_loss, correct = 0.0, 0
        with torch.no_grad():
            for inputs, lbls in val_loader:
                iv = inputs["input_values"].to(device)
                am = inputs["attention_mask"].to(device)
                lbls = lbls.to(device)
                outputs = model(iv, attention_mask=am, labels=lbls)
                val_loss += outputs.loss.item() * lbls.size(0)
                preds = outputs.logits.argmax(dim=1)
                correct += (preds == lbls).sum().item()
        avg_val = val_loss / len(val_loader.dataset)
        acc     = correct  / len(val_loader.dataset)
        scheduler.step(avg_val)

        print(f"Epoch {epoch} | Train Loss: {avg_train:.4f} | "
              f"Val Loss: {avg_val:.4f} | Val Acc: {acc:.4f}")
