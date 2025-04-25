import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader
from typing import List
from torch import nn

class AudioSpectrogramDataset(Dataset):
    def __init__(
        self,
        file_list: List[str],
        labels: List[int],  # expects integer class labels
        sample_rate: int = 16000,
        n_fft: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
        segment_length: float = 5.0,
    ):
        assert len(file_list) == len(labels), "file_list and labels must be the same length"
        self.file_list = file_list
        self.labels = labels
        self.sample_rate = sample_rate
        self.segment_length = segment_length
        self.samples_per_segment = int(sample_rate * segment_length)
        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        path = self.file_list[idx]
        label = self.labels[idx]  # already an integer

        waveform, sr = torchaudio.load(path)
        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        waveform = waveform.mean(dim=0, keepdim=True)
        mel_spec = self.mel_transform(waveform)
        mel_spec_db = self.amplitude_to_db(mel_spec)

        frames_per_segment = int(self.samples_per_segment / self.mel_transform.hop_length)
        total_frames = mel_spec_db.size(-1)
        segments = []
        for start in range(0, total_frames, frames_per_segment):
            end = start + frames_per_segment
            if end <= total_frames:
                segments.append(mel_spec_db[:, :, start:end])
        segments = torch.stack(segments)  # (num_segments, 1, n_mels, frames)
        return segments, torch.tensor(label)


def collate_fn(batch):
    specs_list, labels_list = zip(*batch)
    all_specs = torch.cat(specs_list, dim=0)
    lengths = [spec.size(0) for spec in specs_list]
    all_labels = torch.tensor(labels_list).repeat_interleave(torch.tensor(lengths))
    return all_specs, all_labels


def get_dataloader(
    file_list: List[str],
    labels: List[int],
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    **dataset_kwargs
):
    dataset = AudioSpectrogramDataset(file_list, labels, **dataset_kwargs)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )
    return loader

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor):
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]

class SpectrogramEmbedding(nn.Module):
    def __init__(self, n_mels: int, d_model: int):
        super().__init__()
        self.proj = nn.Linear(n_mels, d_model)
        self.pos_encoder = PositionalEncoding(d_model)

    def forward(self, specs: torch.Tensor) -> torch.Tensor:
        specs = specs.squeeze(1)
        x = specs.permute(0, 2, 1)
        x = self.proj(x)
        x = self.pos_encoder(x)
        return x