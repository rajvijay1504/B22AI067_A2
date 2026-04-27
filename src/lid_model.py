""" Task 1.1: Multi-Head Frame-Level Language Identification
Languages: Hindi (0)  |  English (1)
Architecture: Conformer-inspired Transformer with per-frame classification head """

from __future__ import annotations
import math
from pathlib import Path
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms as T
import numpy as np
from sklearn.metrics import f1_score


# Feature extraction
class MelFeatureExtractor(nn.Module):
    """
    Extracts 80-dim log-mel filterbank features at 10ms frame shift.
    Input : waveform (B, T)   @ cfg.sample_rate
    Output: (B, n_frames, 80)
    """
    def __init__(self, sample_rate: int = 16000, n_mels: int = 80,
                 win_length_ms: int = 25, hop_length_ms: int = 10):
        super().__init__()
        win_length = int(sample_rate * win_length_ms / 1000)
        hop_length = int(sample_rate * hop_length_ms / 1000)
        self.mel = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=512,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            power=2.0,
        )
        self.amp_to_db = T.AmplitudeToDB(stype="power", top_db=80)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        device = waveform.device
        mel = self.mel.cpu()(waveform.cpu()).to(device)
        mel = self.amp_to_db(mel)
        mel = mel.transpose(1, 2)         # (B, T, n_mels)
        # CMVN per utterance
        mel = (mel - mel.mean(dim=1, keepdim=True)) / (mel.std(dim=1, keepdim=True) + 1e-8)
        return mel


# Positional Encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 50000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))   # (1, max_len, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


# Conformer Feed-Forward
class FeedForward(nn.Module):
    def __init__(self, d_model: int, expansion: int = 4, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * expansion),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * expansion, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return x + 0.5 * self.net(x)


# Convolution Module (Conformer)
class ConvModule(nn.Module):
    def __init__(self, d_model: int, kernel_size: int = 31, dropout: float = 0.1):
        super().__init__()
        assert (kernel_size - 1) % 2 == 0, "kernel_size must be odd"
        self.ln = nn.LayerNorm(d_model)
        self.pw1 = nn.Conv1d(d_model, 2 * d_model, 1)
        self.glu = nn.GLU(dim=1)
        self.dw = nn.Conv1d(d_model, d_model, kernel_size,
                            padding=(kernel_size - 1) // 2, groups=d_model)
        self.bn = nn.BatchNorm1d(d_model)
        self.act = nn.SiLU()
        self.pw2 = nn.Conv1d(d_model, d_model, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, d)
        residual = x
        x = self.ln(x).transpose(1, 2)   # (B, d, T)
        x = self.glu(self.pw1(x))
        x = self.act(self.bn(self.dw(x)))
        x = self.drop(self.pw2(x))
        return residual + x.transpose(1, 2)


# Conformer Block
class ConformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.ff1  = FeedForward(d_model, dropout=dropout)
        self.attn = nn.MultiheadAttention(d_model, num_heads,
                                          dropout=dropout, batch_first=True)
        self.ln_attn = nn.LayerNorm(d_model)
        self.conv = ConvModule(d_model, kernel_size=31, dropout=dropout)
        self.ff2  = FeedForward(d_model, dropout=dropout)
        self.ln   = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.ff1(x)
        attn_out, _ = self.attn(x, x, x, key_padding_mask=key_padding_mask)
        x = self.ln_attn(x + attn_out)
        x = self.conv(x)
        x = self.ff2(x)
        return self.ln(x)


# Main LID Model
class FrameLevelLID(nn.Module):
    """
    Frame-level Language ID classifier.
    Input : (B, T_frames, 80)
    Output: (B, T_frames, num_classes)  — logits per frame
    """
    def __init__(self, feat_dim: int = 80, hidden_dim: int = 256,
                 num_heads: int = 8, num_layers: int = 4,
                 num_classes: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_proj = nn.Linear(feat_dim, hidden_dim)
        self.pos_enc    = PositionalEncoding(hidden_dim, dropout=dropout)
        self.conformer  = nn.ModuleList([
            ConformerBlock(hidden_dim, num_heads, dropout) for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor,
                padding_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.input_proj(x)          # (B, T, hidden)
        x = self.pos_enc(x)
        for block in self.conformer:
            x = block(x, key_padding_mask=padding_mask)
        return self.classifier(x)       # (B, T, num_classes)


# Training helper
def train_lid(model: FrameLevelLID, dataloader, optimizer, device: str,
              num_epochs: int = 20) -> FrameLevelLID:
    """
    Train the LID model on (mel_features, frame_labels) pairs.
    frame_labels: LongTensor of shape (B, T) with 0=Hindi, 1=English
    """
    model.to(device).train()
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    for epoch in range(num_epochs):
        total_loss, total_correct, total_frames = 0.0, 0, 0

        for mels, labels in dataloader:
            mels, labels = mels.to(device), labels.to(device)
            optimizer.zero_grad()

            logits = model(mels)               # (B, T, 2)
            B, T, C = logits.shape
            loss = criterion(logits.view(B * T, C), labels.view(B * T))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            preds = logits.argmax(-1)
            mask  = labels != -1
            total_correct += (preds[mask] == labels[mask]).sum().item()
            total_frames  += mask.sum().item()
            total_loss    += loss.item()

        acc = total_correct / max(total_frames, 1)
        print(f"[LID] Epoch {epoch+1:3d}/{num_epochs}  "
              f"Loss={total_loss/len(dataloader):.4f}  Acc={acc:.4f}")

    return model


def evaluate_lid(model: FrameLevelLID, dataloader, device: str) -> dict:
    """Returns frame-level accuracy and F1 per class."""
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for mels, labels in dataloader:
            mels = mels.to(device)
            logits = model(mels)              # (B, T, 2)
            preds  = logits.argmax(-1).cpu()  # (B, T)
            mask   = labels != -1
            all_preds.append(preds[mask])
            all_labels.append(labels[mask])

    all_preds  = torch.cat(all_preds).numpy()
    all_labels = torch.cat(all_labels).numpy()
    f1 = f1_score(all_labels, all_preds, average="macro")
    acc = (all_preds == all_labels).mean()
    print(f"[LID-EVAL] Macro-F1={f1:.4f}  Acc={acc:.4f}")
    return {"f1": f1, "accuracy": float(acc)}


# Inference: segment a waveform with language timestamps
@torch.no_grad()
def predict_language_segments(
    model: FrameLevelLID,
    feature_extractor: MelFeatureExtractor,
    waveform: torch.Tensor,       # (1, T) at 16 kHz
    device: str = "cpu",
    frame_shift_ms: int = 10,
    min_segment_frames: int = 10,
    chunk_frames: int = 1000,     # process 1000 frames (~10s) at a time
) -> list[dict]:
    """
    Returns list of dicts: [{"start_ms": ..., "end_ms": ..., "lang": "Hindi"/"English"}]
    """
    model.eval().to(device)
    feature_extractor.to(device)

    waveform = waveform.to(device)
    mels     = feature_extractor.cpu()(waveform.cpu()).to(waveform.device)
    # mels: (1, T_frames, 80) — process in chunks to avoid OOM
    T_frames = mels.shape[1]
    frame_labels = []

    for start in range(0, T_frames, chunk_frames):
        end = min(start + chunk_frames, T_frames)
        chunk = mels[:, start:end, :]          # (1, chunk, 80)
        logits_chunk = model(chunk)             # (1, chunk, 2)
        labels_chunk = logits_chunk[0].argmax(-1).cpu().numpy()
        frame_labels.extend(labels_chunk.tolist())

    lang_map  = {0: "Hindi", 1: "English"}
    segments  = []
    i, n      = 0, len(frame_labels)

    while i < n:
        lang = int(frame_labels[i])
        j = i + 1
        while j < n and int(frame_labels[j]) == lang:
            j += 1
        if (j - i) >= min_segment_frames:
            segments.append({
                "start_ms": i * frame_shift_ms,
                "end_ms":   j * frame_shift_ms,
                "lang":     lang_map[lang],
                "frames":   (i, j),
            })
        i = j

    return segments

# Synthetic dataset builder (for demo / fine-tuning without labelled data)
class SyntheticLIDDataset(torch.utils.data.Dataset):
    """
    Minimal synthetic dataset that assigns Hindi label to frames where
    energy is below a simple threshold (placeholder).
    Replace with real frame-labelled data for production use.
    """
    def __init__(self, wav_paths: list[str], sample_rate: int = 16000,
                 max_frames: int = 300):
        self.wav_paths  = wav_paths
        self.sr         = sample_rate
        self.max_frames = max_frames
        self.feat_ext   = MelFeatureExtractor(sample_rate)

    def __len__(self):
        return len(self.wav_paths)

    def __getitem__(self, idx):
        wav, sr = torchaudio.load(self.wav_paths[idx])
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)
        if wav.shape[0] > 1:
            wav = wav.mean(0, keepdim=True)

        with torch.no_grad():
            mel = self.feat_ext(wav)           # (1, T, 80)
        mel = mel[0, : self.max_frames]       # (T, 80)

        # Placeholder: label every frame as English=1 (replace with real labels)
        labels = torch.ones(mel.shape[0], dtype=torch.long)
        return mel, labels


# Factory / save / load
def build_lid_model(cfg: dict) -> FrameLevelLID:
    return FrameLevelLID(
        feat_dim=cfg["lid"]["feature_dim"],
        hidden_dim=cfg["lid"]["hidden_dim"],
        num_heads=cfg["lid"]["num_heads"],
        num_layers=cfg["lid"]["num_layers"],
        num_classes=cfg["lid"]["num_classes"],
    )


def save_lid_model(model: FrameLevelLID, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    print(f"[LID] Saved weights → {path}")


def load_lid_model(path: str, cfg: dict, device: str = "cpu") -> FrameLevelLID:
    model = build_lid_model(cfg)
    model.load_state_dict(torch.load(path, map_location=device))
    model.eval()
    return model