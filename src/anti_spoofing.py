"""Part IV: Adversarial Robustness & Spoofing Detection

Task 4.1: LFCC/CQCC feature extraction + lightweight CM binary classifier
Task 4.2: FGSM adversarial perturbation to fool LID at SNR > 40 dB

EER target: < 10%
"""

from __future__ import annotations
import math
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.functional as TF
import torchaudio.transforms as Transforms
import librosa
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d


# 1.  LFCC Feature Extraction
class LFCCExtractor:
    """
    Linear Frequency Cepstral Coefficients.
    Uses a linear (not mel) filterbank before DCT – more sensitive to fine
    spectral detail needed for spoofing detection.
    """

    def __init__(self, sample_rate: int = 16000, n_fft: int = 512,
                 hop_length: int = 160, n_linear_filters: int = 70,
                 n_coeffs: int = 60, f_min: float = 0.0,
                 f_max: Optional[float] = None):
        self.sample_rate      = sample_rate
        self.n_fft            = n_fft
        self.hop_length       = hop_length
        self.n_linear_filters = n_linear_filters
        self.n_coeffs         = n_coeffs
        self.f_max            = f_max or sample_rate / 2.0
        self._build_filterbank()

    def _build_filterbank(self):
        freqs = np.linspace(self.f_min if hasattr(self, 'f_min') else 0.0,
                            self.f_max, self.n_linear_filters + 2)
        fft_freqs = np.fft.rfftfreq(self.n_fft, d=1.0 / self.sample_rate)
        fb = np.zeros((self.n_linear_filters, len(fft_freqs)), dtype=np.float32)
        for m in range(self.n_linear_filters):
            lo, ctr, hi = freqs[m], freqs[m+1], freqs[m+2]
            for k, f in enumerate(fft_freqs):
                if lo <= f <= ctr:
                    fb[m, k] = (f - lo) / (ctr - lo + 1e-8)
                elif ctr < f <= hi:
                    fb[m, k] = (hi - f) / (hi - ctr + 1e-8)
        self.filterbank = fb   # (n_filters, n_fft//2+1)

    def __call__(self, waveform: np.ndarray) -> np.ndarray:
        """
        waveform: (T,) float32
        Returns: (n_coeffs, n_frames) LFCC matrix
        """
        # STFT magnitude
        stft = np.abs(librosa.stft(
            waveform, n_fft=self.n_fft, hop_length=self.hop_length,
            win_length=self.n_fft, window="hann",
        )) ** 2                                  # (n_fft//2+1, n_frames)

        # Linear filterbank
        filtered = self.filterbank @ stft        # (n_filters, n_frames)
        log_filtered = np.log(filtered + 1e-8)

        # DCT (Type-II) → LFCC
        from scipy.fftpack import dct
        lfcc = dct(log_filtered, type=2, axis=0, norm="ortho")[:self.n_coeffs]
        return lfcc.astype(np.float32)           # (n_coeffs, n_frames)


def extract_lfcc_tensor(
    wav_path: str,
    n_coeffs: int = 60,
    sample_rate: int = 16000,
    max_frames: int = 300,
) -> torch.Tensor:
    """
    Load wav → compute LFCC → return (1, n_coeffs, max_frames) tensor.
    """
    waveform, sr = torchaudio.load(wav_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(0, keepdim=True)
    if sr != sample_rate:
        waveform = TF.resample(waveform, sr, sample_rate)

    wav_np = waveform.squeeze().numpy()
    extractor = LFCCExtractor(sample_rate=sample_rate, n_coeffs=n_coeffs)
    lfcc = extractor(wav_np)                    # (n_coeffs, T_frames)

    # Pad or trim to max_frames
    T = lfcc.shape[1]
    if T < max_frames:
        lfcc = np.pad(lfcc, ((0, 0), (0, max_frames - T)))
    else:
        lfcc = lfcc[:, :max_frames]

    return torch.from_numpy(lfcc).unsqueeze(0)  # (1, n_coeffs, max_frames)


# 2.  CQCC wrapper (uses python-cqt library)
def extract_cqcc(waveform: np.ndarray, sample_rate: int = 16000,
                 n_coeffs: int = 60) -> np.ndarray:
    """
    Constant-Q Cepstral Coefficients (fallback to LFCC if nnAudio unavailable).
    """
    try:
        import nnAudio.features
        cqt_layer = nnAudio.features.cqt.CQT(
            sr=sample_rate, hop_length=512, fmin=32.7, n_bins=84,
            bins_per_octave=12,
        )
        wav_t = torch.from_numpy(waveform).unsqueeze(0)
        cqt_mag = cqt_layer(wav_t).squeeze().numpy()
        from scipy.fftpack import dct
        log_cqt = np.log(np.abs(cqt_mag) + 1e-8)
        cqcc = dct(log_cqt, type=2, axis=0, norm="ortho")[:n_coeffs]
        return cqcc.astype(np.float32)
    except ImportError:
        print("[CQCC] nnAudio not installed – falling back to LFCC")
        extractor = LFCCExtractor(n_coeffs=n_coeffs)
        return extractor(waveform)


# 3.  Countermeasure (CM) Binary Classifier
class SpoofCM(nn.Module):
    """
    Lightweight CNN + LSTM anti-spoofing classifier.
    Input : (B, 1, n_coeffs, n_frames)  LFCC feature map
    Output: (B, 2)  logits  [bonafide, spoof]
    """

    def __init__(self, n_coeffs: int = 60, n_frames: int = 300,
                 hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((2, 2)),

            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, n_frames // 16)),
        )
        # After adaptive pool: (B, 128, 4, n_frames//16)
        cnn_out_dim = 128 * 4
        self.lstm = nn.LSTM(cnn_out_dim, hidden_dim, num_layers=2,
                            batch_first=True, bidirectional=True,
                            dropout=dropout)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, n_coeffs, n_frames)
        feat = self.cnn(x)                     # (B, 128, 4, T')
        B, C, H, T = feat.shape
        feat = feat.view(B, C * H, T).transpose(1, 2)  # (B, T', C*H)
        _, (h, _) = self.lstm(feat)
        h_cat = torch.cat([h[-2], h[-1]], dim=-1)  # (B, hidden*2)
        return self.classifier(h_cat)


def train_spoof_cm(
    model: SpoofCM,
    bonafide_paths: List[str],
    spoof_paths: List[str],
    device: str = "cpu",
    num_epochs: int = 30,
    lr: float = 1e-3,
) -> SpoofCM:
    """Train the CM on bonafide (real) vs spoof (synthesised) audio."""
    device = "cpu"
    model.to(device).train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)

    # Build dataset
    data = []
    for p in bonafide_paths:
        data.append((extract_lfcc_tensor(p).to(device), torch.tensor(0)))  # 0=bonafide
    for p in spoof_paths:
        data.append((extract_lfcc_tensor(p).to(device), torch.tensor(1)))  # 1=spoof

    for epoch in range(num_epochs):
        np.random.shuffle(data)
        total_loss, correct = 0.0, 0

        for feat, label in data:
            feat  = feat.unsqueeze(0).to(device)
            label = label.unsqueeze(0).to(device)
            optimizer.zero_grad()
            logit = model(feat)
            loss  = criterion(logit, label)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct    += (logit.argmax(-1) == label).item()

        scheduler.step()
        acc = correct / len(data)
        if (epoch + 1) % 5 == 0:
            print(f"[CM] Epoch {epoch+1:3d}/{num_epochs}  "
                  f"Loss={total_loss/len(data):.4f}  Acc={acc:.4f}")

    return model


def compute_eer(
    model: SpoofCM,
    bonafide_paths: List[str],
    spoof_paths: List[str],
    device: str = "cpu",
) -> float:
    """
    Compute Equal Error Rate (EER).  Must be < 10% to pass.
    """
    device = "cpu"
    model.eval().to(device)
    scores, labels_list = [], []

    with torch.no_grad():
        for p in bonafide_paths:
            feat = extract_lfcc_tensor(p).unsqueeze(0).to(device)
            logit = model(feat)
            score = F.softmax(logit, dim=-1)[0, 0].item()  # bonafide score
            scores.append(score)
            labels_list.append(0)

        for p in spoof_paths:
            feat = extract_lfcc_tensor(p).unsqueeze(0).to(device)
            logit = model(feat)
            score = F.softmax(logit, dim=-1)[0, 0].item()
            scores.append(score)
            labels_list.append(1)

    scores_np = np.array(scores)
    labels_np = np.array(labels_list)

    fpr, tpr, _ = roc_curve(labels_np, scores_np, pos_label=0)
    fnr = 1 - tpr
    eer = brentq(lambda x: interp1d(fpr, fnr - x)(x), 0.0, 1.0)
    print(f"[CM-EER] EER = {eer*100:.2f}%  (threshold < 10%)")
    return float(eer)


# 4.  FGSM Adversarial Perturbation (Task 4.2)
def compute_snr_db(signal: torch.Tensor, noise: torch.Tensor) -> float:
    """SNR in dB between a signal and additive noise."""
    s_pow = signal.pow(2).mean() + 1e-12
    n_pow = noise.pow(2).mean() + 1e-12
    return 10.0 * math.log10(s_pow.item() / n_pow.item())


def fgsm_attack_lid(
    lid_model: nn.Module,
    feature_extractor,            # MelFeatureExtractor
    waveform: torch.Tensor,       # (1, T)  at 16 kHz
    true_label: int,              # 0=Hindi, 1=English
    target_label: int,            # label we want to induce
    device: str = "cpu",
    epsilon_init: float = 0.001,
    epsilon_max: float = 0.05,
    n_steps: int = 20,
    target_snr_db: float = 40.0,
) -> Tuple[torch.Tensor, float, float]:
    """
    Find minimum ε such that:
      (a) FGSM perturbation flips LID from true_label → target_label on ≥50% frames
      (b) SNR of perturbed signal vs original remains ≥ target_snr_db

    Returns: (perturbed_waveform, epsilon, achieved_snr_db)
    """
    lid_model.eval().to(device)
    feature_extractor.to(device)
    waveform = waveform.to(device)

    criterion = nn.CrossEntropyLoss()
    epsilons  = np.linspace(epsilon_init, epsilon_max, n_steps)

    for eps in epsilons:
        wav_adv = waveform.clone().detach().requires_grad_(True)

        # Forward through feature extractor + LID
        mels   = feature_extractor(wav_adv)        # (1, T_frames, 80)
        logits = lid_model(mels)                   # (1, T_frames, 2)
        B, T, C = logits.shape

        # Target: flip all frames to target_label
        tgt_labels = torch.full((B, T), target_label, dtype=torch.long, device=device)
        loss = criterion(logits.view(B * T, C), tgt_labels.view(B * T))
        loss.backward()

        with torch.no_grad():
            grad_sign  = wav_adv.grad.sign()
            perturbation = eps * grad_sign
            wav_perturbed = (waveform + perturbation).clamp(-1.0, 1.0)

        # Check SNR
        snr = compute_snr_db(waveform, perturbation)

        # Check LID flip rate
        with torch.no_grad():
            mels_adv   = feature_extractor(wav_perturbed)
            logits_adv = lid_model(mels_adv)
            pred_labels = logits_adv.argmax(-1)          # (1, T)
            flip_rate   = (pred_labels == target_label).float().mean().item()

        print(f"[FGSM] ε={eps:.5f}  SNR={snr:+.1f}dB  flip_rate={flip_rate:.2%}")

        if flip_rate >= 0.5 and snr >= target_snr_db:
            print(f"[FGSM] ✓ Found adversarial ε={eps:.5f} "
                  f"(SNR={snr:.1f}dB ≥ {target_snr_db}dB, flip={flip_rate:.0%})")
            return wav_perturbed.detach(), float(eps), float(snr)

        # If SNR constraint breached – try smaller eps
        if snr < target_snr_db:
            print(f"[FGSM] SNR constraint violated at ε={eps:.5f}; stopping search")
            break

    # Return best attempt even if constraints not fully met
    wav_adv_final = waveform.clone().detach().requires_grad_(True)
    mels   = feature_extractor(wav_adv_final)
    logits = lid_model(mels)
    B, T, C = logits.shape
    tgt_labels = torch.full((B, T), target_label, dtype=torch.long, device=device)
    loss = criterion(logits.view(B * T, C), tgt_labels.view(B * T))
    loss.backward()
    with torch.no_grad():
        perturbation = epsilons[-1] * wav_adv_final.grad.sign()
        wav_perturbed = (waveform + perturbation).clamp(-1.0, 1.0)
    snr_final = compute_snr_db(waveform, perturbation)
    print(f"[FGSM] Best effort: ε={epsilons[-1]:.5f} SNR={snr_final:.1f}dB")
    return wav_perturbed.detach(), float(epsilons[-1]), float(snr_final)


# 5.  LID Timestamp Accuracy (within 200 ms)
def evaluate_lid_timestamp_accuracy(
    predicted_segments: List[dict],
    reference_segments: List[dict],
    tolerance_ms: float = 200.0,
) -> float:
    """
    Fraction of predicted language-switch timestamps that land within
    ±200 ms of the reference switch timestamp.

    Returns accuracy ∈ [0, 1].
    """
    def get_switch_times(segs: List[dict]) -> List[float]:
        times = []
        for i in range(1, len(segs)):
            if segs[i]["lang"] != segs[i-1]["lang"]:
                times.append(segs[i]["start_ms"])
        return times

    ref_times  = get_switch_times(reference_segments)
    pred_times = get_switch_times(predicted_segments)

    if not ref_times:
        return 1.0

    correct = 0
    for r in ref_times:
        if any(abs(p - r) <= tolerance_ms for p in pred_times):
            correct += 1

    acc = correct / len(ref_times)
    print(f"[LID-TS] Switch accuracy within {tolerance_ms}ms: "
          f"{correct}/{len(ref_times)} = {acc:.4f}")
    return acc