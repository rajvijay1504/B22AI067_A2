"""Utilities:
  - Download lecture from YouTube (yt-dlp)
  - Extract specific time-window segment
  - Build N-gram corpus from subtitle/transcript files
  - Audio augmentation for LID training data """

from __future__ import annotations
import os
import subprocess
import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torchaudio
import torchaudio.functional as TF


def download_youtube_audio(
    url: str,
    output_path: str,
    start_sec: Optional[int] = None,
    duration_sec: Optional[int] = None,
    sample_rate: int = 22050,
) -> str:
    """Download audio from YouTube using yt-dlp. Trims with ffmpeg if time bounds given."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    tmp_path = str(Path(output_path).with_suffix(".tmp.%(ext)s"))

    cmd = [
        "yt-dlp",
        "--extract-audio",
        "--audio-format", "wav",
        "--audio-quality", "0",
        "--output", tmp_path,
        "--no-playlist",
        url,
    ]
    print(f"[Data] Downloading audio: {url}")
    subprocess.run(cmd, check=True)

    tmp_wav = str(Path(output_path).with_suffix(".tmp.wav"))
    if not os.path.exists(tmp_wav):
        parent = Path(output_path).parent
        candidates = list(parent.glob("*.tmp.*"))
        if candidates:
            tmp_wav = str(candidates[0])
        else:
            raise FileNotFoundError(f"Downloaded file not found near {output_path}")

    if start_sec is not None and duration_sec is not None:
        print(f"[Data] Trimming  start={start_sec}s  duration={duration_sec}s ...")
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-ss", str(start_sec),
            "-i", tmp_wav,
            "-t", str(duration_sec),
            "-ar", str(sample_rate),
            "-ac", "1",
            "-f", "wav",
            output_path,
        ]
        subprocess.run(ffmpeg_cmd, check=True)
        os.remove(tmp_wav)
    else:
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-i", tmp_wav,
            "-ar", str(sample_rate),
            "-ac", "1",
            output_path,
        ]
        subprocess.run(ffmpeg_cmd, check=True)
        os.remove(tmp_wav)

    print(f"[Data] Saved lecture segment -> {output_path}")
    return output_path


def extract_segment(
    source_path: str,
    output_path: str,
    start_sec: float,
    duration_sec: float,
    target_sr: int = 22050,
) -> Tuple[torch.Tensor, int]:
    """Extract a time window from a WAV file without re-downloading."""
    waveform, sr = torchaudio.load(source_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(0, keepdim=True)
    if sr != target_sr:
        waveform = TF.resample(waveform, sr, target_sr)
        sr = target_sr

    start_sample = int(start_sec * sr)
    end_sample   = int((start_sec + duration_sec) * sr)
    segment = waveform[:, start_sample: end_sample]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(output_path, segment, sr)
    print(f"[Data] Segment saved -> {output_path}  "
          f"({start_sec:.0f}s - {start_sec+duration_sec:.0f}s, {segment.shape[1]/sr:.1f}s)")
    return segment, sr


def download_youtube_subtitles(url: str, output_dir: str, lang: str = "en") -> str:
    """Download auto-generated VTT subtitles from YouTube using yt-dlp."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cmd = [
        "yt-dlp",
        "--skip-download",
        "--write-auto-sub",
        "--sub-lang", lang,
        "--sub-format", "vtt",
        "--output", str(Path(output_dir) / "subtitles.%(ext)s"),
        url,
    ]
    subprocess.run(cmd, check=True)
    candidates = list(Path(output_dir).glob("*.vtt"))
    if candidates:
        return str(candidates[0])
    raise FileNotFoundError("Subtitles not found")


def vtt_to_text(vtt_path: str) -> str:
    """Parse a WebVTT file and return plain text."""
    import re
    text_lines = []
    with open(vtt_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("WEBVTT") or "-->" in line:
                continue
            line = re.sub(r"<[^>]+>", "", line)
            if line:
                text_lines.append(line)
    return " ".join(text_lines)


def build_ngram_corpus(url: str, output_corpus_path: str) -> str:
    """Download subtitles and save as plain text corpus for N-gram LM training."""
    try:
        vtt = download_youtube_subtitles(url, "data/subs")
        corpus = vtt_to_text(vtt)
    except Exception as e:
        print(f"[Corpus] Subtitle download failed ({e}); using seed corpus only")
        corpus = ""

    from src.transcription import NgramLM
    seed = NgramLM.SPEECH_CORPUS_SEED
    full_corpus = seed + "\n" + corpus

    Path(output_corpus_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_corpus_path, "w", encoding="utf-8") as f:
        f.write(full_corpus)
    print(f"[Corpus] Saved N-gram corpus ({len(full_corpus.split())} tokens) -> {output_corpus_path}")
    return full_corpus


def add_room_reverb(waveform: np.ndarray, sample_rate: int = 16000,
                    rt60: float = 0.3) -> np.ndarray:
    """Simulate room reverb using an exponential decay impulse response."""
    decay_samples = int(rt60 * sample_rate)
    ir = np.exp(-3.0 * np.arange(decay_samples) / decay_samples).astype(np.float32)
    reverbed = np.convolve(waveform, ir)[:len(waveform)]
    reverbed = reverbed / (np.max(np.abs(reverbed)) + 1e-8)
    return reverbed.astype(np.float32)


def add_gaussian_noise(waveform: np.ndarray, snr_db: float = 20.0) -> np.ndarray:
    sig_power   = np.mean(waveform ** 2)
    noise_power = sig_power / (10 ** (snr_db / 10))
    noise = np.random.normal(0, math.sqrt(noise_power), waveform.shape).astype(np.float32)
    return (waveform + noise).astype(np.float32)


def augment_audio(waveform: np.ndarray, sample_rate: int,
                  apply_reverb: bool = True, snr_db: Optional[float] = 20.0) -> np.ndarray:
    """Chain noise and reverb augmentations for training robustness."""
    aug = waveform.copy()
    if snr_db is not None:
        aug = add_gaussian_noise(aug, snr_db)
    if apply_reverb:
        aug = add_room_reverb(aug, sample_rate)
    return aug


def save_maithili_parallel_corpus(path: str = "data/maithili_parallel.txt"):
    """Write the Maithili technical dictionary to a parallel corpus file for MT fine-tuning."""
    from src.phonetic_mapping import MAITHILI_DICT

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Maithili Technical Parallel Corpus (500 terms)\n")
        f.write("# Format: English TAB Maithili\n\n")
        for en, mai in MAITHILI_DICT.items():
            f.write(f"{en}\t{mai}\n")
    print(f"[Corpus] Maithili parallel corpus saved -> {path}  "
          f"({len(MAITHILI_DICT)} entries)")