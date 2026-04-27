"""Task 1.3: Denoising & Normalization
Methods:  (a) Spectral Subtraction   (b) DeepFilterNet wrapper
Also handles VAD-based silence removal and loudness normalisation. """

from __future__ import annotations
import numpy as np
import torch
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
from pathlib import Path
from typing import Tuple



# 1.  Spectral Subtraction
def spectral_subtraction(
    waveform: np.ndarray,           # (T,)  mono, float32
    sample_rate: int = 16000,
    n_fft: int = 512,
    hop_length: int = 128,
    noise_frames: int = 20,         # first N frames used to estimate noise PSD
    over_subtraction: float = 1.5,  # α  –  controls aggressiveness
    spectral_floor: float = 0.002,  # β  –  prevents negative spectral values
) -> np.ndarray:
    """
    Classic power-spectrum subtraction (Boll 1979).
    Returns denoised waveform as float32 numpy array, same length as input.
    """
    # STFT
    stft = np.array([
        np.fft.rfft(waveform[i * hop_length: i * hop_length + n_fft] *
                    np.hanning(n_fft), n_fft)
        for i in range((len(waveform) - n_fft) // hop_length + 1)
    ])                                        # (n_frames, n_fft//2+1)

    power = np.abs(stft) ** 2
    phase = np.angle(stft)

    # Noise PSD estimate from first `noise_frames` frames
    noise_psd = np.mean(power[:noise_frames], axis=0, keepdims=True)

    # Subtraction  P_signal = max(P_noisy - α*P_noise,  β*P_noisy)
    cleaned_power = np.maximum(
        power - over_subtraction * noise_psd,
        spectral_floor * power,
    )
    cleaned_magnitude = np.sqrt(cleaned_power)

    # Reconstruct with original phase
    cleaned_stft = cleaned_magnitude * np.exp(1j * phase)

    # Overlap-add (ISTFT)
    output = np.zeros(len(waveform), dtype=np.float32)
    window = np.hanning(n_fft)
    norm   = np.zeros(len(waveform), dtype=np.float32)

    for i, frame in enumerate(cleaned_stft):
        sig = np.real(np.fft.irfft(frame, n_fft)) * window
        start = i * hop_length
        end   = start + n_fft
        if end > len(output):
            sig = sig[: len(output) - start]
            output[start:] += sig
            norm[start:]   += window[:len(sig)]
        else:
            output[start:end] += sig
            norm[start:end]   += window

    norm = np.where(norm < 1e-8, 1.0, norm)
    output /= norm
    return output.astype(np.float32)



# 2.  DeepFilterNet wrapper  (optional – requires deepfilternet package)
def denoise_deepfilter(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """
    Wraps DeepFilterNet (pip install deepfilternet).
    Falls back to spectral subtraction if package not installed.
    """
    try:
        from df.enhance import enhance, init_df, load_audio, save_audio
        from df.io import resample

        model, df_state, _ = init_df()
        # DeepFilterNet expects 48kHz
        wav_48k = F.resample(waveform, sample_rate, 48000)
        enhanced_48k = enhance(model, df_state, wav_48k)
        enhanced = F.resample(enhanced_48k, 48000, sample_rate)
        return enhanced

    except ImportError:
        print("[Denoising] DeepFilterNet not installed – falling back to spectral subtraction")
        wav_np = waveform.squeeze().numpy()
        out_np = spectral_subtraction(wav_np, sample_rate)
        return torch.from_numpy(out_np).unsqueeze(0)



# 3.  WebRTC VAD for silence removal
def vad_trim(waveform: np.ndarray, sample_rate: int = 16000,
             aggressiveness: int = 2, frame_ms: int = 30) -> np.ndarray:
    """
    Remove leading/trailing silence using WebRTC VAD.
    Falls back to librosa trim if webrtcvad not available.
    """
    try:
        import webrtcvad
        vad = webrtcvad.Vad(aggressiveness)
        frame_len = int(sample_rate * frame_ms / 1000)
        pcm = (waveform * 32768).astype(np.int16)

        speech_frames = []
        for i in range(0, len(pcm) - frame_len, frame_len):
            frame = pcm[i: i + frame_len].tobytes()
            if vad.is_speech(frame, sample_rate):
                speech_frames.append(waveform[i: i + frame_len])

        if speech_frames:
            return np.concatenate(speech_frames)
        return waveform

    except ImportError:
        import librosa
        trimmed, _ = librosa.effects.trim(waveform, top_db=25)
        return trimmed



# 4.  Loudness normalisation  (EBU R128-style using RMS)
def rms_normalize(waveform: np.ndarray, target_db: float = -23.0) -> np.ndarray:
    rms = np.sqrt(np.mean(waveform ** 2) + 1e-9)
    target_rms = 10 ** (target_db / 20)
    return (waveform * (target_rms / rms)).astype(np.float32)



# 5.  High-level preprocessing pipeline
def preprocess_audio(
    input_path: str,
    output_path: str,
    method: str = "spectral_subtraction",  # or "deepfilter"
    target_sr: int = 16000,
    noise_frames: int = 20,
    over_subtraction: float = 1.5,
    spectral_floor: float = 0.002,
    normalize: bool = True,
    do_vad: bool = True,
) -> Tuple[torch.Tensor, int]:
    """
    Full preprocessing:
      1. Load & resample
      2. Stereo → mono
      3. Denoise  (spectral subtraction or DeepFilterNet)
      4. VAD trim
      5. RMS normalise
      6. Save

    Returns (waveform_tensor, sample_rate).
    """
    waveform, sr = torchaudio.load(input_path)

    # Resample
    if sr != target_sr:
        waveform = F.resample(waveform, sr, target_sr)
        sr = target_sr

    # Mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(0, keepdim=True)

    wav_np = waveform.squeeze().numpy()

    #  Denoise 
    if method == "deepfilter":
        waveform_denoised = denoise_deepfilter(waveform, sr).squeeze().numpy()
    else:
        print(f"[Denoising] Applying spectral subtraction  (α={over_subtraction}, β={spectral_floor})")
        waveform_denoised = spectral_subtraction(
            wav_np, sr,
            noise_frames=noise_frames,
            over_subtraction=over_subtraction,
            spectral_floor=spectral_floor,
        )

    #  VAD trim 
    if do_vad:
        print("[Denoising] Running VAD trim …")
        waveform_denoised = vad_trim(waveform_denoised, sr)

    #  Normalise 
    if normalize:
        waveform_denoised = rms_normalize(waveform_denoised)

    #  Save 
    out_tensor = torch.from_numpy(waveform_denoised).unsqueeze(0)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    torchaudio.save(output_path, out_tensor, sr)
    print(f"[Denoising] Saved denoised audio → {output_path}  "
          f"({waveform_denoised.shape[0]/sr:.1f} s)")
    return out_tensor, sr



# 6.  Compute SNR between original and processed
def compute_snr(clean: np.ndarray, noisy: np.ndarray) -> float:
    """Signal-to-Noise Ratio in dB between two equal-length signals."""
    min_len = min(len(clean), len(noisy))
    c, n = clean[:min_len], noisy[:min_len]
    noise = n - c
    snr = 10 * np.log10((np.mean(c ** 2) + 1e-9) / (np.mean(noise ** 2) + 1e-9))
    return float(snr)