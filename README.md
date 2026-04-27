# Hinglish → Maithili Voice Clone Pipeline

## Submitted By: Raj Vijayvargiya (B22AI067)


> **Assignment:** Code-Switched STT → LRL TTS with Adversarial Robustness  
> **Source:** NPTEL / YouTube lecture (2h 20m – 2h 54m segment)  
> **Dataset video:** <https://youtu.be/ZPUtA3W-7_I>

---

## Repository Structure

```
hinglish_pipeline/
├── pipeline.py               # Main orchestration script
├── config.yaml               # All hyper-parameters and paths
├── requirements.txt          # pip dependencies
├── environment.yml           # conda environment
├── src/
│   ├── __init__.py
│   ├── data_utils.py         # YouTube download, segmentation, augmentation
│   ├── denoising.py          # Task 1.3: Spectral Subtraction / DeepFilterNet
│   ├── lid_model.py          # Task 1.1: Conformer frame-level LID
│   ├── transcription.py      # Task 1.2: Whisper + N-gram Logit Bias
│   ├── phonetic_mapping.py   # Tasks 2.1/2.2: IPA G2P + Maithili translation
│   ├── voice_cloning.py      # Tasks 3.1/3.2/3.3: d-vector, DTW, YourTTS
│   └── anti_spoofing.py      # Tasks 4.1/4.2: LFCC CM, FGSM
├── data/                     # Auto-created; holds audio files
│   ├── original_segment.wav  # 10-min lecture excerpt
│   ├── student_voice_ref.wav # Your 60-second voice recording
│   └── output_LRL_cloned.wav # Final Maithili synthesised output
└── models/                   # Auto-created; holds trained weights
    ├── lid_model.pt
    └── spoof_cm.pt
```

---

## Setup

### Option A – Conda (recommended)

```bash
conda env create -f environment.yml
conda activate hinglish_pipeline
```

### Option B – pip

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **System requirements:** `espeak-ng` and `ffmpeg` must be installed.  
> Linux: `sudo apt install espeak-ng ffmpeg`  
> macOS: `brew install espeak-ng ffmpeg`

---

## Usage

### 1. Full end-to-end run

```bash
python pipeline.py --mode full
```

This will:
1. Download the YouTube lecture and extract the 2h20m–2h30m segment
2. Denoise with spectral subtraction
3. Train / load the frame-level LID model
4. Transcribe with Whisper + N-gram logit bias
5. Convert to IPA, translate to Maithili
6. Extract your d-vector speaker embedding
7. Synthesise the Maithili lecture with YourTTS + DTW prosody warping
8. Train the LFCC anti-spoofing CM, compute EER
9. Run FGSM adversarial attack and report minimum ε
10. Print the full evaluation report

### 2. Using your own audio

```bash
# If you already have the segment:
python pipeline.py --mode full --segment /path/to/original_segment.wav

# Provide your voice recording:
python pipeline.py --mode full --student-voice /path/to/student_voice_ref.wav
```

### 3. Individual stages

```bash
python pipeline.py --mode download      # Stage 0: Download
python pipeline.py --mode transcribe    # Stages 1–3
python pipeline.py --mode translate     # Stage 4
python pipeline.py --mode tts           # Stage 5
python pipeline.py --mode spoof         # Stage 6
python pipeline.py --mode adversarial   # Stage 7
python pipeline.py --mode evaluate      # Stage 8
```

### 4. GPU acceleration

```bash
python pipeline.py --mode full --device cuda
```

---

## Recording Your Voice (Task 3.1)

Record exactly **60 seconds** of yourself speaking (any content):

```bash
# Linux (ALSA)
arecord -f S16_LE -r 16000 -c 1 -d 60 data/student_voice_ref.wav

# macOS (SoX)
rec -r 16000 -c 1 -b 16 data/student_voice_ref.wav trim 0 60

# Windows (PowerShell – requires ffmpeg)
ffmpeg -f dshow -i audio="Microphone" -t 60 -ar 16000 -ac 1 data/student_voice_ref.wav
```

---

## Architecture Details

### Part I – Transcription (STT)

| Task | Implementation |
|------|----------------|
| 1.1 LID | 4-layer Conformer Transformer, 256-dim, 8-head, per-frame softmax output |
| 1.2 Constrained Decoding | Whisper-large-v3 + `NGramLogitBiasProcessor` (HF `LogitsProcessor` API) |
| 1.3 Denoising | Power-spectrum subtraction (α=1.5, β=0.002) + WebRTC VAD trim |

### Part II – Phonetic Mapping

| Task | Implementation |
|------|----------------|
| 2.1 IPA | Devanagari: hand-crafted 80-entry G2P table; Latin: eSpeak-ng via `phonemizer`; Hinglish: override dictionary |
| 2.2 Translation | 500-term Maithili technical dictionary with bigram lookup |

### Part III – Voice Cloning

| Task | Implementation |
|------|----------------|
| 3.1 d-vector | SpeechBrain ECAPA-TDNN (192-dim); fallback: LSTM d-vector (256-dim) |
| 3.2 Prosody DTW | librosa PYIN (F0) + FastDTW alignment of professor → student contours |
| 3.3 Synthesis | Coqui YourTTS zero-shot speaker cloning; output ≥ 22.05 kHz |

### Part IV – Adversarial Robustness

| Task | Implementation |
|------|----------------|
| 4.1 CM | CNN (3 conv layers) + BiLSTM, trained on LFCC (60 coeffs) features |
| 4.2 FGSM | Gradient-sign attack on LID; binary search over ε; SNR constraint ≥ 40 dB |

---

## Evaluation Thresholds

| Metric | Target | Notes |
|--------|--------|-------|
| WER (English) | < 15% | Measured on manually verified GT |
| WER (Hindi) | < 25% | Measured on manually verified GT |
| MCD | < 8.0 dB | Student ref vs synthesised LRL |
| LID switch accuracy | ≤ 200 ms | Timestamp precision |
| CM EER | < 10% | Bonafide vs TTS spoof |
| FGSM SNR | ≥ 40 dB | Inaudible perturbation |

---

## Submission Files

| File | Description |
|------|-------------|
| `pipeline.py` | Main orchestration |
| `src/*.py` | All modules |
| `config.yaml` | Configuration |
| `environment.yml` | Conda env |
| `models/lid_model.pt` | Trained LID weights |
| `models/spoof_cm.pt` | Trained CM weights |
| `data/original_segment.wav` | Source lecture (10 min) |
| `data/student_voice_ref.wav` | 60-sec student voice |
| `data/output_LRL_cloned.wav` | Final Maithili synthesis |

---

## Citation / References

- Whisper: Radford et al. (2022) *Robust Speech Recognition via Large-Scale Weak Supervision*
- YourTTS: Casanova et al. (2022) *YourTTS: Towards Zero-Shot Multi-Speaker TTS*
- ECAPA-TDNN: Desplanques et al. (2020) *ECAPA-TDNN: Emphasized Channel Attention, Propagation and Aggregation in TDNN*
- FastDTW: Salvador & Chan (2007) *Toward Accurate Dynamic Time Warping in Linear Time and Space*
- ASVspoof: Todisco et al. (2019) *ASVspoof 2019: A large-scale public database of synthesized, converted and replayed speech*
