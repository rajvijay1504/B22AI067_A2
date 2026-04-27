"""Task 1.2: Constrained Decoding via Logit Bias + N-gram Language Model
Uses Whisper-large-v3 with a custom LogitProcessor that boosts probabilities
of tokens matching a KenLM / count-based N-gram LM trained on lecture syllabi."""

from __future__ import annotations
import re
import math
from pathlib import Path
from collections import defaultdict, Counter
from typing import List, Dict, Optional, Tuple

import torch
import numpy as np
import torchaudio
import torchaudio.functional as TF


class NgramLM:
    """Smoothed N-gram LM trained on speech-course syllabus text for Whisper logit biasing."""

    SPEECH_CORPUS_SEED = """
    stochastic gradient descent hidden markov model gaussian mixture model
    cepstrum mel frequency cepstral coefficients filterbank feature extraction
    spectrogram short time fourier transform fundamental frequency formant
    phoneme grapheme phonology morphology syntax semantics prosody
    waveform sample rate quantization aliasing nyquist theorem
    encoder decoder attention mechanism transformer neural network
    connectionist temporal classification CTC loss beam search decoding
    language model perplexity word error rate WER character error rate CER
    speaker diarization voice activity detection speaker recognition
    zero shot transfer learning fine tuning pre trained model
    code switching language identification multilingual speech
    sequence to sequence model acoustic model pronunciation dictionary
    forced alignment time stamps Viterbi algorithm forward backward
    recurrent neural network LSTM GRU bidirectional attention
    convolution pooling batch normalization dropout regularization
    adam optimizer learning rate warmup cosine annealing
    end to end automatic speech recognition ASR TTS text to speech
    vocoder waveform synthesis WaveNet WaveGlow HiFi-GAN
    spectral subtraction Wiener filter noise reduction voice enhancement
    speaker embedding d-vector x-vector ECAPA-TDNN speaker verification
    prosody pitch energy duration fundamental frequency F0 contour
    dynamic time warping DTW alignment phonetic transcription IPA
    international phonetic alphabet grapheme to phoneme G2P epitran
    low resource language code mixed Hinglish Maithili Santhali Gondi
    adversarial perturbation FGSM fast gradient sign method epsilon
    spoofing detection anti spoofing countermeasure LFCC CQCC EER
    equal error rate mel cepstral distortion MCD voice cloning VITS YourTTS
    zero shot voice cloning speaker adaptation cross lingual synthesis
    """

    def __init__(self, order: int = 3):
        self.order = order
        self.ngrams: Dict[tuple, Counter] = defaultdict(Counter)
        self.unigrams: Counter = Counter()
        self.vocab: set = set()

    def tokenize(self, text: str) -> List[str]:
        return re.findall(r"[a-zA-Z0-9\-']+", text.lower())

    def train(self, corpus: str):
        tokens = self.tokenize(corpus)
        self.vocab.update(tokens)
        self.unigrams.update(tokens)
        for n in range(2, self.order + 1):
            for i in range(len(tokens) - n + 1):
                ctx  = tuple(tokens[i: i + n - 1])
                word = tokens[i + n - 1]
                self.ngrams[ctx][word] += 1
        print(f"[NgramLM] Trained  order={self.order}  vocab={len(self.vocab)}  "
              f"unigrams={len(self.unigrams)}")

    def train_from_seed(self):
        """Train on the built-in speech-course seed corpus."""
        self.train(self.SPEECH_CORPUS_SEED)

    def score(self, word: str, context: tuple) -> float:
        """Kneser-Ney-style interpolated log-probability."""
        d = 0.75
        word = word.lower()
        ctx  = context[-self.order + 1:] if context else ()

        for n in range(len(ctx), -1, -1):
            sub_ctx = ctx[-n:] if n > 0 else ()
            cnt = self.ngrams.get(sub_ctx, Counter())
            total = sum(cnt.values())
            if total > 0 and word in cnt:
                prob = max(cnt[word] - d, 0) / total
                return math.log(prob + 1e-8)

        total_uni = sum(self.unigrams.values())
        return math.log((self.unigrams.get(word, 0) + 1) / (total_uni + len(self.vocab)))

    @property
    def technical_terms(self) -> List[str]:
        return list(self.vocab)


class NGramLogitBiasProcessor:
    """
    Whisper LogitsProcessor that adds positive bias to token IDs matching
    high-scoring N-gram terms. Compatible with HuggingFace generate().
    """

    def __init__(self, tokenizer, ngram_lm: NgramLM, bias_weight: float = 5.0):
        self.tokenizer   = tokenizer
        self.ngram_lm    = ngram_lm
        self.bias_weight = bias_weight
        self._build_bias_table()

    def _build_bias_table(self):
        """Pre-compute token_id -> bias for every technical term."""
        self.bias_ids: Dict[int, float] = {}
        for term in self.ngram_lm.technical_terms:
            ids = self.tokenizer.encode(" " + term, add_special_tokens=False)
            score = self.ngram_lm.score(term, ())
            normalised = self.bias_weight * (1.0 / (1.0 + math.exp(-score)))
            for tid in ids:
                existing = self.bias_ids.get(tid, 0.0)
                self.bias_ids[tid] = max(existing, normalised)

        print(f"[LogitBias] Bias table  {len(self.bias_ids)} token IDs  "
              f"bias_weight={self.bias_weight}")

    def __call__(self, input_ids: torch.LongTensor,
                 scores: torch.FloatTensor) -> torch.FloatTensor:
        for tid, bias in self.bias_ids.items():
            if tid < scores.shape[-1]:
                scores[:, tid] += bias
        return scores


def load_whisper(model_size: str = "large-v3", device: str = "cpu"):
    """Load Whisper model and processor from HuggingFace."""
    from transformers import WhisperProcessor, WhisperForConditionalGeneration

    model_id = f"openai/whisper-{model_size}"
    print(f"[Whisper] Loading {model_id} ...")
    processor = WhisperProcessor.from_pretrained(model_id)
    model     = WhisperForConditionalGeneration.from_pretrained(model_id)
    model     = model.to(device).eval()
    return processor, model


def transcribe_segment(
    audio_path: str,
    processor,
    model,
    ngram_lm: NgramLM,
    device: str = "cpu",
    chunk_length_s: int = 30,
    beam_size: int = 5,
    logit_bias_weight: float = 5.0,
    language: Optional[str] = None,
) -> List[Dict]:
    """
    Transcribe audio using Whisper with N-gram logit bias.
    Returns list of dicts: [{"text": ..., "start": float, "end": float, "language": ...}]
    """
    from transformers import LogitsProcessorList

    waveform, sr = torchaudio.load(audio_path)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(0, keepdim=True)
    if sr != 16000:
        waveform = TF.resample(waveform, sr, 16000)
        sr = 16000

    wav_np = waveform.squeeze().numpy()
    total_dur = len(wav_np) / sr

    logit_processor = NGramLogitBiasProcessor(
        processor.tokenizer, ngram_lm, bias_weight=logit_bias_weight
    )
    logit_processor_list = LogitsProcessorList([logit_processor])

    results   = []
    chunk_len = chunk_length_s * sr
    n_chunks  = math.ceil(len(wav_np) / chunk_len)

    for i in range(n_chunks):
        start_s = i * chunk_length_s
        end_s   = min((i + 1) * chunk_length_s, total_dur)
        chunk   = wav_np[i * chunk_len: (i + 1) * chunk_len]

        if len(chunk) < chunk_len:
            chunk = np.pad(chunk, (0, chunk_len - len(chunk)))

        inputs = processor(
            chunk, sampling_rate=16000,
            return_tensors="pt", truncation=False, padding="longest",
        ).to(device)

        forced_decoder_ids = None
        if language:
            forced_decoder_ids = processor.get_decoder_prompt_ids(
                language=language, task="transcribe"
            )

        with torch.no_grad():
            gen_ids = model.generate(
                **inputs,
                forced_decoder_ids=forced_decoder_ids,
                num_beams=beam_size,
                logits_processor=logit_processor_list,
                return_timestamps=True,
            )

        chunk_result = processor.batch_decode(gen_ids, return_timestamps=True,
                                              skip_special_tokens=True)

        for item in chunk_result:
            if isinstance(item, dict):
                text  = item.get("text", "").strip()
                ts    = item.get("chunks", [])
            else:
                text  = str(item).strip()
                ts    = []

            if ts:
                for chunk_ts in ts:
                    results.append({
                        "text":     chunk_ts.get("text", "").strip(),
                        "start":    start_s + (chunk_ts.get("timestamp", (0, 0))[0] or 0),
                        "end":      start_s + (chunk_ts.get("timestamp", (0, 0))[1] or chunk_length_s),
                        "language": language or "auto",
                    })
            else:
                results.append({
                    "text":     text,
                    "start":    start_s,
                    "end":      end_s,
                    "language": language or "auto",
                })

        print(f"[Whisper] Chunk {i+1}/{n_chunks}  {start_s:.0f}s-{end_s:.0f}s  -> {len(results)} segs")

    return results


def save_transcript(segments: List[Dict], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for seg in segments:
            f.write(f"[{seg['start']:.2f} -> {seg['end']:.2f}]  "
                    f"[{seg['language']}]  {seg['text']}\n")
    print(f"[Whisper] Transcript saved -> {path}")


def compute_wer(reference: str, hypothesis: str) -> float:
    """Levenshtein-based word error rate."""
    ref = reference.lower().split()
    hyp = hypothesis.lower().split()
    n, m = len(ref), len(hyp)
    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[:]
        dp[0] = i
        for j in range(1, m + 1):
            if ref[i-1] == hyp[j-1]:
                dp[j] = prev[j-1]
            else:
                dp[j] = 1 + min(prev[j], dp[j-1], prev[j-1])
    return dp[m] / max(n, 1)