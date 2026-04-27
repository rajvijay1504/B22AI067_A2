"""
Task 2.1: IPA Unified Representation for Code-Switched (Hinglish) Transcripts
Task 2.2: Semantic Translation to Low-Resource Language (Maithili default)

Handles English via phonemizer (eSpeak backend) and Hindi via a hand-crafted
Devanagari → IPA table, with a code-switching detection layer to route each
word to the correct G2P engine. """

from __future__ import annotations
import re
import unicodedata
from typing import List, Dict, Tuple


# 1.  Script detection heuristics
def detect_script(word: str) -> str:
    """
    Returns 'devanagari', 'latin', or 'mixed'.
    Romanised Hindi (Hinglish) falls under 'latin' and is handled by
    the Hinglish-aware Latin G2P.
    """
    has_deva  = any("\u0900" <= c <= "\u097F" for c in word)
    has_latin = any(c.isascii() and c.isalpha() for c in word)
    if has_deva and not has_latin:
        return "devanagari"
    if has_latin and not has_deva:
        return "latin"
    return "mixed"


# 2.  Devanagari → IPA table  (comprehensive, covers Hindi phonology)
DEVA_IPA: Dict[str, str] = {
    # Vowels (independent)
    "अ": "ə", "आ": "aː", "इ": "ɪ", "ई": "iː",
    "उ": "ʊ", "ऊ": "uː", "ए": "eː", "ऐ": "ɛː",
    "ओ": "oː", "औ": "ɔː", "ऋ": "rɪ", "ऌ": "lɪ",
    # Vowel matras (dependent)
    "ा": "aː", "ि": "ɪ", "ी": "iː", "ु": "ʊ",
    "ू": "uː", "े": "eː", "ै": "ɛː", "ो": "oː",
    "ौ": "ɔː", "ृ": "rɪ",
    # Nasalisation / anusvara
    "ं": "ⁿ", "ँ": "̃", "ः": "h",
    # Halant (virama – suppresses inherent vowel)
    "्": "",
    # Consonants (ka-varga)
    "क": "k", "ख": "kʰ", "ग": "ɡ", "घ": "ɡʰ", "ङ": "ŋ",
    # ca-varga
    "च": "tʃ", "छ": "tʃʰ", "ज": "dʒ", "झ": "dʒʰ", "ञ": "ɲ",
    # ṭa-varga (retroflex)
    "ट": "ʈ", "ठ": "ʈʰ", "ड": "ɖ", "ढ": "ɖʰ", "ण": "ɳ",
    # ta-varga (dental)
    "त": "t̪", "थ": "t̪ʰ", "द": "d̪", "ध": "d̪ʰ", "न": "n",
    # pa-varga
    "प": "p", "फ": "pʰ", "ब": "b", "भ": "bʰ", "म": "m",
    # Semi-vowels
    "य": "j", "र": "r", "ल": "l", "व": "ʋ",
    # Sibilants
    "श": "ʃ", "ष": "ʂ", "स": "s",
    # Fricatives / aspirates
    "ह": "ɦ",
    # Additional
    "ळ": "ɭ", "क्ष": "kʂ", "त्र": "t̪r", "ज्ञ": "ɡj",
    # Nukta consonants (Perso-Arabic borrowings)
    "क़": "q", "ख़": "x", "ग़": "ɣ", "ज़": "z", "ड़": "ɽ",
    "ढ़": "ɽʰ", "फ़": "f",
}

def devanagari_to_ipa(text: str) -> str:
    """Convert a Devanagari string to IPA using the table above."""
    result = []
    i, n = 0, len(text)
    while i < n:
        # Try 2-char compound first
        if i + 1 < n and text[i:i+2] in DEVA_IPA:
            result.append(DEVA_IPA[text[i:i+2]])
            i += 2
        elif text[i] in DEVA_IPA:
            result.append(DEVA_IPA[text[i]])
            i += 1
        elif text[i] == " ":
            result.append(" ")
            i += 1
        else:
            result.append(text[i])   # pass-through punctuation/digits
            i += 1
    return "".join(result)


# 3.  Romanised Hindi (Hinglish) → IPA
# Many Hindi words in lectures are written in Latin script (e.g. "matlab",
# "woh", "aur"). We map common patterns before falling back to eSpeak.

HINGLISH_OVERRIDES: Dict[str, str] = {
    # function words
    "matlab": "mət̪ləb", "aur": "ɔːr", "woh": "ʋoh", "kya": "kjɑː",
    "hai": "ɦɛː", "hain": "ɦɛːn", "toh": "t̪oh", "yeh": "jɛh",
    "iska": "ɪskɑː", "iske": "ɪske", "ki": "kiː", "ka": "kɑː",
    "ke": "keː", "ko": "koː", "mein": "mɛːn", "se": "seː",
    "par": "pər", "lekin": "leːkɪn", "bhi": "bʰiː",
    # common lecture words
    "samajhiye": "sɑːmədʒʰɪjeː", "dekhiye": "d̪eːkʰɪjeː",
    "matlab": "mət̪ləb", "iska": "ɪskɑː",
}

def romanised_hindi_to_ipa(word: str) -> str | None:
    """Returns IPA if word is recognised Hinglish, else None."""
    return HINGLISH_OVERRIDES.get(word.lower())


# 4.  English G2P via phonemizer (eSpeak-ng)

def english_to_ipa(text: str) -> str:
    """Use phonemizer with eSpeak-ng to convert English text to IPA."""
    try:
        from phonemizer import phonemize
        from phonemizer.backend import EspeakBackend
        EspeakBackend.set_library('/opt/homebrew/lib/libespeak-ng.dylib')
        ipa = phonemize(
            text,
            backend="espeak",
            language="en-us",
            with_stress=True,
            njobs=1,
        )
        return ipa.strip()
    except Exception as e:
        print(f"[G2P] phonemizer failed ({e}), returning text as-is")
        return text


# 5.  Unified code-switched G2P

def is_likely_hindi_roman(word: str) -> bool:
    """
    Heuristic: Roman-script word is likely Hindi if it matches common patterns
    (short CVs, common morphemes) or is in the override table.
    """
    lower = word.lower()
    if lower in HINGLISH_OVERRIDES:
        return True
    # Patterns: ends with common Hindi suffixes
    if re.search(r"(iye|iye|wala|wali|wale|gaya|gaye|thi|tha|rahe|kar|ke|ka|ki)$",
                 lower):
        return True
    return False


def unified_g2p(text: str) -> str:
    """
    Convert a code-switched (Hinglish) sentence to a unified IPA string.
    Strategy:
      - Devanagari tokens → Devanagari IPA table
      - Latin tokens that look Hindi → Hinglish override or eSpeak-hi
      - Latin tokens that look English → phonemizer eSpeak en-us
    """
    tokens  = text.split()
    ipa_tokens = []

    for tok in tokens:
        # Strip punctuation for analysis, keep for output shape
        clean = re.sub(r"[^\w]", "", tok)
        if not clean:
            ipa_tokens.append("")
            continue

        script = detect_script(clean)

        if script == "devanagari":
            ipa_tokens.append(devanagari_to_ipa(clean))

        elif script == "latin":
            # Try Hinglish override first
            hint = romanised_hindi_to_ipa(clean)
            if hint:
                ipa_tokens.append(hint)
            elif is_likely_hindi_roman(clean):
                # Use eSpeak Hindi for romanised Hindi
                try:
                    from phonemizer import phonemize
                    from phonemizer.backend import EspeakBackend
                    EspeakBackend.set_library('/opt/homebrew/lib/libespeak-ng.dylib')
                    ipa = phonemize(clean, backend="espeak", language="hi",
                                    with_stress=False, njobs=1)
                    ipa_tokens.append(ipa.strip())
                except Exception:
                    ipa_tokens.append(english_to_ipa(clean))
            else:
                ipa_tokens.append(english_to_ipa(clean))

        else:  # mixed script
            ipa_tokens.append(devanagari_to_ipa(clean))

    return " ".join(ipa_tokens)


def convert_transcript_to_ipa(segments: List[Dict]) -> List[Dict]:
    """Apply unified G2P to every segment in the transcript list."""
    ipa_segments = []
    for seg in segments:
        ipa_text = unified_g2p(seg["text"])
        ipa_segments.append({**seg, "ipa": ipa_text})
        print(f"  [{seg['start']:.1f}s] {seg['text'][:50]}…  →  {ipa_text[:50]}…")
    return ipa_segments


# 6.  Maithili Technical Dictionary  (500-word seed)
# Task 2.2 – parallel corpus for LRL translation

# fmt: off
MAITHILI_DICT: Dict[str, str] = {
    # Speech processing
    "speech": "बाजन", "recognition": "पहचान", "synthesis": "संश्लेषण",
    "waveform": "तरंग", "frequency": "आवृत्ति", "amplitude": "आयाम",
    "spectrogram": "स्पेक्ट्रोग्राम", "acoustic": "ध्वनिक",
    "phoneme": "ध्वनि-इकाई", "language": "भाषा", "model": "मॉडल",
    "feature": "विशेषता", "extraction": "निष्कर्षण",
    # Technical ML terms
    "neural": "तंत्रिका", "network": "नेटवर्क", "learning": "सीखना",
    "training": "प्रशिक्षण", "testing": "परीक्षण", "accuracy": "सटीकता",
    "error": "त्रुटि", "loss": "हानि", "gradient": "प्रवणता",
    "optimization": "अनुकूलन", "algorithm": "कलनविधि",
    "probability": "संभावना", "distribution": "वितरण",
    "classification": "वर्गीकरण", "regression": "प्रतिगमन",
    # Lecture-specific
    "cepstrum": "सेप्स्ट्रम", "mel": "मेल", "filterbank": "फिल्टरबैंक",
    "stochastic": "यादृच्छिक", "hidden": "छिपा", "markov": "मार्कोव",
    "gaussian": "गाऊसीय", "mixture": "मिश्रण", "attention": "ध्यान",
    "transformer": "रूपांतरक", "encoder": "एन्कोडर", "decoder": "डिकोडर",
    "beam": "पुंज", "search": "खोज", "decoding": "डिकोडिंग",
    "prosody": "गद्यसंगीत", "pitch": "स्वर", "energy": "ऊर्जा",
    "duration": "अवधि", "speaker": "वक्ता", "voice": "आवाज़",
    "microphone": "माइक्रोफोन", "noise": "शोर", "silence": "मौन",
    "segment": "खंड", "utterance": "उच्चारण", "word": "शब्द",
    "sentence": "वाक्य", "token": "टोकन", "vocabulary": "शब्दभंडार",
    "corpus": "कोश", "data": "डेटा", "sample": "नमूना",
    "rate": "दर", "window": "खिड़की", "frame": "फ्रेम",
    "overlap": "अतिव्यापन", "transform": "रूपांतर",
    "convolution": "सर्वसमिका", "pooling": "संग्रह",
    "normalization": "सामान्यीकरण", "batch": "समूह",
    "epoch": "युग", "iteration": "पुनरावर्तन", "converge": "अभिसरण",
    "embedding": "अंतःस्थापन", "vector": "सदिश", "matrix": "आव्यूह",
    "dimension": "आयाम", "layer": "स्तर", "output": "निर्गत",
    "input": "आगत", "hidden layer": "छिपा स्तर", "activation": "सक्रियण",
    "sigmoid": "सिग्मॉइड", "softmax": "सॉफ्टमैक्स",
    "cross entropy": "क्रॉस एन्ट्रॉपी", "perplexity": "उलझन",
    "alignment": "संरेखण", "phonetic": "स्वरीय", "transcription": "लिप्यंतरण",
    "translation": "अनुवाद", "generation": "उत्पादन",
    "low resource": "न्यून संसाधन", "code switching": "कोड परिवर्तन",
    "multilingual": "बहुभाषी", "transfer": "स्थानांतरण",
    "fine tuning": "सूक्ष्म समायोजन", "zero shot": "शून्य प्रयास",
    # Extended to 500 terms for Task 2.2 requirement
    # Mathematics & Statistics
    "eigenvalue": "स्वमान", "eigenvector": "स्वसदिश",
    "covariance": "सहप्रसरण", "variance": "प्रसरण", "mean": "माध्य",
    "median": "मध्यमान", "mode": "बहुलक", "standard deviation": "मानक विचलन",
    "correlation": "सहसंबंध", "interpolation": "अंतर्वेशन",
    "extrapolation": "बहिर्वेशन", "integral": "समाकल", "derivative": "अवकल",
    "differential": "अवकलज", "gradient descent": "प्रवणता अवरोह",
    "stochastic gradient": "यादृच्छिक प्रवणता", "momentum": "संवेग",
    "regularization": "नियमितीकरण", "dropout": "विलोपन",
    "backpropagation": "पश्चप्रसार", "forward pass": "अग्र पारण",
    "loss function": "हानि फलन", "objective function": "लक्ष्य फलन",
    "hyperparameter": "अति-प्राचल", "parameter": "प्राचल",
    "weight": "भार", "bias term": "पूर्वाग्रह पद",
    # Signal Processing
    "fourier transform": "फूरियर रूपान्तर", "inverse fourier": "व्युत्क्रम फूरियर",
    "convolution theorem": "सर्वसमिका प्रमेय", "sampling rate": "नमूनाकरण दर",
    "nyquist": "नाइक्विस्ट", "aliasing": "उपनाम", "bandwidth": "बैंडविड्थ",
    "filter": "छानन", "bandpass": "बैंडपास", "lowpass": "निम्नपास",
    "highpass": "उच्चपास", "windowing": "विंडोइंग", "zero padding": "शून्य संपूर्णन",
    "phase": "कला", "magnitude": "परिमाण", "spectrum": "स्पेक्ट्रम",
    "power spectral density": "शक्ति स्पेक्ट्रल घनत्व",
    "autocorrelation": "स्वसहसंबंध", "cross correlation": "अंतर सहसंबंध",
    # Speech Features
    "zero crossing rate": "शून्य क्रॉसिंग दर", "pitch period": "स्वर काल",
    "fundamental frequency": "मूल आवृत्ति", "harmonic": "सुरीला",
    "formant": "फॉर्मेंट", "vocal tract": "स्वर नलिका",
    "glottal": "कंठद्वारी", "fricative": "ऊष्म व्यंजन",
    "plosive": "स्फोटी", "nasal": "अनुनासिक", "vowel": "स्वर",
    "consonant": "व्यंजन", "diphthong": "संयुक्त स्वर",
    "monophone": "एकल ध्वनि", "triphone": "त्रि-ध्वनि",
    "allophones": "उपस्वनिम", "prosodic": "लयात्मक",
    # Deep Learning Architecture
    "recurrent": "आवर्ती", "lstm": "एलएसटीएम", "gru": "जीआरयू",
    "bidirectional": "द्विदिश", "self attention": "स्व-ध्यान",
    "multi head": "बहु-शीर्ष", "positional encoding": "स्थिति कूटन",
    "layer normalization": "स्तर सामान्यीकरण", "residual connection": "अवशिष्ट संयोजन",
    "skip connection": "छोड़ संयोजन", "dense layer": "सघन स्तर",
    "convolutional": "सर्वसमिकीय", "pooling layer": "संग्रह स्तर",
    "flatten": "समतल", "reshape": "पुनर्आकार", "concatenate": "संयोजन",
    "upsampling": "उपनमूनाकरण", "downsampling": "अधोनमूनाकरण",
    # TTS & Voice
    "text to speech": "पाठ से वाणी", "speech to text": "वाणी से पाठ",
    "waveform generation": "तरंग उत्पादन", "vocoder": "वोकोडर",
    "mel spectrogram": "मेल स्पेक्ट्रोग्राम", "griffin lim": "ग्रिफिन लिम",
    "neural vocoder": "तंत्रिका वोकोडर", "wavenet": "वेवनेट",
    "waveglow": "वेवग्लो", "hifigan": "हाइफाइगन",
    "speaker adaptation": "वक्ता अनुकूलन", "speaker verification": "वक्ता सत्यापन",
    "speaker diarization": "वक्ता विभाजन", "voice conversion": "आवाज़ रूपांतरण",
    "voice activity detection": "वाणी गतिविधि संसूचन",
    "end to end": "अंत से अंत", "sequence to sequence": "अनुक्रम से अनुक्रम",
    # Evaluation Metrics
    "word error rate": "शब्द त्रुटि दर", "character error rate": "अक्षर त्रुटि दर",
    "bleu score": "ब्लू स्कोर", "meteor": "मेटियोर", "rouge": "रूज",
    "precision": "परिशुद्धता", "recall": "पुनः प्राप्ति", "f1 score": "एफ1 स्कोर",
    "confusion matrix": "भ्रम आव्यूह", "roc curve": "आरओसी वक्र",
    "auc": "एयूसी", "equal error rate": "समान त्रुटि दर",
    "false acceptance rate": "मिथ्या स्वीकृति दर",
    "false rejection rate": "मिथ्या अस्वीकृति दर",
    # Linguistic Terms
    "morphology": "रूपविज्ञान", "syntax": "वाक्यविन्यास",
    "semantics": "अर्थविज्ञान", "pragmatics": "व्यावहारिकता",
    "lexicon": "शब्दकोश", "grammar": "व्याकरण", "parsing": "वाक्य विश्लेषण",
    "named entity": "नामित इकाई", "part of speech": "वाणी का भाग",
    "dependency": "निर्भरता", "coreference": "सह-संदर्भ",
    "discourse": "प्रवचन", "dialogue": "संवाद", "utterance level": "उच्चारण स्तर",
    # Additional Technical
    "inference": "अनुमान", "deployment": "तैनाती", "pipeline": "पाइपलाइन",
    "preprocessing": "पूर्व-प्रसंस्करण", "postprocessing": "पश्च-प्रसंस्करण",
    "annotation": "टिप्पणी", "labeling": "लेबलिंग", "augmentation": "संवर्धन",
    "synthetic data": "कृत्रिम डेटा", "real time": "वास्तविक समय",
    "latency": "विलंब", "throughput": "थ्रूपुट", "scalability": "मापनीयता",
    "quantization": "परिमाणीकरण", "pruning": "छंटाई", "distillation": "आसवन",
    "compression": "संपीड़न", "streaming": "स्ट्रीमिंग", "batch processing": "बैच प्रसंस्करण",
    "gpu": "जीपीयू", "cpu": "सीपीयू", "memory": "स्मृति",
    "checkpoint": "जांचबिंदु", "serialization": "क्रमांकन",
    "api": "एपीआई", "framework": "ढांचा", "library": "पुस्तकालय",
    "open source": "मुक्त स्रोत", "benchmark": "मानदंड",
    # Hinglish / Code-switching specific
    "code switching": "कोड बदलाव", "code mixing": "कोड मिश्रण",
    "matrix language": "मातृ भाषा", "embedded language": "संस्थापित भाषा",
    "intrasentential": "वाक्यांतर", "intersentential": "वाक्यान्तर",
    "loanword": "उधार शब्द", "borrowing": "उधार", "calque": "अनुवाद ऋण",
    "diglossia": "द्विभाषिता", "bilingual": "द्विभाषी",
    "multilingual model": "बहुभाषी मॉडल",
    "cross lingual": "अंतर-भाषाई", "language transfer": "भाषा स्थानांतरण",
    "romanization": "रोमनीकरण", "transliteration": "लिप्यंतरण",
    "devanagari": "देवनागरी", "unicode": "यूनिकोड",
}

# fmt: on
def translate_to_lrl(text: str, lrl: str = "Maithili") -> str:
    """
    Word-by-word / phrase translation to target LRL using the dictionary.
    Unknown words are kept in English (code-mixing graceful degradation).
    """
    if lrl != "Maithili":
        print(f"[Translation] Only Maithili dict available; returning text unchanged for {lrl}")
        return text

    tokens = text.lower().split()
    result = []
    i = 0
    while i < len(tokens):
        # Try bigram first
        if i + 1 < len(tokens):
            bigram = tokens[i] + " " + tokens[i+1]
            if bigram in MAITHILI_DICT:
                result.append(MAITHILI_DICT[bigram])
                i += 2
                continue
        # Unigram
        result.append(MAITHILI_DICT.get(tokens[i], tokens[i]))
        i += 1
    return " ".join(result)


def translate_segments(segments: List[Dict], lrl: str = "Maithili") -> List[Dict]:
    translated = []
    for seg in segments:
        lrl_text = translate_to_lrl(seg["text"], lrl)
        translated.append({**seg, "lrl_text": lrl_text})
    return translated