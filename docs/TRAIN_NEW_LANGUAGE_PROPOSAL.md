# Proposal: Training Qwen3-TTS on a New Language (e.g., Polish)

This document proposes a practical rollout to add high-quality support for a currently unsupported language in this repository.

## Goals

- Add explicit language conditioning for the new language (e.g., `polish`).
- Fine-tune model behavior to produce natural pronunciation and prosody.
- Validate quality with repeatable checks before release.

## Proposed Rollout

## Phase 1 — Data and Token Registration

### 1) Prepare language dataset
Create a JSONL dataset with fields:
- `audio`: target waveform path
- `text`: transcript in target language
- `ref_audio`: reference speaker waveform path

Recommendations:
- 24kHz mono wav.
- Start with clean, studio-like recordings.
- Include diverse punctuation, numerals, abbreviations, and domain text.

### 2) Extract audio codes
Use existing pipeline:

```bash
python finetuning/prepare_data.py \
  --device cuda:0 \
  --tokenizer_model_path Qwen/Qwen3-TTS-Tokenizer-12Hz \
  --input_jsonl train_raw.jsonl \
  --output_jsonl train_with_codes.jsonl
```

### 3) Register language token in model config
If language is not already in `codec_language_id`, register it:

```bash
python scripts/register_language.py \
  --model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --language polish \
  --output_path ./qwen3-tts-polish-base
```

This allocates a language ID and resizes codec embedding if required.

---

## Phase 2 — Fine-tuning Strategy

### 4) Start with `lang_only` mode
This is the fastest/lowest-risk starting point for unsupported languages.

```bash
python scripts/finetune_language.py \
  --init_model_path ./qwen3-tts-polish-base \
  --train_jsonl train_with_codes.jsonl \
  --output_dir output/polish_lang_only \
  --train_mode lang_only \
  --target_language polish \
  --num_epochs 8 \
  --batch_size 2 \
  --learning_rate 1e-4
```

### 5) Escalate to LoRA/full only if needed
- If pronunciation quality is still weak -> try `lora`.
- If style/prosody transfer remains limited with enough data -> evaluate `full` fine-tuning.

---

## Phase 3 — Evaluation and Acceptance

### 6) Build a fixed prompt evaluation set
Create a benchmark set with:
- Diacritics and special letters (for Polish: `ą ć ę ł ń ó ś ź ż`).
- Numerals, dates, and abbreviations.
- Short and long utterances.

### 7) Compare checkpoints by identical prompts
For each checkpoint:
- Generate audio with `language="polish"`.
- Listen for pronunciation errors and unstable rhythm.
- Track objective proxies where available (ASR CER/WER, duration stability).

### 8) Release criteria
Recommend release only when:
- No recurring systematic mispronunciations on benchmark prompts.
- Stable voice identity and prosody across long/short sentences.
- Better quality than base (pre-finetune) baseline on same prompts.

---

## Operational Enhancements (Recommended)

1. Add a one-command orchestrator script to run:
   - data checks -> code extraction -> fine-tuning -> language registration verification.
2. Add language-specific text normalization pre-step.
3. Add CI smoke test that ensures `language="polish"` resolves and generates without fallback errors.

---

## Example Inference After Training

```python
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

model = Qwen3TTSModel.from_pretrained(
    "output/polish_lang_only/checkpoint-epoch-8",
    device_map="cuda:0",
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

wavs, sr = model.generate_voice_clone(
    text="Cześć, jak się masz?",
    language="polish",
    ref_audio="data/ref.wav",
)

sf.write("polish_sample.wav", wavs[0], sr)
```

## Notes

- This proposal is intentionally incremental: get a working Polish model quickly, then improve quality with better data and broader adaptation modes.
- For implementation details of current tooling, see `docs/LANGUAGE_FINETUNING.md`, `scripts/register_language.py`, and `scripts/finetune_language.py`.
