# Language Fine-Tuning Guide for Qwen3-TTS

This guide explains how to fine-tune Qwen3-TTS models for new language support using the `scripts/finetune_language.py` script.

## Overview

The language fine-tuning script provides three training modes optimized for different scenarios:

1. **`lang_only`**: Most efficient - only updates language embedding row(s) and minimal adapters (~1-2% of parameters)
2. **`lora`**: Efficient - uses LoRA (Low-Rank Adaptation) for parameter-efficient fine-tuning (~5-10% of parameters)
3. **`full`**: Complete - full model fine-tuning (all parameters, highest quality but most resource-intensive)

## Prerequisites

### Required Dependencies

```bash
pip install qwen-tts
# For LoRA support (optional)
pip install peft
```

### Data Preparation

Your training data must be in JSONL format with audio codes pre-computed. Use the existing `finetuning/prepare_data.py` script:

```bash
cd finetuning
python prepare_data.py \
  --device cuda:0 \
  --tokenizer_model_path Qwen/Qwen3-TTS-Tokenizer-12Hz \
  --input_jsonl train_raw.jsonl \
  --output_jsonl train_with_codes.jsonl
```

#### Raw JSONL Format

Each line should contain:
- `audio`: path to target training audio (wav)
- `text`: transcript corresponding to audio
- `ref_audio`: path to reference speaker audio (wav)
- `language`: (optional) language identifier (e.g., "polish", "french")

Example:
```jsonl
{"audio":"./data/polish/utt0001.wav","text":"Dzień dobry, jak się masz?","ref_audio":"./data/polish/ref.wav","language":"polish"}
{"audio":"./data/polish/utt0002.wav","text":"To jest przykładowy tekst.","ref_audio":"./data/polish/ref.wav","language":"polish"}
```

## Quick Start

### 1. Language Embedding Only (Recommended for New Languages)

Best for: Adding a completely new language with limited data (10-100 samples)

```bash
python scripts/finetune_language.py \
  --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --train_jsonl data/polish_train_with_codes.jsonl \
  --output_dir output/polish_lang_only \
  --train_mode lang_only \
  --target_language polish \
  --num_epochs 10 \
  --batch_size 4 \
  --learning_rate 1e-4
```

**Key parameters:**
- `--train_mode lang_only`: Only update language embeddings and top 2 layers
- `--learning_rate 1e-4`: Higher LR is safe since we're only training a small subset
- `--num_epochs 10`: More epochs needed with minimal parameters

### 2. LoRA Fine-Tuning (Recommended for Existing Languages)

Best for: Improving existing language support or domain adaptation (100-1000 samples)

```bash
python scripts/finetune_language.py \
  --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --train_jsonl data/polish_train_with_codes.jsonl \
  --output_dir output/polish_lora \
  --train_mode lora \
  --target_language polish \
  --lora_r 8 \
  --lora_alpha 32 \
  --num_epochs 5 \
  --batch_size 2 \
  --learning_rate 2e-5
```

**Key parameters:**
- `--lora_r 8`: LoRA rank (higher = more capacity, use 8-16)
- `--lora_alpha 32`: LoRA scaling (typically 2-4x the rank)
- `--lora_dropout 0.1`: Dropout for regularization

### 3. Full Fine-Tuning (Advanced)

Best for: Maximum quality with large datasets (1000+ samples)

```bash
python scripts/finetune_language.py \
  --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --train_jsonl data/polish_train_with_codes.jsonl \
  --output_dir output/polish_full \
  --train_mode full \
  --target_language polish \
  --num_epochs 3 \
  --batch_size 2 \
  --gradient_accumulation_steps 4 \
  --learning_rate 2e-5 \
  --gradient_checkpointing
```

**Key parameters:**
- `--gradient_checkpointing`: Saves memory at cost of ~20% speed
- `--gradient_accumulation_steps 4`: Simulate larger batch size
- `--mixed_precision bf16`: Use BF16 for faster training (default)

## Training for Unknown Languages

If your language is not registered in the model, you can still fine-tune! The model will use "auto" language detection:

```bash
python scripts/finetune_language.py \
  --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --train_jsonl data/unknown_lang_train_with_codes.jsonl \
  --output_dir output/unknown_lang \
  --train_mode lang_only \
  --num_epochs 10
```

**Note:** For better results, consider registering your language first using `scripts/register_language.py`:

```bash
python scripts/register_language.py \
  --model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --language polish \
  --output_path ./qwen3-tts-polish

# Then fine-tune on the registered model
python scripts/finetune_language.py \
  --init_model_path ./qwen3-tts-polish \
  --train_jsonl data/polish_train_with_codes.jsonl \
  --output_dir output/polish_lang_only \
  --train_mode lang_only \
  --target_language polish
```

## Advanced Configuration

### Mixed Precision Training

Control precision for speed/memory tradeoffs:

```bash
--mixed_precision bf16  # BF16 (recommended for A100, H100)
--mixed_precision fp16  # FP16 (good for older GPUs)
--mixed_precision no    # FP32 (slowest but most stable)
```

### Learning Rate Scheduling

Built-in linear warmup + decay:

```bash
--learning_rate 2e-5      # Peak learning rate
--warmup_steps 100        # Warmup steps before peak
--weight_decay 0.01       # L2 regularization
```

### Checkpoint Management

```bash
--save_steps 500          # Save every N steps (in addition to epoch saves)
--resume_from_checkpoint output/polish_lora/checkpoint-epoch-2  # Resume training
```

### Logging

```bash
--log_with tensorboard    # TensorBoard (default)
--log_with wandb          # Weights & Biases
--logging_steps 10        # Log every N steps
```

View TensorBoard logs:
```bash
tensorboard --logdir output/polish_lora/logs
```

## Recommended Settings

### Small Dataset (10-100 samples)
```bash
--train_mode lang_only
--num_epochs 10
--batch_size 4
--learning_rate 1e-4
--warmup_steps 50
```

### Medium Dataset (100-1000 samples)
```bash
--train_mode lora
--lora_r 8
--num_epochs 5
--batch_size 2
--gradient_accumulation_steps 4
--learning_rate 2e-5
--warmup_steps 100
```

### Large Dataset (1000+ samples)
```bash
--train_mode full
--num_epochs 3
--batch_size 2
--gradient_accumulation_steps 8
--learning_rate 2e-5
--warmup_steps 200
--gradient_checkpointing
```

## Inference After Fine-Tuning

### For `lang_only` and `full` modes:

```python
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

device = "cuda:0"
tts = Qwen3TTSModel.from_pretrained(
    "output/polish_lang_only/checkpoint-epoch-9",
    device_map=device,
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

# Generate with the fine-tuned language
wavs, sr = tts.generate(
    text="Dzień dobry, jak się masz?",
    language="polish",  # Use your target language
    ref_audio="path/to/ref_audio.wav"
)
sf.write("output.wav", wavs[0], sr)
```

### For `lora` mode:

LoRA adapters are saved separately. Load them with PEFT:

```python
from peft import PeftModel
from qwen_tts import Qwen3TTSModel

# Load base model
tts = Qwen3TTSModel.from_pretrained("Qwen/Qwen3-TTS-12Hz-1.7B-Base")

# Load LoRA adapter
tts.model = PeftModel.from_pretrained(
    tts.model,
    "output/polish_lora/checkpoint-epoch-4"
)

# Use normally
wavs, sr = tts.generate(text="...", language="polish", ref_audio="...")
```

## Troubleshooting

### Out of Memory (OOM)

Try these in order:
1. Reduce `--batch_size` (e.g., from 2 to 1)
2. Increase `--gradient_accumulation_steps` (e.g., from 4 to 8)
3. Enable `--gradient_checkpointing`
4. Use `--train_mode lora` or `--train_mode lang_only`
5. Use `--mixed_precision fp16` or `bf16`

### Loss Not Decreasing

1. Check data quality (audio-text alignment)
2. Increase learning rate for `lang_only` mode
3. Add more training epochs
4. Reduce `--weight_decay` if overfitting is not an issue

### Language Not Recognized

If you get a warning about language not found:
1. Run `scripts/register_language.py` first to add the language token
2. Or simply omit `--target_language` and the model will use "auto" mode

## Hardware Requirements

### Minimum (lang_only mode, 0.6B model)
- GPU: 8GB VRAM (e.g., RTX 3060)
- RAM: 16GB
- Storage: 10GB

### Recommended (lora mode, 1.7B model)
- GPU: 16GB VRAM (e.g., RTX 4080, A4000)
- RAM: 32GB
- Storage: 20GB

### High-End (full mode, 1.7B model)
- GPU: 24GB+ VRAM (e.g., RTX 4090, A5000, A100)
- RAM: 64GB
- Storage: 50GB

## Best Practices

1. **Start Small**: Begin with `lang_only` mode to validate your data pipeline
2. **Monitor Logs**: Watch training loss and generated samples regularly
3. **Validate Early**: Test inference after first epoch to catch issues
4. **Save Often**: Use `--save_steps` for long training runs
5. **Register Languages**: Use `register_language.py` before fine-tuning when possible
6. **Quality Data**: Clean, aligned audio-text pairs are more important than quantity

## Complete Example Workflow

```bash
# 1. Register new language (optional but recommended)
python scripts/register_language.py \
  --model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
  --language polish \
  --output_path ./qwen3-tts-polish

# 2. Prepare training data
cd finetuning
python prepare_data.py \
  --device cuda:0 \
  --tokenizer_model_path Qwen/Qwen3-TTS-Tokenizer-12Hz \
  --input_jsonl ../data/polish_raw.jsonl \
  --output_jsonl ../data/polish_with_codes.jsonl

# 3. Fine-tune with lang_only mode (quick validation)
cd ..
python scripts/finetune_language.py \
  --init_model_path ./qwen3-tts-polish \
  --train_jsonl data/polish_with_codes.jsonl \
  --output_dir output/polish_lang_only \
  --train_mode lang_only \
  --target_language polish \
  --num_epochs 10 \
  --batch_size 4

# 4. Test the model
python -c "
import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

tts = Qwen3TTSModel.from_pretrained(
    'output/polish_lang_only/checkpoint-epoch-9',
    device_map='cuda:0',
    dtype=torch.bfloat16
)

wavs, sr = tts.generate(
    text='Dzień dobry, jak się masz?',
    language='polish',
    ref_audio='data/polish_ref.wav'
)
sf.write('test_output.wav', wavs[0], sr)
"

# 5. If results are good, optionally do LoRA for refinement
python scripts/finetune_language.py \
  --init_model_path ./qwen3-tts-polish \
  --train_jsonl data/polish_with_codes.jsonl \
  --output_dir output/polish_lora \
  --train_mode lora \
  --target_language polish \
  --num_epochs 5
```

## Support

For issues or questions:
- Check the main README: `../README.md`
- Review existing finetuning guide: `../finetuning/README.md`
- Open an issue on GitHub: https://github.com/QwenLM/Qwen3-TTS/issues
