# Tools Directory

This directory contains utility scripts for Qwen3-TTS development and analysis.

## tokenizer_audit.py

A script to audit the **text tokenizer** (Qwen2Tokenizer) used by Qwen3-TTS for Polish language support.

### Purpose

This tool evaluates whether the current text tokenizer adequately handles Polish text or if it needs to be extended with Polish-specific vocabulary. It analyzes:

- **Tokenization efficiency**: Average tokens per character
- **Sequence lengths**: How long Polish text sequences become after tokenization
- **Unknown tokens**: Whether Polish characters or patterns produce unknown/fallback tokens
- **Lossless encoding**: Whether text can be perfectly reconstructed after tokenization

### Usage

```bash
python tools/tokenizer_audit.py --model-path <model-path> [--output <output-file>]
```

**Arguments:**
- `--model-path`: HuggingFace model ID or local path (e.g., `Qwen/Qwen3-TTS-12Hz-1.7B-Base`)
- `--output`: Output JSON file path (default: `tokenizer_audit_results.json`)

**Example:**
```bash
python tools/tokenizer_audit.py \
    --model-path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --output audit_results/polish_audit.json
```

### Test Data

The script includes curated Polish text samples covering:

1. **News**: Formal text with proper nouns, dates, numbers
2. **Conversational**: Informal dialogue with common phrases
3. **Numbers and Dates**: Numeric expressions, prices, phone numbers
4. **Special Characters**: Polish diacritics (ą, ć, ę, ł, ń, ó, ś, ź, ż)
5. **Technical**: Code snippets, URLs, technical terminology
6. **Mixed**: Real-world combinations of text types

### Output

The script produces:

1. **Console output**: Real-time statistics and recommendations
2. **JSON file**: Detailed metrics for each sample and category
3. **Recommendations**: Whether to extend the tokenizer or use baseline

### Interpreting Results

**Tokens per Character Ratio:**
- **< 0.35**: Good efficiency, baseline tokenizer sufficient
- **0.35 - 0.5**: Moderate efficiency, extension could help
- **> 0.5**: Poor efficiency, extension strongly recommended

**Unknown Tokens:**
- **0 unknown tokens**: All Polish characters handled correctly
- **Any unknown tokens**: Tokenizer extension REQUIRED

**Lossless Encoding:**
- **100%**: Perfect reconstruction, no information loss
- **< 100%**: Some data loss, may affect quality

### Integration with CI

The script is integrated with GitHub Actions (`.github/workflows/tokenizer_audit.yml`) to:
- Run automatically on PRs that modify tokenizer or model code
- Store results as artifacts for comparison across changes
- Comment on PRs with summary statistics

### Requirements

- Python 3.9+
- transformers
- torch
- numpy
