# Language Token Registration Script

## Overview

The `register_language.py` script allows you to add new language tokens to a Qwen3-TTS model checkpoint. This enables explicit language conditioning during inference for languages not included in the base model.

## Why Use This Script?

Without a dedicated language token:
- You can still fine-tune the model on new language data (see Task 1)
- However, you cannot explicitly condition on the new language at inference time
- The model will use "auto" language detection/fallback

With a registered language token:
- You can pass `language="polish"` (or any registered language) during inference
- The model will use a dedicated language_id token for better conditioning
- The language mapping persists in the model's configuration

## Features

- **Safe ID allocation**: Automatically finds the next available language ID in the codec token range
- **Smart embedding resize**: Only resizes embeddings when necessary to accommodate new tokens
- **Config persistence**: Language mapping is saved in `config.talker_config.codec_language_id`
- **Force overwrite**: Option to update existing language registrations
- **Comprehensive validation**: Ensures model integrity with automatic checks

## Usage

### Basic Usage

Register a new language with auto-allocated ID:

```bash
python scripts/register_language.py \
    --model_path /path/to/qwen3-tts-model \
    --language polish \
    --output_path /path/to/output-model
```

### Advanced Options

#### Specify explicit language ID:

```bash
python scripts/register_language.py \
    --model_path /path/to/qwen3-tts-model \
    --language polish \
    --language_id 4210 \
    --output_path /path/to/output-model
```

#### Overwrite existing language registration:

```bash
python scripts/register_language.py \
    --model_path /path/to/qwen3-tts-model \
    --language polish \
    --output_path /path/to/output-model \
    --force
```

### Arguments

- `--model_path` (required): Path to the base model checkpoint (local directory or HuggingFace repo)
- `--language` (required): Language name to register (e.g., 'polish', 'french'). Will be normalized to lowercase.
- `--output_path` (required): Path where the updated model will be saved
- `--language_id` (optional): Explicit language ID to use. If not specified, the next available ID is auto-allocated.
- `--force` (optional): Force overwrite if the language already exists in the model

## How It Works

1. **Load checkpoint and config**: Loads the model and configuration from the specified path
2. **Allocate language ID**: 
   - If `--language_id` is provided, uses that ID
   - Otherwise, finds the next available ID after existing special tokens and language IDs
   - Default codec special tokens use IDs 4196-4205
3. **Update config**: Adds the new language mapping to `config.talker_config.codec_language_id`
4. **Resize embeddings (if needed)**: 
   - Checks if the new token ID fits within the current vocabulary size
   - If not, resizes the codec embedding layer to accommodate it
   - Preserves existing embedding weights and initializes new embeddings with small random values
5. **Save checkpoint**: Saves the updated model with the new configuration

## Language ID Allocation Strategy

The script uses a safe allocation strategy to avoid conflicts:

1. Collects all used IDs from:
   - Special codec tokens (pad, bos, eos, think, nothink, think_bos, think_eos)
   - Existing language IDs in `codec_language_id`
2. Finds the maximum ID across all used IDs
3. Returns `max_id + 1` as the next available ID

Default special token IDs:
- `codec_pad_id`: 4196
- `codec_bos_id`: 4197
- `codec_eos_token_id`: 4198
- `codec_think_id`: 4202
- `codec_nothink_id`: 4203
- `codec_think_bos_id`: 4204
- `codec_think_eos_id`: 4205

So by default, the first language ID allocated would be 4206.

## Example Workflow

### 1. Register Polish Language

```bash
python scripts/register_language.py \
    --model_path ./qwen3-tts-base \
    --language polish \
    --output_path ./qwen3-tts-base-polish
```

Output:
```
Loading model from ./qwen3-tts-base...
Registering language 'polish' with ID 4206
Loading model weights...
Resizing codec embeddings from 3072 to 4207
Saving updated model to ./qwen3-tts-base-polish...

Registration complete!
  Language: polish
  Language ID: 4206
  Embedding resized: True
  Model saved to: ./qwen3-tts-base-polish

============================================================
SUCCESS: Language registration completed
============================================================
```

### 2. Use the Updated Model

```python
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel

# Load the updated model
model = Qwen3TTSModel.from_pretrained("./qwen3-tts-base-polish")

# Check supported languages
print(model.get_supported_languages())
# Output: ['auto', 'chinese', 'english', 'japanese', 'polish']

# Generate speech with Polish language conditioning
wavs, sr = model.generate(
    text="Cześć, jak się masz?",
    language="polish",  # Now explicitly uses language ID 4206
    speaker="default",
)
```

### 3. Language Resolution

The language registry automatically normalizes language inputs:

```python
# All of these resolve to the same language ID
model.generate(text="...", language="polish")   # -> ID 4206
model.generate(text="...", language="Polish")   # -> ID 4206
model.generate(text="...", language="pl")       # -> ID 4206
model.generate(text="...", language="pl-PL")    # -> ID 4206
```

## Error Handling

The script includes comprehensive error handling:

### Language Already Exists

```bash
$ python scripts/register_language.py --model_path ./model --language polish --output_path ./output

ValueError: Language 'polish' already registered with ID 4206. Use --force to overwrite.
```

Use `--force` to overwrite:

```bash
$ python scripts/register_language.py --model_path ./model --language polish --output_path ./output --force

Warning: Overwriting existing language 'polish'
...
```

### Invalid Model Path

```bash
$ python scripts/register_language.py --model_path /nonexistent --language polish --output_path ./output

OSError: Can't load the configuration of '/nonexistent'...
```

## Testing

The script includes comprehensive unit tests in `tests/test_register_language.py`:

```bash
# Run all tests
python -m pytest tests/test_register_language.py -v

# Run specific test class
python -m pytest tests/test_register_language.py::TestFindNextAvailableLanguageId -v

# Run with coverage
python -m pytest tests/test_register_language.py --cov=scripts.register_language
```

Test coverage includes:
- Language ID allocation with various configurations
- Embedding resize logic and boundary conditions
- Configuration updates
- Integration with language registry
- Error handling for edge cases

## Integration with Fine-tuning

This script is designed to work alongside the fine-tuning workflow (Task 1):

1. **Fine-tune on new language data** (without language token):
   ```bash
   python finetuning/sft.py --data polish_data --output ./finetuned-model
   ```

2. **Register the language token** (enables explicit conditioning):
   ```bash
   python scripts/register_language.py \
       --model_path ./finetuned-model \
       --language polish \
       --output_path ./finetuned-model-with-polish-token
   ```

3. **Use at inference**:
   ```python
   model.generate(text="Cześć!", language="polish")
   ```

## Technical Details

### Config Structure

The language mapping is stored in the model's configuration:

```json
{
  "talker_config": {
    "codec_language_id": {
      "chinese": 4200,
      "english": 4201,
      "japanese": 4202,
      "polish": 4206
    },
    ...
  }
}
```

### Embedding Resize

When resizing embeddings:
- Old embeddings are preserved exactly
- New embeddings are initialized with `Normal(0, initializer_range)`
- The `vocab_size` in the config is updated to match

### Language Registry Integration

The script automatically updates the model's `LanguageRegistry` instance, which handles:
- Language normalization (case-insensitive, ISO code mapping)
- Language resolution (string → language ID)
- Fallback behavior for unknown languages

## Limitations

- The script requires loading the full model into memory
- Resizing embeddings requires retraining or fine-tuning to be effective
- Language IDs must be unique within the model
- The script does not handle distributed/sharded checkpoints

## See Also

- [Task 1: Fine-tuning Documentation](../docs/finetuning.md)
- [Language Registry Documentation](../qwen_tts/langs/README.md)
- [Model Configuration Reference](../qwen_tts/core/models/configuration_qwen3_tts.py)
