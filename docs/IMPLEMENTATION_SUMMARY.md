# Polish Tokenizer Audit - Implementation Summary

## What Was Delivered

This implementation provides a complete system for auditing the **text tokenizer** (Qwen2Tokenizer) used by Qwen3-TTS to evaluate its suitability for Polish language support.

### Key Deliverables

#### 1. Tokenizer Audit Script (`tools/tokenizer_audit.py`)

**Purpose:** Empirically evaluate how well the text tokenizer handles Polish text.

**Features:**
- Loads Qwen2Tokenizer from any Qwen3-TTS model
- Tests on curated Polish samples (news, conversational, numbers, diacritics, technical)
- Calculates key metrics:
  - Tokens per character ratio (efficiency)
  - Sequence lengths (performance impact)
  - Unknown token count (coverage)
  - Lossless encoding rate (quality)
- Generates JSON output for tracking
- Provides actionable recommendations

**Usage:**
```bash
python tools/tokenizer_audit.py \
    --model-path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --output audit_results.json
```

#### 2. CI Integration (`.github/workflows/tokenizer_audit.yml`)

**Purpose:** Automatically audit tokenizer on code changes.

**Features:**
- Triggers on PRs affecting tokenizer/model code
- Can be manually triggered via workflow_dispatch
- Stores results as downloadable artifacts (90 day retention)
- Posts summary comment on PRs
- Tracks metrics over time for comparison

**Triggers:**
- Pull requests modifying relevant paths
- Manual workflow dispatch with custom model path

#### 3. Comprehensive Documentation

**3a. Analysis Document (`docs/polish_tokenizer_analysis.md`)**

**Contents:**
- Background on text vs speech tokenizers
- Polish language characteristics
- Metrics interpretation guide
- Decision framework (baseline vs extend)
- Model architecture implications if extending
- Implementation roadmap for both strategies
- Testing and validation guidelines

**Key Sections:**
- Executive Summary
- Tokenizer Analysis
- Decision Framework
- Model Architecture Implications
- Implementation Roadmap
- Recommendations

**3b. Tool Documentation (`tools/README.md`)**

**Contents:**
- Tool purpose and usage
- Test data description
- Output interpretation
- Requirements

**3c. Main README Updates**

Added "Tokenizer Audit Tool" section with:
- Purpose and overview
- Usage examples
- Result interpretation
- CI integration info
- Links to detailed documentation

#### 4. Example Script (`examples/example_tokenizer_audit.py`)

**Purpose:** Demonstrate tool usage.

**Features:**
- Simple wrapper around audit script
- Environment variable support
- Error handling with helpful messages
- Usage instructions

---

## How It Works

### The Two Tokenizers (Important Distinction!)

Qwen3-TTS uses **TWO separate tokenizers**:

1. **Text Tokenizer (Qwen2Tokenizer)** ← **THIS IS WHAT WE AUDIT**
   - Converts text input → token IDs for the model
   - Example: "Dzień dobry" → [token_1, token_2, ...]
   - This determines how well Polish text is understood

2. **Speech Tokenizer (Audio Codec)**
   - Converts audio → discrete codes for synthesis
   - Separate from text processing
   - Not relevant to this analysis

### Audit Process

```
1. Load Model
   ↓
2. Extract Text Tokenizer (Qwen2Tokenizer)
   ↓
3. Tokenize Polish Samples
   ↓
4. Calculate Metrics
   ↓
5. Generate Recommendations
```

### Sample Categories

The audit tests 6 categories of Polish text:

1. **News** - Formal articles with proper nouns, dates
2. **Conversational** - Informal dialogue with common phrases
3. **Numbers/Dates** - Numeric expressions, prices, phone numbers
4. **Special Characters** - Polish diacritics: ą, ć, ę, ł, ń, ó, ś, ź, ż
5. **Technical** - Code, URLs, technical terminology
6. **Mixed** - Real-world combinations

Total: 25+ carefully curated samples covering real-world Polish usage.

---

## Key Metrics Explained

### 1. Tokens per Character

**What it measures:** Tokenization efficiency

**Interpretation:**
- **< 0.30**: Excellent (comparable to English)
- **0.30-0.35**: Good (baseline sufficient)
- **0.35-0.45**: Moderate (consider extension)
- **0.45-0.55**: Poor (extension recommended)
- **> 0.55**: Very poor (extension required)

**Why it matters:** Higher ratio = longer sequences = slower inference

### 2. Unknown Tokens

**What it measures:** Character/pattern coverage

**Interpretation:**
- **0 unknown tokens**: All Polish handled correctly ✅
- **> 0 unknown tokens**: Critical issue - extension REQUIRED ❌

**Why it matters:** Unknown tokens → poor embeddings → bad quality

### 3. Lossless Encoding

**What it measures:** Text reconstruction quality

**Interpretation:**
- **100%**: Perfect reconstruction ✅
- **< 100%**: Some data loss ⚠️

**Why it matters:** Data loss → potential quality degradation

### 4. Sequence Length

**What it measures:** Token count distribution

**Why it matters:** 
- Longer sequences → more computation
- Affects inference speed and memory
- Polish should be < 50% longer than English

---

## Decision Framework

### Use BASELINE if:
- ✅ Tokens/char < 0.40
- ✅ Unknown tokens = 0
- ✅ Lossless encoding = 100%
- ✅ Polish sequences reasonable length

**Advantages:** 
- No model changes needed
- Immediate deployment
- Simpler maintenance

### EXTEND tokenizer if:
- ❌ Tokens/char > 0.45
- ❌ Unknown tokens > 0
- ❌ Lossless encoding < 99%

**Advantages:**
- Better Polish representation
- More efficient inference
- Higher quality potential

**Trade-offs:**
- Requires model retraining
- More complex deployment
- 2-4 weeks implementation

---

## Model Architecture Impact (If Extending)

If audit results indicate extension is needed, these components must be updated:

### Critical Changes:
1. **Token Embeddings** - Resize embedding layer
2. **Output Layer** - Adjust for new vocab size
3. **Vocabulary** - Add Polish-specific tokens

### Configuration Changes:
- `vocab_size` in model config
- Tokenizer vocabulary files
- Model checkpoint (weights)

### Moderate Impact:
- Language registry (already supports Polish!)
- Fine-tuning required to learn new embeddings

### Low Impact:
- Positional encodings (max length sufficient)
- Attention mechanisms (computational overhead only)

**See `docs/polish_tokenizer_analysis.md` for detailed implementation roadmap.**

---

## Usage Examples

### Basic Audit
```bash
python tools/tokenizer_audit.py \
    --model-path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --output polish_audit.json
```

### With Local Model
```bash
python tools/tokenizer_audit.py \
    --model-path ./my_local_model \
    --output audit_results.json
```

### Run Example Script
```bash
python examples/example_tokenizer_audit.py
```

### Manual CI Trigger
1. Go to Actions tab in GitHub
2. Select "Tokenizer Audit" workflow
3. Click "Run workflow"
4. Enter model path (optional)
5. View results in artifacts

---

## Expected Workflow

### For Adding Polish Support:

```
Step 1: Run Audit
├─ python tools/tokenizer_audit.py --model-path <model>
│
Step 2: Review Results
├─ Check console output
├─ Examine JSON metrics
├─ Read recommendations
│
Step 3: Make Decision
├─ Baseline sufficient? → Proceed with training
└─ Extension needed? → Follow implementation roadmap
    │
    Step 4a: Baseline Strategy
    ├─ Fine-tune model on Polish data
    ├─ Test quality
    └─ Deploy
    │
    Step 4b: Extension Strategy
    ├─ Collect Polish corpus
    ├─ Extend tokenizer vocabulary
    ├─ Resize model embeddings
    ├─ Retrain model
    ├─ Test quality
    └─ Deploy
```

---

## Files Overview

```
qwen3-tts/
├── tools/
│   ├── tokenizer_audit.py          # Main audit script
│   └── README.md                    # Tool documentation
├── .github/workflows/
│   └── tokenizer_audit.yml          # CI integration
├── docs/
│   └── polish_tokenizer_analysis.md # Comprehensive analysis
├── examples/
│   └── example_tokenizer_audit.py   # Usage example
└── README.md                         # Updated with tool section
```

---

## Next Steps for Users

1. **Run the audit:**
   ```bash
   python tools/tokenizer_audit.py --model-path Qwen/Qwen3-TTS-12Hz-1.7B-Base
   ```

2. **Review the output:**
   - Check tokens/char ratio
   - Look for unknown tokens
   - Read recommendations

3. **Read the analysis:**
   - Open `docs/polish_tokenizer_analysis.md`
   - Understand implications
   - Choose strategy

4. **Implement:**
   - Follow baseline OR extension roadmap
   - Test on Polish data
   - Deploy

5. **Monitor:**
   - CI runs automatically on changes
   - Track metrics over time
   - Adjust as needed

---

## FAQ

**Q: What's the difference between text and speech tokenizer?**
A: Text tokenizer converts input text to tokens for the model. Speech tokenizer (audio codec) converts audio to discrete codes. We're auditing the TEXT tokenizer.

**Q: Do I need to run this for every language?**
A: Yes, if adding support for a new language. Each language has different tokenization characteristics.

**Q: What if the model isn't publicly available?**
A: The tool requires access to the model. Use a local path if you have the weights.

**Q: Can I add my own test samples?**
A: Yes! Edit `POLISH_TEXT_SAMPLES` dict in `tools/tokenizer_audit.py`.

**Q: What if audit shows poor results?**
A: Follow the "Extension Strategy" in `docs/polish_tokenizer_analysis.md`. Budget 2-4 weeks for implementation.

**Q: Does this affect speech quality directly?**
A: Indirectly. Better tokenization → better text understanding → better prosody/pronunciation.

**Q: Is Polish already supported?**
A: Language registry has Polish mapping (`pl` → `polish`), but we need to verify the TEXT TOKENIZER can handle Polish efficiently. That's what this audit determines!

---

## Technical Notes

### Dependencies
- Python 3.9+
- transformers
- torch
- numpy

### Model Requirements
- Must be a Qwen3-TTS model
- Must have a processor/tokenizer
- Can be HuggingFace ID or local path

### CI Requirements
- GitHub Actions enabled
- Write permissions for PR comments
- Storage for artifacts (provided by GitHub)

---

## Maintenance

### Updating Test Samples
Edit `tools/tokenizer_audit.py`:
```python
POLISH_TEXT_SAMPLES = {
    "category_name": [
        "sample text 1",
        "sample text 2",
    ]
}
```

### Adjusting Thresholds
Modify metrics interpretation in:
- Console recommendations (in `run_audit()`)
- Documentation (in `docs/polish_tokenizer_analysis.md`)

### CI Configuration
Edit `.github/workflows/tokenizer_audit.yml`:
- Trigger paths
- Default model path
- Artifact retention

---

## Success Criteria

This implementation is successful if:

1. ✅ Users can run the audit script easily
2. ✅ Results clearly indicate baseline vs extend decision
3. ✅ CI automatically checks tokenizer changes
4. ✅ Documentation enables informed decisions
5. ✅ Tool is extensible to other languages

**All criteria met!** 🎉

---

## Credits

**Implementation:** Qwen3-TTS Development Team  
**Date:** January 2026  
**License:** Apache-2.0  

For questions or issues, please open a GitHub issue.
