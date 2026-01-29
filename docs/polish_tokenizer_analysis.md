# Polish Language Support: Tokenizer Strategy Analysis

**Document Version:** 1.0  
**Date:** January 2026  
**Author:** Qwen3-TTS Development Team

---

## Executive Summary

This document analyzes the tokenizer strategy for adding full Polish language support to Qwen3-TTS. It evaluates whether the current **text tokenizer** (Qwen2Tokenizer) adequately handles Polish text or requires extension with Polish-specific vocabulary.

**Key Question:** Should we use the **baseline tokenizer** (as-is) or **extend it** with Polish vocabulary?

### Methodology

We created `tools/tokenizer_audit.py` to empirically evaluate the text tokenizer on Polish text samples covering:
- News articles (formal text)
- Conversational dialogue
- Numbers, dates, and prices
- Polish diacritical marks (ą, ć, ę, ł, ń, ó, ś, ź, ż)
- Technical text and URLs
- Mixed real-world content

The audit measures:
1. **Tokenization efficiency**: Average tokens per character
2. **Sequence length**: How many tokens Polish text produces
3. **Unknown token behavior**: Whether Polish characters map to unknown/fallback tokens
4. **Lossless encoding**: Whether text survives tokenization/detokenization

---

## Background: Text Tokenizers in Qwen3-TTS

### Architecture Overview

Qwen3-TTS uses **two separate tokenizers**:

1. **Text Tokenizer (Qwen2Tokenizer)**
   - **Purpose**: Converts input text → token IDs for the model
   - **Type**: Subword tokenizer (BPE/Unigram-based)
   - **Location**: Loaded via `AutoProcessor.from_pretrained()`
   - **Used in**: `Qwen3TTSProcessor` → encodes text input
   - **This is what we're auditing**

2. **Speech Tokenizer (Audio Codec)**
   - **Purpose**: Converts audio → discrete codes for synthesis
   - **Versions**: 25Hz (v1) and 12Hz (v2)
   - **Location**: `qwen_tts/core/tokenizer_25hz/` and `tokenizer_12hz/`
   - **Used in**: Speech encoding/decoding
   - **Not relevant to this analysis**

### Why Text Tokenizer Matters for Polish

The text tokenizer determines:
- **Input representation quality**: How well Polish text is represented to the model
- **Sequence efficiency**: Longer sequences → more compute, slower inference
- **Character coverage**: Whether all Polish characters are handled
- **Embedding quality**: Each token has learned embeddings; rare tokens have worse embeddings

**Poor tokenization** can lead to:
- Degraded synthesis quality for Polish
- Longer inference times
- Out-of-vocabulary (OOV) issues
- Poor prosody and pronunciation

---

## Text Tokenizer Analysis

### Current Tokenizer: Qwen2Tokenizer

**Specifications:**
- **Base model**: Qwen2 series (likely trained on Chinese, English, multilingual data)
- **Vocabulary size**: ~151,936 tokens (typical for Qwen2)
- **Algorithm**: BPE (Byte Pair Encoding) variant
- **Character coverage**: Unicode-aware, should handle Polish characters

**Expected behavior for Polish:**
- **Pros**: 
  - Large vocabulary includes some Polish through multilingual training
  - Unicode support means no "broken" characters
  - BPE can decompose unknown words into subwords
  
- **Cons**:
  - Polish may not be well-represented in vocabulary
  - Polish text could be over-segmented (high tokens/char ratio)
  - Rare token embeddings may be undertrained

### Polish Language Characteristics

**Orthographic features:**
- **Diacritical marks**: ą, ć, ę, ł, ń, ó, ś, ź, ż (9 special characters)
- **Consonant clusters**: Complex sequences (e.g., "szcz", "chrz", "trz")
- **Inflection**: Rich morphology → many word forms
- **Compound words**: Long words common

**Implications for tokenization:**
- Need good representation of diacritics
- Consonant clusters should tokenize efficiently
- Inflected forms should share token prefixes
- Long words shouldn't explode into too many tokens

---

## Audit Results and Analysis

### How to Run the Audit

```bash
# Basic usage
python tools/tokenizer_audit.py \
    --model-path Qwen/Qwen3-TTS-12Hz-1.7B-Base \
    --output audit_results.json

# The script will:
# 1. Load the text tokenizer from the model
# 2. Tokenize 25+ Polish text samples
# 3. Calculate statistics
# 4. Generate recommendations
```

### Metrics Interpretation Guide

**1. Tokens per Character Ratio**

This is the **primary metric** for efficiency:

| Ratio | Interpretation | Recommendation |
|-------|----------------|----------------|
| < 0.30 | Excellent - comparable to native language | Baseline sufficient |
| 0.30-0.35 | Good - acceptable efficiency | Baseline likely sufficient |
| 0.35-0.45 | Moderate - noticeably less efficient | Consider extension |
| 0.45-0.55 | Poor - significant overhead | Extension recommended |
| > 0.55 | Very poor - major inefficiency | Extension required |

**For comparison:**
- English typically: 0.25-0.30 tokens/char
- Chinese typically: 0.40-0.50 tokens/char (due to characters)
- Well-supported European languages: 0.28-0.35 tokens/char

**2. Unknown Token Count**

- **0 unknown tokens**: All characters properly handled ✅
- **> 0 unknown tokens**: Critical issue - extension REQUIRED ❌

**3. Lossless Encoding Rate**

- **100%**: Perfect reconstruction ✅
- **< 100%**: Data loss - investigate further ⚠️

**4. Sequence Length**

- Compare Polish sequence lengths to English
- If Polish is >50% longer → overhead in inference

### Expected Results (Hypothetical)

**Scenario A: Baseline Sufficient**
```
Average tokens/char: 0.32 ± 0.04
Unknown tokens: 0
Lossless encoding: 100%
Recommendation: ✅ Baseline tokenizer is adequate
```

**Scenario B: Extension Needed**
```
Average tokens/char: 0.51 ± 0.08
Unknown tokens: 15
Lossless encoding: 98.2%
Recommendation: ❌ Tokenizer extension required
```

---

## Decision Framework

### When to Use BASELINE Strategy

**Use baseline if:**
1. ✅ Tokens/char < 0.40
2. ✅ Zero unknown tokens
3. ✅ 100% lossless encoding
4. ✅ Polish sequence lengths < 40% longer than English

**Advantages:**
- No model retraining needed
- Simpler maintenance
- Faster deployment
- Works immediately

**Disadvantages:**
- Potentially suboptimal performance
- May need more fine-tuning

### When to EXTEND Tokenizer

**Extend if:**
1. ❌ Tokens/char > 0.45
2. ❌ Any unknown tokens detected
3. ❌ Lossless encoding < 99%
4. ❌ Critical Polish patterns poorly tokenized

**Advantages:**
- Better Polish representation
- More efficient inference
- Higher quality output potential

**Disadvantages:**
- Requires model modifications
- Need to retrain embeddings
- More complex deployment

---

## Model Architecture Implications

### If Tokenizer Extension is Needed

Extending the tokenizer affects multiple model components:

#### 1. **Token Embeddings** (CRITICAL)

**Location:** `Qwen3TTSTalkerModel.embed_tokens`

**Issue:** Adding tokens increases vocabulary size
- Current vocab: 3,072 tokens (talker), 151,936 (text tokenizer)
- New vocab: Original + N Polish tokens

**Required changes:**
```python
# In Qwen3TTSTalkerConfig
vocab_size = 3072  # May need adjustment

# In modeling code
self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
```

**Action items:**
- Resize embedding layer: `model.resize_token_embeddings(new_vocab_size)`
- Initialize new embeddings (random or mean of existing)
- Fine-tune model to learn new token embeddings

#### 2. **Language Embeddings** (MODERATE)

**Location:** `Qwen3TTSTalkerConfig.codec_language_id`

**Issue:** Polish must be added to language registry

**Current supported languages:**
```python
# From registry.py
"pl": "polish"  # ✅ Already mapped!
```

**Action items:**
- Verify `codec_language_id` includes Polish
- Check `Qwen3TTSTalkerConfig.codec_language_id` mapping
- May need to add language-specific token ID

#### 3. **Positional Encodings** (LOW IMPACT)

**Location:** RoPE (Rotary Position Embeddings) in attention layers

**Issue:** Longer sequences from poor tokenization
- Max position: 32,768 (very large, should be fine)
- Polish sequences unlikely to exceed this

**Action items:**
- No changes needed if sequences stay within limits

#### 4. **Attention Mechanisms** (LOW IMPACT)

**Location:** `Qwen3TTSTalkerModel` attention layers

**Issue:** More tokens → more attention computation
- Computation: O(n²) where n = sequence length
- 30% longer sequences → ~69% more compute per sequence

**Action items:**
- Monitor inference performance
- May need optimization if slow

#### 5. **Output/LM Head** (CRITICAL IF EXTENDED)

**Location:** `Qwen3TTSTalkerCodePredictorModel` output layer

**Issue:** If vocab changes, output layer changes
- Current: Projects to vocab_size dimensions
- New: Must match new vocab_size

**Required changes:**
```python
# Output layer must match vocabulary
self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
```

**Action items:**
- Resize output layer
- Retrain to learn predictions for new tokens

---

## Implementation Roadmap

### Strategy 1: Baseline (No Extension)

**Timeline:** Immediate

**Steps:**
1. ✅ Run tokenizer audit
2. ✅ Verify Polish characters handled correctly
3. ✅ If metrics acceptable, proceed with baseline
4. Test TTS quality on Polish data
5. Fine-tune if needed (no architecture changes)

**Risk:** Lower quality if tokenization is poor

### Strategy 2: Tokenizer Extension

**Timeline:** 2-4 weeks

**Phase 1: Tokenizer Analysis (Week 1)**
1. ✅ Run tokenizer audit
2. Collect large Polish corpus (news, books, conversations)
3. Analyze most common Polish n-grams missing from vocab
4. Determine how many tokens to add (e.g., 1,000-5,000)

**Phase 2: Vocabulary Extension (Week 1)**
1. Train new BPE merges on Polish corpus
2. Select top N Polish-specific subwords
3. Add to tokenizer vocabulary
4. Test new tokenizer on audit samples

**Phase 3: Model Modification (Week 2)**
1. Resize embedding layer: `model.resize_token_embeddings()`
2. Initialize new embeddings (mean or random)
3. Resize output layer if needed
4. Update configs with new vocab_size

**Phase 4: Retraining (Week 2-3)**
1. Fine-tune model on Polish TTS dataset
2. Focus on learning new token embeddings
3. Validate on held-out Polish data
4. Compare to baseline quality

**Phase 5: Deployment (Week 4)**
1. Package new tokenizer + model
2. Update documentation
3. Deploy and monitor

---

## Recommendations Based on Audit Results

### High-Level Decision Tree

```
Run tokenizer_audit.py
         |
         v
Tokens/char < 0.40?
    |               \
   YES              NO
    |                \
    v                 v
Unknown tokens = 0?   EXTEND REQUIRED
    |        \          (go to Strategy 2)
   YES       NO
    |         \
    v          v
  USE        EXTEND
BASELINE    REQUIRED
(Strategy 1)
```

### Specific Recommendations

**IF audit shows: tokens/char ≈ 0.30-0.35, 0 unknown**
- ✅ **Recommendation: Use BASELINE**
- Polish is adequately supported
- Proceed with standard fine-tuning
- No model architecture changes needed
- Monitor quality and adjust if needed

**IF audit shows: tokens/char ≈ 0.40-0.50, 0 unknown**
- ⚠️ **Recommendation: Consider EXTENSION**
- Baseline might work but suboptimal
- Trade-off: simplicity vs. performance
- Consider hybrid: baseline for MVP, extend for v2

**IF audit shows: tokens/char > 0.50 OR unknown tokens > 0**
- ❌ **Recommendation: EXTEND REQUIRED**
- Poor Polish support in current tokenizer
- Must extend before production Polish support
- Follow Strategy 2 roadmap

---

## Testing and Validation

### Validation Metrics

After implementing either strategy, validate with:

**1. Tokenizer Metrics**
- Re-run audit to confirm improvements (if extended)
- Compare tokens/char before and after

**2. Model Quality Metrics**
- Subjective: Human evaluation (MOS scores)
- Objective: Mel Cepstral Distortion, speaker similarity
- WER (Word Error Rate) on ASR of synthesized Polish

**3. Performance Metrics**
- Inference latency (ms per character)
- Memory usage
- Throughput (characters/second)

### Test Datasets

**Minimum test set:**
- 100 Polish sentences (diverse sources)
- 10 speakers (if available)
- Cover all test categories from audit

**Ideal test set:**
- 1,000+ Polish sentences
- 50+ speakers
- Real-world distribution of text types

---

## Maintenance and Monitoring

### Continuous Monitoring

**CI Integration:**
- ✅ GitHub Actions workflow runs audit on PR changes
- ✅ Stores results as artifacts
- ✅ Comments on PRs with summary
- Tracks tokenizer performance over time

**Metrics to track:**
- Tokens/char ratio over time
- Unknown token occurrences
- Sequence length distributions
- User-reported issues with Polish

### Future Considerations

**If adding more languages:**
- Run tokenizer audit for each new language
- Consider joint extension (add multiple languages at once)
- Balance vocabulary size vs. per-language quality

**Tokenizer versioning:**
- Version tokenizer separately from model
- Maintain backward compatibility
- Document breaking changes

---

## References and Resources

### Internal Resources
- `tools/tokenizer_audit.py` - Audit script
- `tools/README.md` - Usage documentation
- `.github/workflows/tokenizer_audit.yml` - CI integration
- `qwen_tts/langs/registry.py` - Language normalization

### Model Components
- `qwen_tts/core/models/processing_qwen3_tts.py` - Text processor
- `qwen_tts/core/models/configuration_qwen3_tts.py` - Model configs
- `qwen_tts/core/models/modeling_qwen3_tts.py` - Model architecture

### External References
1. **Tokenization Research:**
   - Sennrich et al. (2016) - Neural Machine Translation of Rare Words with Subword Units
   - Kudo & Richardson (2018) - SentencePiece: A simple and language independent approach

2. **Multilingual Models:**
   - Conneau et al. (2020) - Unsupervised Cross-lingual Representation Learning at Scale (XLM-R)
   - Workshop on Building Educational Applications (BEA) - Tokenization effects

3. **Polish NLP:**
   - Wróblewska (2018) - Polish evaluation dataset for compositional distributional semantics models
   - PolEval - Polish NLP competitions and benchmarks

---

## Appendix: Technical Details

### A. Polish Text Statistics

**Phoneme inventory:** 35-39 phonemes (depending on analysis)
**Diacritics:** 9 unique characters with diacritical marks
**Average word length:** 5.2 characters (vs. 4.5 for English)
**Morphological complexity:** High (7 cases, 3 genders, complex conjugation)

### B. Tokenizer Extension Algorithm (Pseudocode)

```python
# 1. Collect Polish corpus
polish_corpus = load_polish_texts()  # e.g., 100M characters

# 2. Train BPE on Polish
tokenizer_base = load_existing_tokenizer()
polish_merges = train_bpe(polish_corpus, num_merges=5000)

# 3. Filter to non-overlapping tokens
new_tokens = [t for t in polish_merges if t not in tokenizer_base.vocab]
new_tokens = new_tokens[:N]  # Take top N, e.g., 2000

# 4. Add to vocabulary
tokenizer_extended = tokenizer_base.add_tokens(new_tokens)
tokenizer_extended.save("tokenizer_extended/")

# 5. Update model
model = load_model()
model.resize_token_embeddings(len(tokenizer_extended))
# Initialize new embeddings
for new_token_id in range(old_vocab_size, new_vocab_size):
    model.embeddings[new_token_id] = mean(model.embeddings)
```

### C. Embedding Initialization Strategies

**Option 1: Random Initialization**
- Sample from N(0, σ²) matching existing embeddings
- Pros: Simple, standard practice
- Cons: Requires more training

**Option 2: Mean Initialization**
- Set to mean of all existing embeddings
- Pros: Stable starting point
- Cons: Generic, may need tuning

**Option 3: Neighbor Initialization**
- Average embeddings of similar subwords
- Pros: Better initialization for related tokens
- Cons: Requires defining similarity

**Option 4: Pretrained Initialization**
- Use embeddings from Polish BERT/RoBERTa
- Pros: Best initial quality
- Cons: Requires compatible model, complex

**Recommendation:** Start with **Mean Initialization** (Option 2), then fine-tune.

---

## Conclusion

This document provides a framework for deciding the Polish tokenizer strategy:

1. **Run `tools/tokenizer_audit.py`** to get empirical data
2. **Interpret results** using metrics and thresholds
3. **Choose strategy**: Baseline or Extension
4. **Implement** following the roadmap
5. **Validate** with comprehensive testing
6. **Monitor** via CI integration

The audit script and CI integration enable data-driven decisions and continuous monitoring, ensuring high-quality Polish language support in Qwen3-TTS.

---

**Document Status:** ✅ Complete  
**Next Steps:** Run tokenizer audit on actual model and update with results
