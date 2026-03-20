#!/usr/bin/env python
# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Fine-tune Qwen3-TTS model for new language support.

This script provides a production-grade CLI to fine-tune a Qwen3-TTS model
on a new language. It supports three training modes:
- lora: Use LoRA (Low-Rank Adaptation) for efficient fine-tuning
- full: Full model fine-tuning (all parameters)
- lang_only: Only update language embedding row(s) and minimal adapters

Features:
- Hugging Face accelerate for distributed training
- Optional PEFT LoRA support
- bf16/fp16 mixed precision training
- Gradient checkpointing and accumulation
- Checkpoint saving and resuming
- TensorBoard logging
"""

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
from accelerate import Accelerator
from safetensors.torch import save_file
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoConfig, get_linear_schedule_with_warmup

# Add parent directory to path to import qwen_tts
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Import after path is set
from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
from qwen_tts.core.models import Qwen3TTSForConditionalGeneration
from qwen_tts.langs.registry import LanguageRegistry
from tools.tokenizer_audit import analyze_tokenization

# Try importing peft, but make it optional
try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    LoraConfig = None
    get_peft_model = None


ROLE_TOKEN_COUNT = 3
SPEAKER_PLACEHOLDER_TOKEN_ID = 0
NUM_CODEC_CHANNELS = 16
IGNORE_INDEX = -100
DEFAULT_LANGUAGE = "auto"
CONFIG_FILES_TO_COPY = ["config.json", "generation_config.json", "tokenizer_config.json"]


@dataclass(frozen=True)
class TrainingSequenceLayout:
    """Resolved layout for a single training sample."""

    prompt_tokens: Tuple[int, ...]
    prompt_start: int
    prompt_end: int
    speaker_position: int
    text_start: int
    text_end: int
    text_eos_position: int
    codec_bos_position: int
    codec_start: int
    codec_end: int
    codec_eos_position: int
    sequence_length: int


def get_language_registry_from_config(config) -> LanguageRegistry:
    """Build the shared language registry used by both training and inference."""
    talker_config = getattr(config, "talker_config", None)
    codec_language_id = getattr(talker_config, "codec_language_id", None) if talker_config is not None else None
    return LanguageRegistry(codec_language_id)


def resolve_training_language(
    registry: LanguageRegistry,
    sample_language: Optional[str],
    target_language: Optional[str] = None,
    *,
    strict: bool = False,
) -> Tuple[str, Optional[int]]:
    """
    Normalize and resolve the language used for conditioning in training.

    `target_language` overrides per-sample language in the same way users expect at inference time.
    """
    requested_language = target_language if target_language is not None else sample_language
    normalized_language = registry.normalize_language(requested_language or DEFAULT_LANGUAGE)
    language_id = registry.resolve_language(normalized_language, strict=strict)
    return normalized_language, language_id


def validate_lang_only_target_language(config, target_language: Optional[str]) -> Tuple[str, int]:
    """Require an explicit registered language token for lang_only training."""
    if not target_language:
        raise ValueError("lang_only mode requires --target_language to be set to a registered language.")

    registry = get_language_registry_from_config(config)
    normalized_language, language_id = resolve_training_language(
        registry,
        sample_language=None,
        target_language=target_language,
        strict=True,
    )
    if language_id is None:
        raise ValueError(
            f"Language '{normalized_language}' is not registered in talker_config.codec_language_id."
        )
    return normalized_language, language_id


def build_codec_conditioning_tokens(config, language_id: Optional[int]) -> List[int]:
    """Mirror inference-time language/speaker conditioning tokens for training."""
    if language_id is None:
        return [
            config.talker_config.codec_nothink_id,
            config.talker_config.codec_think_bos_id,
            config.talker_config.codec_think_eos_id,
            SPEAKER_PLACEHOLDER_TOKEN_ID,
            config.talker_config.codec_pad_id,
            config.talker_config.codec_bos_id,
        ]

    return [
        config.talker_config.codec_think_id,
        config.talker_config.codec_think_bos_id,
        language_id,
        config.talker_config.codec_think_eos_id,
        SPEAKER_PLACEHOLDER_TOKEN_ID,
        config.talker_config.codec_pad_id,
        config.talker_config.codec_bos_id,
    ]


def build_training_sequence_layout(text_ids_len: int, codec_ids_len: int, prompt_tokens: Sequence[int]) -> TrainingSequenceLayout:
    """Return named positions for the collate_fn sequence layout."""
    prompt_positions = len(prompt_tokens) - 1
    prompt_start = ROLE_TOKEN_COUNT
    prompt_end = prompt_start + prompt_positions
    text_start = prompt_end
    text_payload_len = text_ids_len - ROLE_TOKEN_COUNT
    text_end = text_start + text_payload_len
    text_eos_position = text_end
    codec_bos_position = text_eos_position + 1
    codec_start = codec_bos_position + 1
    codec_end = codec_start + codec_ids_len
    codec_eos_position = codec_end
    sequence_length = codec_eos_position + 1

    return TrainingSequenceLayout(
        prompt_tokens=tuple(prompt_tokens),
        prompt_start=prompt_start,
        prompt_end=prompt_end,
        speaker_position=prompt_start + (len(prompt_tokens) - 3),
        text_start=text_start,
        text_end=text_end,
        text_eos_position=text_eos_position,
        codec_bos_position=codec_bos_position,
        codec_start=codec_start,
        codec_end=codec_end,
        codec_eos_position=codec_eos_position,
        sequence_length=sequence_length,
    )


def find_unregistered_training_languages(
    train_data: Sequence[dict],
    registry: LanguageRegistry,
    target_language: Optional[str],
) -> List[str]:
    """List normalized languages in training data that are not registered in the codec mapping."""
    if target_language:
        normalized, language_id = resolve_training_language(
            registry,
            sample_language=None,
            target_language=target_language,
            strict=False,
        )
        if normalized != DEFAULT_LANGUAGE and language_id is None:
            return [normalized]
        return []

    missing = set()
    for item in train_data:
        normalized, language_id = resolve_training_language(
            registry,
            sample_language=item.get("language", DEFAULT_LANGUAGE),
            target_language=None,
            strict=False,
        )
        if normalized != DEFAULT_LANGUAGE and language_id is None:
            missing.add(normalized)
    return sorted(missing)


def run_tokenizer_preflight_audit(
    processor,
    train_data: Sequence[dict],
    languages: Sequence[str],
    output_dir: str,
    max_samples: int = 128,
):
    """Audit tokenizer behavior on training texts before training an unregistered language."""
    if not languages:
        return None

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        print("Skipping tokenizer preflight audit: processor does not expose a tokenizer.")
        return None

    samples = [item["text"] for item in train_data[:max_samples] if item.get("text")]
    if not samples:
        print("Skipping tokenizer preflight audit: no training texts available.")
        return None

    analyses = [analyze_tokenization(tokenizer, text) for text in samples]
    avg_tokens_per_char = sum(item["tokens_per_char"] for item in analyses) / len(analyses)
    max_sequence_length = max(item["num_tokens"] for item in analyses)
    total_unk_tokens = sum(item["num_unk_tokens"] for item in analyses)
    lossless_count = sum(1 for item in analyses if item["is_lossless"])
    summary = {
        "languages": list(languages),
        "num_samples": len(analyses),
        "avg_tokens_per_char": avg_tokens_per_char,
        "max_sequence_length": max_sequence_length,
        "total_unk_tokens": total_unk_tokens,
        "lossless_encoding_rate": lossless_count / len(analyses),
    }

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "tokenizer_audit_preflight.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "samples": analyses}, f, indent=2, ensure_ascii=False)

    print(
        "Tokenizer preflight audit for unregistered language(s) "
        f"{', '.join(languages)}: avg_tokens_per_char={avg_tokens_per_char:.3f}, "
        f"max_sequence_length={max_sequence_length}, total_unk_tokens={total_unk_tokens}, "
        f"lossless={lossless_count}/{len(analyses)}. Saved to {output_path}"
    )
    return summary


def forward_training_batch(model, batch):
    """Shared forward path for training and tests."""
    input_ids = batch["input_ids"]
    codec_ids = batch["codec_ids"]
    ref_mels = batch["ref_mels"]
    text_embedding_mask = batch["text_embedding_mask"]
    codec_embedding_mask = batch["codec_embedding_mask"]
    attention_mask = batch["attention_mask"]
    codec_0_labels = batch["codec_0_labels"]
    codec_mask = batch["codec_mask"]

    speaker_embedding = model.speaker_encoder(
        ref_mels.to(model.device).to(model.dtype)
    ).detach()

    input_text_ids = input_ids[:, :, 0]
    input_codec_ids = input_ids[:, :, 1]

    input_text_embedding = model.talker.model.text_embedding(input_text_ids) * text_embedding_mask
    input_codec_embedding = model.talker.model.codec_embedding(input_codec_ids) * codec_embedding_mask
    row_indices = torch.arange(input_codec_embedding.shape[0], device=input_codec_embedding.device)
    speaker_positions = batch["speaker_positions"].to(input_codec_embedding.device)
    input_codec_embedding[row_indices, speaker_positions, :] = speaker_embedding

    input_embeddings = input_text_embedding + input_codec_embedding

    for i in range(1, NUM_CODEC_CHANNELS):
        codec_i_embedding = model.talker.code_predictor.get_input_embeddings()[i - 1](codec_ids[:, :, i])
        codec_i_embedding = codec_i_embedding * codec_mask.unsqueeze(-1)
        input_embeddings = input_embeddings + codec_i_embedding

    outputs = model.talker(
        inputs_embeds=input_embeddings[:, :-1, :],
        attention_mask=attention_mask[:, :-1],
        labels=codec_0_labels[:, 1:],
        output_hidden_states=True,
    )

    hidden_states = outputs.hidden_states[0][-1]
    talker_hidden_states = hidden_states[codec_mask[:, 1:]]
    talker_codec_ids = codec_ids[codec_mask]
    _, sub_talker_loss = model.talker.forward_sub_talker_finetune(
        talker_codec_ids,
        talker_hidden_states,
    )

    return outputs.loss + sub_talker_loss


class LanguageDataset(torch.utils.data.Dataset):
    """Dataset for language-specific fine-tuning."""
    
    def __init__(self, data_list, processor, config, target_language=None):
        """
        Initialize the dataset.
        
        Args:
            data_list: List of data samples (JSONL parsed)
            processor: The TTS processor
            config: Model configuration
            target_language: Target language for fine-tuning (optional)
        """
        self.data_list = data_list
        self.processor = processor
        self.config = config
        self.target_language = target_language
        self.language_registry = get_language_registry_from_config(config)
        
        # Import here to avoid circular dependencies
        import librosa
        import numpy as np
        from qwen_tts.core.models.modeling_qwen3_tts import mel_spectrogram
        
        self.librosa = librosa
        self.np = np
        self.mel_spectrogram_fn = mel_spectrogram
    
    def __len__(self):
        return len(self.data_list)
    
    def _load_audio_to_np(self, x: str):
        """Load audio file to numpy array."""
        audio, sr = self.librosa.load(x, sr=None, mono=True)
        if audio.ndim > 1:
            audio = self.np.mean(audio, axis=-1)
        return audio.astype(self.np.float32), int(sr)
    
    def _build_assistant_text(self, text: str) -> str:
        """Build assistant text format."""
        return f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
    
    def _tokenize_texts(self, text):
        """Tokenize text input."""
        input_data = self.processor(text=text, return_tensors="pt", padding=True)
        input_id = input_data["input_ids"]
        input_id = input_id.unsqueeze(0) if input_id.dim() == 1 else input_id
        return input_id
    
    @torch.inference_mode()
    def extract_mels(self, audio, sr):
        """Extract mel spectrograms from audio."""
        assert sr == 24000, "Only support 24kHz audio"
        mels = self.mel_spectrogram_fn(
            torch.from_numpy(audio).unsqueeze(0),
            n_fft=1024,
            num_mels=128,
            sampling_rate=24000,
            hop_size=256,
            win_size=1024,
            fmin=0,
            fmax=12000
        ).transpose(1, 2)
        return mels
    
    def __getitem__(self, idx):
        """Get a single item from the dataset."""
        item = self.data_list[idx]
        
        audio_path = item["audio"]
        text = item["text"]
        audio_codes = item["audio_codes"]
        language = item.get('language', DEFAULT_LANGUAGE)
        ref_audio_path = item['ref_audio']
        normalized_language, language_id = resolve_training_language(
            self.language_registry,
            sample_language=language,
            target_language=self.target_language,
            strict=False,
        )
        
        text = self._build_assistant_text(text)
        text_ids = self._tokenize_texts(text)
        
        audio_codes = torch.tensor(audio_codes, dtype=torch.long)
        
        # Load reference audio
        ref_audio_list = [ref_audio_path] if not isinstance(ref_audio_path, list) else ref_audio_path
        wav, sr = self._load_audio_to_np(ref_audio_list[0])
        ref_mel = self.extract_mels(audio=wav, sr=sr)
        
        # Remove last 5 tokens which are typically special end tokens
        # This matches the format expected by the model during training
        return {
            "text_ids": text_ids[:, :-5],
            "audio_codes": audio_codes,
            "ref_mel": ref_mel,
            "language": normalized_language,
            "language_id": language_id,
        }
    
    def collate_fn(self, batch):
        """Collate function for DataLoader."""
        layouts = []
        for data in batch:
            prompt_tokens = build_codec_conditioning_tokens(self.config, data.get("language_id"))
            layouts.append(
                build_training_sequence_layout(
                    text_ids_len=data["text_ids"].shape[1],
                    codec_ids_len=data["audio_codes"].shape[0],
                    prompt_tokens=prompt_tokens,
                )
            )

        max_length = max(layout.sequence_length for layout in layouts)
        b, t = len(batch), max_length

        input_ids = torch.zeros((b, t, 2), dtype=torch.long)
        codec_ids = torch.zeros((b, t, NUM_CODEC_CHANNELS), dtype=torch.long)
        text_embedding_mask = torch.zeros((b, t), dtype=torch.bool)
        codec_embedding_mask = torch.zeros((b, t), dtype=torch.bool)
        codec_mask = torch.zeros((b, t), dtype=torch.bool)
        attention_mask = torch.zeros((b, t), dtype=torch.long)
        codec_0_labels = torch.full((b, t), IGNORE_INDEX, dtype=torch.long)
        speaker_positions = torch.zeros((b,), dtype=torch.long)

        for i, (data, layout) in enumerate(zip(batch, layouts)):
            text_ids = data['text_ids']
            audio_codec_0 = data['audio_codes'][:, 0]
            audio_codecs = data['audio_codes']

            input_ids[i, :ROLE_TOKEN_COUNT, 0] = text_ids[0, :ROLE_TOKEN_COUNT]
            input_ids[i, layout.prompt_start:layout.prompt_end, 0] = self.config.tts_pad_token_id
            input_ids[i, layout.prompt_end - 1, 0] = self.config.tts_bos_token_id
            input_ids[i, layout.text_start:layout.text_end, 0] = text_ids[0, ROLE_TOKEN_COUNT:]
            input_ids[i, layout.text_eos_position, 0] = self.config.tts_eos_token_id
            input_ids[i, layout.codec_bos_position:layout.codec_eos_position + 1, 0] = self.config.tts_pad_token_id
            text_embedding_mask[i, :layout.sequence_length] = True

            input_ids[i, layout.prompt_start:layout.prompt_end, 1] = torch.tensor(
                layout.prompt_tokens[:-1],
                dtype=torch.long,
            )
            input_ids[i, layout.text_start:layout.text_eos_position + 1, 1] = self.config.talker_config.codec_pad_id
            input_ids[i, layout.codec_bos_position, 1] = self.config.talker_config.codec_bos_id
            input_ids[i, layout.codec_start:layout.codec_end, 1] = audio_codec_0
            input_ids[i, layout.codec_eos_position, 1] = self.config.talker_config.codec_eos_token_id

            codec_0_labels[i, layout.codec_start:layout.codec_end] = audio_codec_0
            codec_0_labels[i, layout.codec_eos_position] = self.config.talker_config.codec_eos_token_id

            codec_ids[i, layout.codec_start:layout.codec_end, :] = audio_codecs

            codec_embedding_mask[i, layout.prompt_start:layout.codec_eos_position + 1] = True
            codec_embedding_mask[i, layout.speaker_position] = False

            codec_mask[i, layout.codec_start:layout.codec_end] = True
            attention_mask[i, :layout.sequence_length] = True
            speaker_positions[i] = layout.speaker_position

        ref_mels = [data['ref_mel'] for data in batch]
        ref_mels = torch.cat(ref_mels, dim=0)
        
        return {
            'input_ids': input_ids,
            'ref_mels': ref_mels,
            'attention_mask': attention_mask,
            'text_embedding_mask': text_embedding_mask.unsqueeze(-1),
            'codec_embedding_mask': codec_embedding_mask.unsqueeze(-1),
            'codec_0_labels': codec_0_labels,
            'codec_ids': codec_ids,
            'codec_mask': codec_mask,
            'speaker_positions': speaker_positions,
        }


def setup_lora_model(model, lora_config):
    """
    Setup LoRA for the model.
    
    Args:
        model: The model to apply LoRA to
        lora_config: LoRA configuration dict
        
    Returns:
        Model with LoRA applied
    """
    if not PEFT_AVAILABLE:
        raise ImportError(
            "PEFT library not available. Install it with: pip install peft"
        )
    
    # Configure LoRA
    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=lora_config.get('r', 8),
        lora_alpha=lora_config.get('lora_alpha', 32),
        lora_dropout=lora_config.get('lora_dropout', 0.1),
        target_modules=lora_config.get('target_modules', [
            'q_proj', 'k_proj', 'v_proj', 'o_proj',
            'gate_proj', 'up_proj', 'down_proj'
        ]),
        bias="none",
    )
    
    # Apply LoRA to model
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    return model


def register_language_embedding_gradient_mask(model, language_id: int):
    """
    Register a gradient hook so only one language row in codec_embedding is updated.

    Args:
        model: Qwen3-TTS model with talker.model.codec_embedding
        language_id: Row index in codec embedding to keep trainable

    Returns:
        torch.utils.hooks.RemovableHandle

    Raises:
        ValueError: If codec embedding is missing or language_id is out of range.
    """
    if not hasattr(model, 'talker') or not hasattr(model.talker, 'model'):
        raise ValueError("Model does not have talker.model for language embedding masking")
    if not hasattr(model.talker.model, 'codec_embedding'):
        raise ValueError("Model does not have talker.model.codec_embedding")

    embedding_weight = model.talker.model.codec_embedding.weight
    vocab_size = embedding_weight.shape[0]
    if language_id < 0 or language_id >= vocab_size:
        raise ValueError(
            f"language_id={language_id} out of codec_embedding range [0, {vocab_size - 1}]"
        )

    def _mask_grad(grad):
        masked_grad = torch.zeros_like(grad)
        masked_grad[language_id] = grad[language_id]
        return masked_grad

    return embedding_weight.register_hook(_mask_grad)


def freeze_all_except_language_embeddings(model, language_id: Optional[int] = None):
    """
    Freeze all parameters except language embeddings and minimal adapters.

    For lang_only mode, this function optionally applies row-wise gradient
    masking so only `language_id` in codec embedding is updated.

    Args:
        model: The model to freeze
        language_id: The specific language ID to train (if known)

    Returns:
        Optional hook handle for language-row gradient masking.
    """
    # Number of top layers to unfreeze for adaptation
    NUM_LAYERS_TO_UNFREEZE = 2
    language_row_hook = None

    # First, freeze everything
    for param in model.parameters():
        param.requires_grad = False

    # Unfreeze codec embedding (where language tokens live)
    if hasattr(model, 'talker') and hasattr(model.talker, 'model'):
        if hasattr(model.talker.model, 'codec_embedding'):
            model.talker.model.codec_embedding.weight.requires_grad = True
            if language_id is not None:
                language_row_hook = register_language_embedding_gradient_mask(model, language_id)
                print(f"Unfroze codec_embedding with row-only gradient mask (language_id={language_id})")
            else:
                print("Unfroze codec_embedding (all rows; no language_id provided)")

    # Optionally unfreeze a few top layers for adaptation
    # This allows the model to adapt to the new language slightly
    num_layers = model.talker.model.config.num_hidden_layers

    for i in range(num_layers - NUM_LAYERS_TO_UNFREEZE, num_layers):
        layer = model.talker.model.layers[i]
        for param in layer.parameters():
            param.requires_grad = True

    print(f"Unfroze last {NUM_LAYERS_TO_UNFREEZE} transformer layers for adaptation")

    # Print trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
          f"({100 * trainable_params / total_params:.2f}%)")

    return language_row_hook


def create_optimizer(model, learning_rate: float, weight_decay: float, language_id: Optional[int] = None):
    """Create AdamW optimizer with safe weight decay behavior for lang_only row masking.

    When `language_id` is set, the codec embedding uses row-level gradient masking.
    In that mode, decoupled weight decay must be disabled for the embedding tensor,
    otherwise non-target rows would still drift.
    """
    embedding_weight = None
    if (
        language_id is not None
        and hasattr(model, 'talker')
        and hasattr(model.talker, 'model')
        and hasattr(model.talker.model, 'codec_embedding')
    ):
        embedding_weight = model.talker.model.codec_embedding.weight

    if embedding_weight is None:
        return AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    embedding_id = id(embedding_weight)
    other_trainable_params = [
        param for param in model.parameters() if param.requires_grad and id(param) != embedding_id
    ]

    param_groups = [
        {
            'params': [embedding_weight],
            'weight_decay': 0.0,
        }
    ]
    if other_trainable_params:
        param_groups.append({'params': other_trainable_params, 'weight_decay': weight_decay})

    return AdamW(param_groups, lr=learning_rate)


def save_checkpoint(
    accelerator: Accelerator,
    model,
    optimizer,
    scheduler,
    epoch: int,
    step: int,
    output_dir: str,
    model_path: str,
    args
):
    """
    Save a checkpoint.
    
    Args:
        accelerator: Accelerator instance
        model: The model to save
        optimizer: The optimizer state to save
        scheduler: The scheduler state to save
        epoch: Current epoch number (1-indexed for checkpoint naming)
        step: Current step number
        output_dir: Output directory for checkpoints
        model_path: Original model path (for copying config files)
        args: Training arguments
    """
    if not accelerator.is_main_process:
        return
    
    checkpoint_dir = os.path.join(output_dir, f"checkpoint-epoch-{epoch}")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    print(f"Saving checkpoint to {checkpoint_dir}")
    
    # Copy config and other files from original model
    import shutil
    for config_file in CONFIG_FILES_TO_COPY:
        src = os.path.join(model_path, config_file)
        if os.path.exists(src):
            dst = os.path.join(checkpoint_dir, config_file)
            shutil.copy2(src, dst)
    
    # Update config with target language if specified
    config_path = os.path.join(checkpoint_dir, 'config.json')
    if os.path.exists(config_path) and args.target_language:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
        
        # Add or update language in codec_language_id if not already there
        talker_config = config_dict.get("talker_config", {})
        codec_language_id = talker_config.get("codec_language_id", {})
        
        # If language is not in the mapping, it might be a new one
        # We'll document this in the config but not add it automatically
        # (user should use register_language.py for that)
        
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
    
    # Save model weights
    unwrapped_model = accelerator.unwrap_model(model)
    
    if args.train_mode == 'lora' and PEFT_AVAILABLE:
        # For LoRA, save only the adapter weights
        unwrapped_model.save_pretrained(checkpoint_dir)
    else:
        # Save full model state dict
        state_dict = {k: v.detach().cpu() for k, v in unwrapped_model.state_dict().items()}
        save_path = os.path.join(checkpoint_dir, "model.safetensors")
        save_file(state_dict, save_path)
    
    # Save training state (for resuming)
    training_state = {
        'epoch': epoch,
        'step': step,
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'args': vars(args),
    }
    
    torch.save(training_state, os.path.join(checkpoint_dir, 'training_state.pt'))
    
    print(f"Checkpoint saved successfully")


def load_checkpoint(checkpoint_dir: str, model, optimizer, scheduler):
    """
    Load a checkpoint for resuming training.
    
    Args:
        checkpoint_dir: Directory containing the checkpoint
        model: Model to load weights into
        optimizer: Optimizer to load state into
        scheduler: Scheduler to load state into
        
    Returns:
        Tuple of (epoch-1, step) where epoch is 0-indexed for use in training loop
    """
    training_state_path = os.path.join(checkpoint_dir, 'training_state.pt')
    
    if not os.path.exists(training_state_path):
        raise FileNotFoundError(f"No training state found at {training_state_path}")
    
    print(f"Loading checkpoint from {checkpoint_dir}")
    
    # Load training state with explicit settings for PyTorch 2.0+
    training_state = torch.load(
        training_state_path,
        map_location='cpu',
        weights_only=False  # Required for optimizer/scheduler state dicts
    )
    # Epoch is stored as 1-indexed, convert back to 0-indexed for training loop
    epoch = training_state['epoch'] - 1
    step = training_state['step']
    
    # Load optimizer state
    optimizer.load_state_dict(training_state['optimizer_state_dict'])
    
    # Load scheduler state
    if scheduler and training_state['scheduler_state_dict']:
        scheduler.load_state_dict(training_state['scheduler_state_dict'])
    
    print(f"Resumed from checkpoint epoch {epoch + 1}, step {step}")
    
    return epoch, step


def train():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Fine-tune Qwen3-TTS for new language support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full fine-tuning
  python scripts/finetune_language.py \\
      --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \\
      --train_jsonl data/polish_train.jsonl \\
      --output_dir output/polish_full \\
      --train_mode full \\
      --target_language polish

  # LoRA fine-tuning (efficient)
  python scripts/finetune_language.py \\
      --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \\
      --train_jsonl data/polish_train.jsonl \\
      --output_dir output/polish_lora \\
      --train_mode lora \\
      --lora_r 8 \\
      --target_language polish

  # Language embedding only (most efficient)
  python scripts/finetune_language.py \\
      --init_model_path Qwen/Qwen3-TTS-12Hz-1.7B-Base \\
      --train_jsonl data/polish_train.jsonl \\
      --output_dir output/polish_lang_only \\
      --train_mode lang_only \\
      --target_language polish
        """
    )
    
    # Model and data arguments
    parser.add_argument(
        "--init_model_path",
        type=str,
        default="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        help="Path or name of the initial model to fine-tune"
    )
    parser.add_argument(
        "--train_jsonl",
        type=str,
        required=True,
        help="Path to training JSONL file (output from prepare_data.py)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="output",
        help="Directory to save checkpoints and logs"
    )
    
    # Training mode
    parser.add_argument(
        "--train_mode",
        type=str,
        choices=['lora', 'full', 'lang_only'],
        default='full',
        help="Training mode: lora (efficient), full (all params), lang_only (embeddings only)"
    )
    parser.add_argument(
        "--target_language",
        type=str,
        default=None,
        help="Target language for fine-tuning (e.g., 'polish'). "
             "If not specified, works with 'auto' (unknown) language."
    )
    
    # LoRA specific arguments
    parser.add_argument(
        "--lora_r",
        type=int,
        default=8,
        help="LoRA rank (only used when train_mode=lora)"
    )
    parser.add_argument(
        "--lora_alpha",
        type=int,
        default=32,
        help="LoRA alpha (only used when train_mode=lora)"
    )
    parser.add_argument(
        "--lora_dropout",
        type=float,
        default=0.1,
        help="LoRA dropout (only used when train_mode=lora)"
    )
    
    # Training hyperparameters
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Training batch size per device"
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of gradient accumulation steps"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=2e-5,
        help="Learning rate"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=100,
        help="Number of warmup steps for learning rate scheduler"
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=1.0,
        help="Maximum gradient norm for clipping"
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.01,
        help="Weight decay"
    )
    
    # Mixed precision and optimization
    parser.add_argument(
        "--mixed_precision",
        type=str,
        choices=['no', 'fp16', 'bf16'],
        default='bf16',
        help="Mixed precision training: no, fp16, or bf16"
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action='store_true',
        help="Enable gradient checkpointing to save memory"
    )
    
    # Logging and checkpointing
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Log every N steps"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=None,
        help="Save checkpoint every N steps (if None, save every epoch)"
    )
    parser.add_argument(
        "--log_with",
        type=str,
        default="tensorboard",
        choices=["tensorboard", "wandb", "all"],
        help="Logging backend to use"
    )
    
    # Resume training
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint directory to resume training from"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Validate arguments
    if args.train_mode == 'lora' and not PEFT_AVAILABLE:
        print("ERROR: LoRA mode requires the 'peft' library.")
        print("Install it with: pip install peft")
        sys.exit(1)
    
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.log_with,
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize logging
    if accelerator.is_main_process:
        accelerator.init_trackers(
            project_name="qwen3-tts-language-finetuning",
            config=vars(args),
            init_kwargs={
                "tensorboard": {"log_dir": os.path.join(args.output_dir, "logs")}
            } if args.log_with == "tensorboard" else {}
        )
    
    # Load model
    accelerator.print(f"Loading model from {args.init_model_path}...")
    
    # Determine dtype based on mixed precision
    if args.mixed_precision == 'bf16':
        dtype = torch.bfloat16
    elif args.mixed_precision == 'fp16':
        dtype = torch.float16
    else:
        dtype = torch.float32
    
    # Load model using Qwen3TTSModel wrapper
    qwen3tts = Qwen3TTSModel.from_pretrained(
        args.init_model_path,
        torch_dtype=dtype,
        attn_implementation="flash_attention_2",
    )
    
    # Load config
    config = AutoConfig.from_pretrained(args.init_model_path, trust_remote_code=True)
    
    # Get the underlying model
    model = qwen3tts.model
    
    # Enable gradient checkpointing if requested
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
        accelerator.print("Gradient checkpointing enabled")
    
    registry = get_language_registry_from_config(config)
    normalized_target_language = None
    language_id = None

    # Keep optional language-row hook alive for entire training lifecycle.
    language_row_hook = None

    # Apply training mode
    if args.train_mode == 'lora':
        accelerator.print("Setting up LoRA...")
        lora_config = {
            'r': args.lora_r,
            'lora_alpha': args.lora_alpha,
            'lora_dropout': args.lora_dropout,
        }
        model = setup_lora_model(model, lora_config)
    
    elif args.train_mode == 'lang_only':
        accelerator.print("Setting up lang_only mode (freezing most parameters)...")
        normalized_target_language, language_id = validate_lang_only_target_language(config, args.target_language)
        accelerator.print(
            f"Using registered language token '{normalized_target_language}' (language_id={language_id})"
        )
        language_row_hook = freeze_all_except_language_embeddings(model, language_id)
    
    else:  # full
        accelerator.print("Using full fine-tuning mode")
    
    # Load data
    accelerator.print(f"Loading training data from {args.train_jsonl}...")
    with open(args.train_jsonl, 'r') as f:
        # Filter out empty lines before parsing JSON
        train_data = [json.loads(line) for line in f if line.strip()]
    
    accelerator.print(f"Loaded {len(train_data)} training samples")

    unregistered_languages = find_unregistered_training_languages(
        train_data,
        registry,
        normalized_target_language or args.target_language,
    )
    if unregistered_languages:
        run_tokenizer_preflight_audit(
            qwen3tts.processor,
            train_data,
            unregistered_languages,
            args.output_dir,
        )
    
    # Create dataset and dataloader
    dataset = LanguageDataset(
        train_data,
        qwen3tts.processor,
        config,
        target_language=args.target_language
    )
    
    train_dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=dataset.collate_fn
    )
    
    # Setup optimizer
    optimizer = create_optimizer(
        model,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        language_id=language_id if args.train_mode == 'lang_only' else None,
    )
    
    # Setup learning rate scheduler
    num_training_steps = len(train_dataloader) * args.num_epochs // args.gradient_accumulation_steps
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Prepare everything with accelerator
    model, optimizer, train_dataloader, scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, scheduler
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    start_step = 0
    if args.resume_from_checkpoint:
        # load_checkpoint returns the 0-indexed epoch to start from
        start_epoch, start_step = load_checkpoint(
            args.resume_from_checkpoint,
            model,
            optimizer,
            scheduler
        )
    
    # Training loop
    accelerator.print("\n" + "="*60)
    accelerator.print("Starting training")
    accelerator.print("="*60)
    accelerator.print(f"  Number of epochs: {args.num_epochs}")
    accelerator.print(f"  Batch size per device: {args.batch_size}")
    accelerator.print(f"  Gradient accumulation steps: {args.gradient_accumulation_steps}")
    accelerator.print(f"  Total batch size: {args.batch_size * args.gradient_accumulation_steps * accelerator.num_processes}")
    accelerator.print(f"  Training mode: {args.train_mode}")
    accelerator.print(f"  Target language: {args.target_language or 'auto/unknown'}")
    accelerator.print(f"  Learning rate: {args.learning_rate}")
    accelerator.print(f"  Mixed precision: {args.mixed_precision}")
    accelerator.print("="*60 + "\n")
    
    model.train()
    global_step = start_step
    
    for epoch in range(start_epoch, args.num_epochs):
        accelerator.print(f"\nEpoch {epoch + 1}/{args.num_epochs}")
        epoch_loss = 0
        num_batches = 0
        
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(model):
                loss = forward_training_batch(model, batch)
                
                # Backward pass
                accelerator.backward(loss)
                
                # Gradient clipping
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                
                # Optimizer step
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
                epoch_loss += loss.item()
                num_batches += 1
            
            # Logging
            if (step + 1) % args.logging_steps == 0:
                avg_loss = epoch_loss / num_batches
                current_lr = scheduler.get_last_lr()[0]
                
                accelerator.print(
                    f"Epoch {epoch + 1} | Step {step + 1}/{len(train_dataloader)} | "
                    f"Loss: {loss.item():.4f} | Avg Loss: {avg_loss:.4f} | "
                    f"LR: {current_lr:.2e}"
                )
                
                # Log to tracking service
                accelerator.log({
                    "train/loss": loss.item(),
                    "train/avg_loss": avg_loss,
                    "train/learning_rate": current_lr,
                    "train/epoch": epoch + 1,
                }, step=global_step)
            
            global_step += 1
            
            # Save checkpoint at intervals if specified
            if args.save_steps and (step + 1) % args.save_steps == 0:
                # Save with 1-indexed epoch number for clarity
                save_checkpoint(
                    accelerator,
                    model,
                    optimizer,
                    scheduler,
                    epoch + 1,  # Save as 1-indexed
                    step + 1,
                    args.output_dir,
                    args.init_model_path,
                    args
                )
        
        # Save checkpoint at end of epoch
        avg_epoch_loss = epoch_loss / num_batches
        accelerator.print(f"Epoch {epoch + 1} completed | Average loss: {avg_epoch_loss:.4f}")
        
        # Save with 1-indexed epoch number for clarity (epoch 0 becomes checkpoint-epoch-1)
        save_checkpoint(
            accelerator,
            model,
            optimizer,
            scheduler,
            epoch + 1,  # Save as 1-indexed
            global_step,
            args.output_dir,
            args.init_model_path,
            args
        )
    
    # End tracking
    accelerator.end_training()
    
    accelerator.print("\n" + "="*60)
    accelerator.print("Training completed!")
    accelerator.print(f"Checkpoints saved to: {args.output_dir}")
    accelerator.print("="*60)


if __name__ == "__main__":
    train()
