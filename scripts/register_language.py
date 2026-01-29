#!/usr/bin/env python
# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0

"""
Register a new language token in a Qwen3-TTS model.

This script allows users to add a new language to a TTS model checkpoint,
enabling explicit language conditioning during inference. It:

1. Loads a base checkpoint and config
2. Allocates a new language_id token (safe allocation strategy)
3. Adds the language mapping to config.talker_config.codec_language_id
4. Ensures model embedding layers support the new token (resize if needed)
5. Saves an updated checkpoint with new weights and config

Usage:
    python scripts/register_language.py \\
        --model_path /path/to/model \\
        --language polish \\
        --output_path /path/to/updated_model

After running this script, inference with language="polish" will resolve
to a real language_id and not fallback to "auto".
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Optional

import torch
from transformers import AutoConfig

from qwen_tts.core.models import Qwen3TTSForConditionalGeneration, Qwen3TTSConfig


def find_next_available_language_id(config: Qwen3TTSConfig) -> int:
    """
    Find the next available language ID token.
    
    Strategy: Look for the maximum ID in the reserved codec token range
    and return the next one. The codec uses special tokens in the 4196-4205
    range, so we start looking after those.
    
    Args:
        config: The model configuration
        
    Returns:
        Next available language ID (integer)
    """
    # Collect all used IDs from special tokens
    used_ids = set()
    
    # Add special codec tokens
    if hasattr(config.talker_config, 'codec_pad_id') and config.talker_config.codec_pad_id is not None:
        used_ids.add(config.talker_config.codec_pad_id)
    if hasattr(config.talker_config, 'codec_bos_id') and config.talker_config.codec_bos_id is not None:
        used_ids.add(config.talker_config.codec_bos_id)
    if hasattr(config.talker_config, 'codec_eos_token_id') and config.talker_config.codec_eos_token_id is not None:
        used_ids.add(config.talker_config.codec_eos_token_id)
    if hasattr(config.talker_config, 'codec_think_id') and config.talker_config.codec_think_id is not None:
        used_ids.add(config.talker_config.codec_think_id)
    if hasattr(config.talker_config, 'codec_nothink_id') and config.talker_config.codec_nothink_id is not None:
        used_ids.add(config.talker_config.codec_nothink_id)
    if hasattr(config.talker_config, 'codec_think_bos_id') and config.talker_config.codec_think_bos_id is not None:
        used_ids.add(config.talker_config.codec_think_bos_id)
    if hasattr(config.talker_config, 'codec_think_eos_id') and config.talker_config.codec_think_eos_id is not None:
        used_ids.add(config.talker_config.codec_think_eos_id)
    
    # Add existing language IDs
    if config.talker_config.codec_language_id:
        used_ids.update(config.talker_config.codec_language_id.values())
    
    # Find the max ID in the codec range and return next available
    if used_ids:
        max_id = max(used_ids)
        return max_id + 1
    else:
        # Default fallback if no IDs are used (unlikely)
        return 4206


def check_embedding_resize_needed(model: Qwen3TTSForConditionalGeneration, new_token_id: int) -> bool:
    """
    Check if embedding layers need to be resized to accommodate the new token ID.
    
    Args:
        model: The TTS model
        new_token_id: The new language token ID
        
    Returns:
        True if resize is needed, False otherwise
    """
    # Check codec embedding size
    codec_embedding = model.talker.model.codec_embedding
    current_vocab_size = codec_embedding.num_embeddings
    
    # Need resize if new_token_id >= current vocab size
    return new_token_id >= current_vocab_size


def resize_embeddings(model: Qwen3TTSForConditionalGeneration, new_vocab_size: int) -> None:
    """
    Resize the codec embedding layer to accommodate new tokens.
    
    This creates a new embedding layer with the expanded size and copies
    over the existing embeddings, initializing new embeddings with small
    random values.
    
    Args:
        model: The TTS model
        new_vocab_size: The new vocabulary size
    """
    old_embedding = model.talker.model.codec_embedding
    old_vocab_size = old_embedding.num_embeddings
    embedding_dim = old_embedding.embedding_dim
    
    # Create new embedding layer
    new_embedding = torch.nn.Embedding(new_vocab_size, embedding_dim, padding_idx=old_embedding.padding_idx)
    
    # Copy old weights
    with torch.no_grad():
        new_embedding.weight[:old_vocab_size] = old_embedding.weight.data
        
        # Initialize new embeddings with small random values (matching standard initialization)
        if new_vocab_size > old_vocab_size:
            torch.nn.init.normal_(
                new_embedding.weight[old_vocab_size:],
                mean=0.0,
                std=model.config.talker_config.initializer_range
            )
    
    # Replace old embedding
    model.talker.model.codec_embedding = new_embedding
    
    # Update config vocab size
    model.config.talker_config.vocab_size = new_vocab_size


def register_language(
    model_path: str,
    language: str,
    output_path: str,
    language_id: Optional[int] = None,
    force: bool = False
) -> Dict[str, any]:
    """
    Register a new language in a TTS model checkpoint.
    
    Args:
        model_path: Path to the base model checkpoint
        language: Language name to register (e.g., "polish")
        output_path: Path where updated model will be saved
        language_id: Optional explicit language ID to use (auto-allocated if None)
        force: If True, overwrite existing language registration
        
    Returns:
        Dictionary with registration details (language, language_id, resized, etc.)
    """
    print(f"Loading model from {model_path}...")
    
    # Load config
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    
    # Normalize language name (lowercase)
    language_normalized = language.lower().strip()
    
    # Check if language already exists
    if config.talker_config.codec_language_id is None:
        config.talker_config.codec_language_id = {}
    
    if language_normalized in config.talker_config.codec_language_id:
        if not force:
            raise ValueError(
                f"Language '{language_normalized}' already registered with ID "
                f"{config.talker_config.codec_language_id[language_normalized]}. "
                f"Use --force to overwrite."
            )
        print(f"Warning: Overwriting existing language '{language_normalized}'")
    
    # Determine language ID
    if language_id is None:
        language_id = find_next_available_language_id(config)
    
    print(f"Registering language '{language_normalized}' with ID {language_id}")
    
    # Load model (on CPU to save memory, can be moved to GPU if needed)
    print("Loading model weights...")
    model = Qwen3TTSForConditionalGeneration.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
    )
    
    # Check if embedding resize is needed
    resize_needed = check_embedding_resize_needed(model, language_id)
    
    if resize_needed:
        new_vocab_size = language_id + 1
        print(f"Resizing codec embeddings from {model.config.talker_config.vocab_size} to {new_vocab_size}")
        resize_embeddings(model, new_vocab_size)
    else:
        print("No embedding resize needed")
    
    # Update config with new language
    config.talker_config.codec_language_id[language_normalized] = language_id
    model.config.talker_config.codec_language_id[language_normalized] = language_id
    
    # Update supported languages in model
    model.supported_languages.append(language_normalized)
    
    # Refresh language registry
    from qwen_tts.langs.registry import LanguageRegistry
    model.language_registry = LanguageRegistry(model.config.talker_config.codec_language_id)
    
    # Save updated model
    print(f"Saving updated model to {output_path}...")
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)
    
    print("\nRegistration complete!")
    print(f"  Language: {language_normalized}")
    print(f"  Language ID: {language_id}")
    print(f"  Embedding resized: {resize_needed}")
    print(f"  Model saved to: {output_path}")
    
    return {
        "language": language_normalized,
        "language_id": language_id,
        "resized": resize_needed,
        "old_vocab_size": model.config.talker_config.vocab_size - (1 if resize_needed else 0),
        "new_vocab_size": model.config.talker_config.vocab_size,
        "output_path": output_path,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Register a new language token in a Qwen3-TTS model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Register Polish language with auto-allocated ID
  python scripts/register_language.py \\
      --model_path ./qwen3-tts-base \\
      --language polish \\
      --output_path ./qwen3-tts-base-polish

  # Register with specific language ID
  python scripts/register_language.py \\
      --model_path ./qwen3-tts-base \\
      --language polish \\
      --language_id 4210 \\
      --output_path ./qwen3-tts-base-polish

  # Overwrite existing language registration
  python scripts/register_language.py \\
      --model_path ./qwen3-tts-base-polish \\
      --language polish \\
      --language_id 4211 \\
      --output_path ./qwen3-tts-base-polish-updated \\
      --force
        """
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the base model checkpoint (local directory or HuggingFace repo)"
    )
    parser.add_argument(
        "--language",
        type=str,
        required=True,
        help="Language name to register (e.g., 'polish', 'french'). Will be normalized to lowercase."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path where the updated model will be saved"
    )
    parser.add_argument(
        "--language_id",
        type=int,
        default=None,
        help="Explicit language ID to use (optional, auto-allocated if not specified)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force overwrite if language already exists"
    )
    
    args = parser.parse_args()
    
    try:
        result = register_language(
            model_path=args.model_path,
            language=args.language,
            output_path=args.output_path,
            language_id=args.language_id,
            force=args.force
        )
        
        print("\n" + "="*60)
        print("SUCCESS: Language registration completed")
        print("="*60)
        
    except Exception as e:
        print("\n" + "="*60)
        print("ERROR: Language registration failed")
        print("="*60)
        print(f"\n{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


if __name__ == "__main__":
    main()
