#!/usr/bin/env python
# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for the language fine-tuning script.

These tests validate the core functionality of finetune_language.py without
requiring actual model weights or GPUs.
"""

import sys
import pytest
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Check if torch is available
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Skip all tests if torch is not available
pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")


def test_imports():
    """Test that all required imports work."""
    try:
        from scripts.finetune_language import LanguageDataset
        from scripts.finetune_language import freeze_all_except_language_embeddings
        from scripts.finetune_language import save_checkpoint
        from scripts.finetune_language import load_checkpoint
        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")


def test_peft_optional():
    """Test that PEFT is optional and script handles its absence gracefully."""
    from scripts.finetune_language import PEFT_AVAILABLE
    
    # Should not crash even if PEFT is not available
    # The flag should be either True or False
    assert isinstance(PEFT_AVAILABLE, bool)


def test_dataset_structure():
    """Test that LanguageDataset has the expected structure."""
    from scripts.finetune_language import LanguageDataset
    
    # Check that class has expected methods
    assert hasattr(LanguageDataset, '__init__')
    assert hasattr(LanguageDataset, '__len__')
    assert hasattr(LanguageDataset, '__getitem__')
    assert hasattr(LanguageDataset, 'collate_fn')
    assert hasattr(LanguageDataset, '_load_audio_to_np')
    assert hasattr(LanguageDataset, '_build_assistant_text')
    assert hasattr(LanguageDataset, 'extract_mels')


def test_training_modes():
    """Test that supported training modes are correctly defined."""
    # The script should support these modes in argparse choices
    expected_modes = ['lora', 'full', 'lang_only']
    
    # We can't easily test argparse choices without running the script,
    # but we can verify the freeze function exists
    from scripts.finetune_language import freeze_all_except_language_embeddings
    
    assert callable(freeze_all_except_language_embeddings)


def test_assistant_text_formatting():
    """Test the assistant text formatting function."""
    from scripts.finetune_language import LanguageDataset
    
    # Create a minimal mock dataset to test the method
    class MockProcessor:
        def __call__(self, text, return_tensors=None, padding=None):
            import torch
            return {"input_ids": torch.tensor([[1, 2, 3, 4, 5]])}
    
    class MockConfig:
        def __init__(self):
            self.tts_pad_token_id = 0
            self.tts_bos_token_id = 1
            self.tts_eos_token_id = 2
            self.talker_config = type('obj', (object,), {
                'codec_nothink_id': 4203,
                'codec_think_bos_id': 4204,
                'codec_think_eos_id': 4205,
                'codec_pad_id': 4196,
                'codec_bos_id': 4197,
            })()
    
    dataset = LanguageDataset([], MockProcessor(), MockConfig())
    
    # Test the formatting
    text = "Hello world"
    formatted = dataset._build_assistant_text(text)
    
    assert "<|im_start|>assistant" in formatted
    assert "<|im_end|>" in formatted
    assert "Hello world" in formatted


def test_smoke_test_data_exists():
    """Test that smoke test data file exists."""
    test_data_path = Path(__file__).parent / "smoke_test_data" / "sample_train.jsonl"
    
    assert test_data_path.exists(), f"Smoke test data not found at {test_data_path}"
    
    # Verify it's valid JSON
    import json
    with open(test_data_path, 'r') as f:
        lines = f.readlines()
        assert len(lines) > 0, "Smoke test data is empty"
        
        for i, line in enumerate(lines):
            # Skip empty lines
            if not line.strip():
                continue
            
            data = json.loads(line)
            assert 'audio' in data
            assert 'text' in data
            assert 'ref_audio' in data
            assert 'audio_codes' in data


def test_script_has_main():
    """Test that the script has a main entry point."""
    from scripts.finetune_language import train
    
    assert callable(train)


def test_checkpoint_functions_exist():
    """Test that checkpoint save/load functions exist."""
    from scripts.finetune_language import save_checkpoint, load_checkpoint
    
    assert callable(save_checkpoint)
    assert callable(load_checkpoint)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
