# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for the language registration script.

Tests language token registration functionality including:
- Config updates
- Embedding resizing
- Round-trip load/save
- Safe ID allocation
- Error handling
"""

import json
import os
import tempfile
import shutil
from pathlib import Path
from unittest import mock

import pytest
import torch

from qwen_tts.core.models import Qwen3TTSConfig, Qwen3TTSForConditionalGeneration
from scripts.register_language import (
    find_next_available_language_id,
    check_embedding_resize_needed,
    resize_embeddings,
    register_language,
)


class TestFindNextAvailableLanguageId:
    """Test cases for finding next available language ID."""
    
    def test_with_existing_language_ids(self):
        """Test ID allocation with existing language IDs."""
        config = Qwen3TTSConfig()
        config.talker_config.codec_language_id = {
            "chinese": 4200,
            "english": 4201,
            "japanese": 4202,
        }
        
        # Should return next available ID (after special tokens which go up to 4205)
        next_id = find_next_available_language_id(config)
        assert next_id == 4206
    
    def test_with_special_tokens(self):
        """Test ID allocation considers special tokens."""
        config = Qwen3TTSConfig()
        config.talker_config.codec_language_id = {}
        config.talker_config.codec_pad_id = 4196
        config.talker_config.codec_bos_id = 4197
        config.talker_config.codec_eos_token_id = 4198
        config.talker_config.codec_think_id = 4202
        config.talker_config.codec_nothink_id = 4203
        config.talker_config.codec_think_bos_id = 4204
        config.talker_config.codec_think_eos_id = 4205
        
        # Should return ID after highest special token
        next_id = find_next_available_language_id(config)
        assert next_id == 4206
    
    def test_with_mixed_ids(self):
        """Test ID allocation with both language and special token IDs."""
        config = Qwen3TTSConfig()
        config.talker_config.codec_language_id = {
            "chinese": 4200,
            "english": 4201,
        }
        config.talker_config.codec_think_id = 4210
        config.talker_config.codec_nothink_id = 4211
        
        # Should return ID after highest used ID
        next_id = find_next_available_language_id(config)
        assert next_id == 4212
    
    def test_with_empty_config(self):
        """Test ID allocation with no existing IDs."""
        config = Qwen3TTSConfig()
        config.talker_config.codec_language_id = {}
        
        # Should return default fallback
        next_id = find_next_available_language_id(config)
        assert next_id == 4206
    
    def test_with_none_language_id(self):
        """Test ID allocation when codec_language_id is None."""
        config = Qwen3TTSConfig()
        config.talker_config.codec_language_id = None
        
        # Should handle None gracefully
        next_id = find_next_available_language_id(config)
        assert next_id >= 4206


class TestCheckEmbeddingResizeNeeded:
    """Test cases for checking if embedding resize is needed."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create minimal model with small vocab for testing
        self.config = Qwen3TTSConfig()
        self.config.talker_config.vocab_size = 100
    
    def test_resize_needed_when_token_exceeds_vocab(self):
        """Test that resize is needed when token ID >= vocab size."""
        # Mock a model with small embedding
        model = mock.MagicMock()
        model.talker.model.codec_embedding.num_embeddings = 100
        
        # Token ID at or above vocab size should require resize
        assert check_embedding_resize_needed(model, 100) is True
        assert check_embedding_resize_needed(model, 101) is True
        assert check_embedding_resize_needed(model, 150) is True
    
    def test_resize_not_needed_when_token_within_vocab(self):
        """Test that resize is not needed when token ID < vocab size."""
        model = mock.MagicMock()
        model.talker.model.codec_embedding.num_embeddings = 100
        
        # Token ID below vocab size should not require resize
        assert check_embedding_resize_needed(model, 50) is False
        assert check_embedding_resize_needed(model, 99) is False
    
    def test_resize_at_boundary(self):
        """Test resize check at vocab size boundary."""
        model = mock.MagicMock()
        model.talker.model.codec_embedding.num_embeddings = 4206
        
        # Exactly at vocab size should require resize
        assert check_embedding_resize_needed(model, 4206) is True
        # Just below should not
        assert check_embedding_resize_needed(model, 4205) is False


class TestResizeEmbeddings:
    """Test cases for resizing embedding layers."""
    
    def test_resize_preserves_old_embeddings(self):
        """Test that resizing preserves existing embedding weights."""
        config = Qwen3TTSConfig()
        config.talker_config.vocab_size = 100
        config.talker_config.hidden_size = 512
        config.talker_config.initializer_range = 0.02
        
        # Create a minimal model structure
        model = mock.MagicMock()
        old_embedding = torch.nn.Embedding(100, 512)
        
        # Set some distinctive weights
        with torch.no_grad():
            old_embedding.weight[50] = torch.ones(512) * 0.5
        
        model.talker.model.codec_embedding = old_embedding
        model.config = config
        
        # Resize to 150
        resize_embeddings(model, 150)
        
        # Check that old embedding was preserved
        new_embedding = model.talker.model.codec_embedding
        assert new_embedding.num_embeddings == 150
        assert torch.allclose(new_embedding.weight[50], torch.ones(512) * 0.5)
    
    def test_resize_initializes_new_embeddings(self):
        """Test that new embeddings are properly initialized."""
        config = Qwen3TTSConfig()
        config.talker_config.vocab_size = 100
        config.talker_config.hidden_size = 512
        config.talker_config.initializer_range = 0.02
        
        model = mock.MagicMock()
        old_embedding = torch.nn.Embedding(100, 512)
        model.talker.model.codec_embedding = old_embedding
        model.config = config
        
        # Resize to 150
        resize_embeddings(model, 150)
        
        # Check that new embeddings exist
        new_embedding = model.talker.model.codec_embedding
        assert new_embedding.num_embeddings == 150
        
        # Check that new embeddings are initialized (not zero, not huge)
        new_weights = new_embedding.weight[100:150]
        assert not torch.allclose(new_weights, torch.zeros_like(new_weights))
        assert torch.abs(new_weights).mean() < 1.0  # Reasonable magnitude
    
    def test_resize_updates_config(self):
        """Test that config vocab size is updated after resize."""
        config = Qwen3TTSConfig()
        config.talker_config.vocab_size = 100
        config.talker_config.hidden_size = 512
        config.talker_config.initializer_range = 0.02
        
        model = mock.MagicMock()
        old_embedding = torch.nn.Embedding(100, 512)
        model.talker.model.codec_embedding = old_embedding
        model.config = config
        
        resize_embeddings(model, 150)
        
        assert model.config.talker_config.vocab_size == 150


class TestRegisterLanguageIntegration:
    """Integration tests for the full registration workflow."""
    
    def setup_method(self):
        """Set up test fixtures with temporary directories."""
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, "test_model")
        self.output_path = os.path.join(self.temp_dir, "test_model_updated")
        
        # Create a minimal test model
        self._create_minimal_model()
    
    def teardown_method(self):
        """Clean up temporary directories."""
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
    
    def _create_minimal_model(self):
        """Create a minimal model for testing."""
        config = Qwen3TTSConfig()
        config.talker_config.codec_language_id = {
            "chinese": 4200,
            "english": 4201,
        }
        config.talker_config.vocab_size = 4202  # Just enough for existing tokens
        
        # Save config
        os.makedirs(self.model_path, exist_ok=True)
        config.save_pretrained(self.model_path)
    
    def test_register_new_language(self):
        """Test registering a new language."""
        # Mock both the config and model loading
        with mock.patch('scripts.register_language.AutoConfig.from_pretrained') as mock_config_load, \
             mock.patch('scripts.register_language.Qwen3TTSForConditionalGeneration.from_pretrained') as mock_from_pretrained:
            
            # Create mock config
            mock_config = Qwen3TTSConfig()
            mock_config.talker_config.codec_language_id = {
                "chinese": 4200,
                "english": 4201,
            }
            mock_config.talker_config.vocab_size = 4202
            mock_config.talker_config.initializer_range = 0.02
            mock_config.talker_config.hidden_size = 512
            mock_config_load.return_value = mock_config
            
            # Create mock model
            mock_model = mock.MagicMock()
            mock_embedding = torch.nn.Embedding(4202, 512)
            mock_model.talker.model.codec_embedding = mock_embedding
            mock_model.config = mock_config
            mock_model.supported_languages = ["chinese", "english"]
            
            mock_from_pretrained.return_value = mock_model
            
            # Register polish
            result = register_language(
                model_path=self.model_path,
                language="Polish",  # Test case normalization
                output_path=self.output_path,
            )
            
            # Check result - should use next ID after special tokens (4205)
            assert result["language"] == "polish"
            assert result["language_id"] == 4206  # After all special tokens
            assert result["resized"] is True
            assert result["new_vocab_size"] == 4207
            
            # Verify model was saved
            mock_model.save_pretrained.assert_called_once_with(self.output_path)
    
    def test_register_with_explicit_id(self):
        """Test registering with explicit language ID."""
        with mock.patch('scripts.register_language.AutoConfig.from_pretrained') as mock_config_load, \
             mock.patch('scripts.register_language.Qwen3TTSForConditionalGeneration.from_pretrained') as mock_from_pretrained:
            
            mock_config = Qwen3TTSConfig()
            mock_config.talker_config.codec_language_id = {
                "chinese": 4200,
                "english": 4201,
            }
            mock_config.talker_config.vocab_size = 4202
            mock_config.talker_config.initializer_range = 0.02
            mock_config.talker_config.hidden_size = 512
            mock_config_load.return_value = mock_config
            
            mock_model = mock.MagicMock()
            mock_embedding = torch.nn.Embedding(4202, 512)
            mock_model.talker.model.codec_embedding = mock_embedding
            mock_model.config = mock_config
            mock_model.supported_languages = ["chinese", "english"]
            
            mock_from_pretrained.return_value = mock_model
            
            # Register with explicit ID
            result = register_language(
                model_path=self.model_path,
                language="french",
                output_path=self.output_path,
                language_id=4210,
            )
            
            assert result["language"] == "french"
            assert result["language_id"] == 4210
            assert result["resized"] is True
    
    def test_register_existing_language_without_force(self):
        """Test that registering existing language fails without --force."""
        with mock.patch('scripts.register_language.AutoConfig.from_pretrained') as mock_config:
            config = Qwen3TTSConfig()
            config.talker_config.codec_language_id = {
                "chinese": 4200,
                "polish": 4201,  # Already exists
            }
            mock_config.return_value = config
            
            # Should raise ValueError
            with pytest.raises(ValueError, match="already registered"):
                register_language(
                    model_path=self.model_path,
                    language="polish",
                    output_path=self.output_path,
                )
    
    def test_register_existing_language_with_force(self):
        """Test that registering existing language succeeds with --force."""
        with mock.patch('scripts.register_language.AutoConfig.from_pretrained') as mock_config_load, \
             mock.patch('scripts.register_language.Qwen3TTSForConditionalGeneration.from_pretrained') as mock_from_pretrained:
            
            mock_config = Qwen3TTSConfig()
            mock_config.talker_config.codec_language_id = {
                "chinese": 4200,
                "polish": 4201,  # Already exists
            }
            mock_config.talker_config.vocab_size = 4202
            mock_config.talker_config.initializer_range = 0.02
            mock_config.talker_config.hidden_size = 512
            mock_config_load.return_value = mock_config
            
            mock_model = mock.MagicMock()
            mock_embedding = torch.nn.Embedding(4202, 512)
            mock_model.talker.model.codec_embedding = mock_embedding
            mock_model.config = mock_config
            mock_model.supported_languages = ["chinese", "polish"]
            
            mock_from_pretrained.return_value = mock_model
            
            # Should succeed with force=True
            result = register_language(
                model_path=self.model_path,
                language="polish",
                output_path=self.output_path,
                force=True,
            )
            
            assert result["language"] == "polish"
    
    def test_no_resize_when_vocab_sufficient(self):
        """Test that no resize happens when vocab is already large enough."""
        with mock.patch('scripts.register_language.AutoConfig.from_pretrained') as mock_config_load, \
             mock.patch('scripts.register_language.Qwen3TTSForConditionalGeneration.from_pretrained') as mock_from_pretrained:
            
            mock_config = Qwen3TTSConfig()
            mock_config.talker_config.codec_language_id = {
                "chinese": 4200,
                "english": 4201,
            }
            mock_config.talker_config.vocab_size = 5000  # Large enough
            mock_config.talker_config.initializer_range = 0.02
            mock_config.talker_config.hidden_size = 512
            mock_config_load.return_value = mock_config
            
            mock_model = mock.MagicMock()
            mock_embedding = torch.nn.Embedding(5000, 512)
            mock_model.talker.model.codec_embedding = mock_embedding
            mock_model.config = mock_config
            mock_model.supported_languages = ["chinese", "english"]
            
            mock_from_pretrained.return_value = mock_model
            
            # Register polish (will get ID 4206 after special tokens)
            result = register_language(
                model_path=self.model_path,
                language="polish",
                output_path=self.output_path,
            )
            
            # Should not need resize
            assert result["resized"] is False


class TestLanguageRegistryIntegration:
    """Test that registered languages work with the language registry."""
    
    def test_registered_language_resolves_correctly(self):
        """Test that a newly registered language resolves to correct ID."""
        from qwen_tts.langs.registry import LanguageRegistry
        
        # Create registry with new language
        codec_language_id = {
            "chinese": 4200,
            "english": 4201,
            "polish": 4202,  # Newly registered
        }
        
        registry = LanguageRegistry(codec_language_id)
        
        # Test resolution
        assert registry.resolve_language("polish") == 4202
        assert registry.resolve_language("Polish") == 4202
        assert registry.resolve_language("pl") == 4202
        assert registry.resolve_language("pl-PL") == 4202
    
    def test_registered_language_no_longer_fallback(self):
        """Test that registered language no longer returns None."""
        from qwen_tts.langs.registry import LanguageRegistry
        
        # Before registration
        registry_before = LanguageRegistry({
            "chinese": 4200,
            "english": 4201,
        })
        assert registry_before.resolve_language("polish") is None
        
        # After registration
        registry_after = LanguageRegistry({
            "chinese": 4200,
            "english": 4201,
            "polish": 4202,
        })
        assert registry_after.resolve_language("polish") == 4202
