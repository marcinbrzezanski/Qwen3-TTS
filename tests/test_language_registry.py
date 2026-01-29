# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for the language registry module.

Tests language normalization, resolution, and fallback behavior.
"""

import pytest
from qwen_tts.langs.registry import LanguageRegistry, normalize_language, resolve_language


class TestLanguageRegistry:
    """Test cases for LanguageRegistry class."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Mock codec_language_id similar to what would be in config
        self.codec_language_id = {
            "chinese": 4200,
            "english": 4201,
            "japanese": 4202,
        }
        self.registry = LanguageRegistry(self.codec_language_id)
    
    def test_normalize_language_lowercase(self):
        """Test normalization converts to lowercase."""
        assert self.registry.normalize_language("English") == "english"
        assert self.registry.normalize_language("CHINESE") == "chinese"
        assert self.registry.normalize_language("Japanese") == "japanese"
    
    def test_normalize_language_codes(self):
        """Test normalization of ISO language codes."""
        # English variants
        assert self.registry.normalize_language("en") == "english"
        assert self.registry.normalize_language("en-US") == "english"
        assert self.registry.normalize_language("en-GB") == "english"
        
        # Chinese variants
        assert self.registry.normalize_language("zh") == "chinese"
        assert self.registry.normalize_language("zh-CN") == "chinese"
        assert self.registry.normalize_language("zh-TW") == "chinese"
        assert self.registry.normalize_language("cmn") == "chinese"
        assert self.registry.normalize_language("Mandarin") == "chinese"
        
        # Japanese variants
        assert self.registry.normalize_language("ja") == "japanese"
        assert self.registry.normalize_language("jp") == "japanese"
        assert self.registry.normalize_language("jpn") == "japanese"
    
    def test_normalize_auto(self):
        """Test normalization of 'auto' and variations."""
        assert self.registry.normalize_language("auto") == "auto"
        assert self.registry.normalize_language("Auto") == "auto"
        assert self.registry.normalize_language("AUTO") == "auto"
        assert self.registry.normalize_language("unknown") == "auto"
    
    def test_normalize_unsupported(self):
        """Test normalization of unsupported languages."""
        # Should return lowercase version
        assert self.registry.normalize_language("Polish") == "polish"
        assert self.registry.normalize_language("French") == "french"
        assert self.registry.normalize_language("unknown_lang") == "unknown_lang"
    
    def test_normalize_polish_variants(self):
        """Test normalization of Polish language variants."""
        assert self.registry.normalize_language("Polish") == "polish"
        assert self.registry.normalize_language("pl") == "polish"
        assert self.registry.normalize_language("pl-PL") == "polish"
        assert self.registry.normalize_language("pol") == "polish"
    
    def test_normalize_empty_string(self):
        """Test normalization of empty string."""
        assert self.registry.normalize_language("") == "auto"
        assert self.registry.normalize_language("  ") == "auto"
    
    def test_resolve_language_supported(self):
        """Test resolution of supported languages."""
        assert self.registry.resolve_language("chinese") == 4200
        assert self.registry.resolve_language("Chinese") == 4200
        assert self.registry.resolve_language("CHINESE") == 4200
        
        assert self.registry.resolve_language("english") == 4201
        assert self.registry.resolve_language("English") == 4201
        
        assert self.registry.resolve_language("japanese") == 4202
    
    def test_resolve_language_codes(self):
        """Test resolution using ISO codes."""
        assert self.registry.resolve_language("en") == 4201
        assert self.registry.resolve_language("zh") == 4200
        assert self.registry.resolve_language("ja") == 4202
    
    def test_resolve_language_auto(self):
        """Test resolution of 'auto' returns None."""
        assert self.registry.resolve_language("auto") is None
        assert self.registry.resolve_language("Auto") is None
        assert self.registry.resolve_language("AUTO") is None
    
    def test_resolve_language_unsupported_non_strict(self):
        """Test resolution of unsupported language in non-strict mode."""
        # Should fallback to None (auto behavior)
        assert self.registry.resolve_language("polish", strict=False) is None
        assert self.registry.resolve_language("french", strict=False) is None
        assert self.registry.resolve_language("unknown_language", strict=False) is None
    
    def test_resolve_language_unsupported_strict(self):
        """Test resolution of unsupported language in strict mode."""
        # Should raise NotImplementedError
        with pytest.raises(NotImplementedError, match="Language 'polish'.*not implemented"):
            self.registry.resolve_language("polish", strict=True)
        
        with pytest.raises(NotImplementedError, match="Language 'french'.*not implemented"):
            self.registry.resolve_language("french", strict=True)
        
        with pytest.raises(NotImplementedError, match="Language 'unknown_lang'.*not implemented"):
            self.registry.resolve_language("unknown_lang", strict=True)
    
    def test_resolve_language_polish_variants(self):
        """Test resolution of Polish variants in different modes."""
        # Non-strict: should fallback to None
        assert self.registry.resolve_language("Polish", strict=False) is None
        assert self.registry.resolve_language("pl", strict=False) is None
        assert self.registry.resolve_language("pl-PL", strict=False) is None
        
        # Strict: should raise error
        with pytest.raises(NotImplementedError):
            self.registry.resolve_language("Polish", strict=True)


class TestStandaloneFunctions:
    """Test cases for standalone convenience functions."""
    
    def test_normalize_language_function(self):
        """Test standalone normalize_language function."""
        assert normalize_language("English") == "english"
        assert normalize_language("en") == "english"
        assert normalize_language("Polish") == "polish"
    
    def test_resolve_language_function(self):
        """Test standalone resolve_language function."""
        codec_language_id = {
            "chinese": 4200,
            "english": 4201,
        }
        
        assert resolve_language("english", codec_language_id) == 4201
        assert resolve_language("en", codec_language_id) == 4201
        assert resolve_language("auto", codec_language_id) is None
        
        # Non-strict: unknown should return None
        assert resolve_language("polish", codec_language_id, strict=False) is None
        
        # Strict: unknown should raise
        with pytest.raises(NotImplementedError):
            resolve_language("polish", codec_language_id, strict=True)


class TestLanguageRegistryEdgeCases:
    """Test edge cases and special scenarios."""
    
    def test_empty_codec_language_id(self):
        """Test registry with no languages."""
        registry = LanguageRegistry({})
        
        # Everything should fallback to None in non-strict mode
        assert registry.resolve_language("english", strict=False) is None
        assert registry.resolve_language("chinese", strict=False) is None
        
        # Auto should still work
        assert registry.resolve_language("auto") is None
    
    def test_none_codec_language_id(self):
        """Test registry with None codec_language_id."""
        registry = LanguageRegistry(None)
        
        # Should initialize with empty dict
        assert registry.resolve_language("english", strict=False) is None
        assert registry.resolve_language("auto") is None
    
    def test_normalization_consistency(self):
        """Test that normalization is consistent across different inputs."""
        registry = LanguageRegistry({"english": 4201})
        
        # All these should resolve to the same ID
        inputs = ["English", "ENGLISH", "english", "en", "EN", "en-US", "en-gb"]
        for inp in inputs:
            assert registry.resolve_language(inp) == 4201
