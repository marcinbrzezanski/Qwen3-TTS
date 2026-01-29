# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0

"""
Language registry for Qwen3-TTS.

Provides normalization and resolution of language strings to language IDs,
with support for fallback behavior when unknown languages are encountered.
"""

from typing import Dict, Optional


class LanguageRegistry:
    """
    Registry for managing language normalization and resolution.
    
    This class handles:
    - Normalization of language strings (e.g., "Polish", "pl", "pl-PL" -> "polish")
    - Resolution of canonical language keys to language IDs
    - Fallback behavior for unknown languages
    """
    
    # Common language code mappings to canonical names
    # Based on ISO 639-1 (2-letter) and common variations
    LANGUAGE_CODE_MAP = {
        # Chinese variants
        "zh": "chinese",
        "zh-cn": "chinese",
        "zh-tw": "chinese",
        "cmn": "chinese",
        "mandarin": "chinese",
        
        # English
        "en": "english",
        "en-us": "english",
        "en-gb": "english",
        
        # Japanese
        "ja": "japanese",
        "jp": "japanese",
        "jpn": "japanese",
        
        # Korean
        "ko": "korean",
        "kr": "korean",
        "kor": "korean",
        
        # German
        "de": "german",
        "deu": "german",
        
        # French
        "fr": "french",
        "fra": "french",
        
        # Spanish
        "es": "spanish",
        "spa": "spanish",
        
        # Italian
        "it": "italian",
        "ita": "italian",
        
        # Portuguese
        "pt": "portuguese",
        "por": "portuguese",
        "pt-br": "portuguese",
        "pt-pt": "portuguese",
        
        # Russian
        "ru": "russian",
        "rus": "russian",
        
        # Polish
        "pl": "polish",
        "pol": "polish",
        "pl-pl": "polish",
        
        # Dutch
        "nl": "dutch",
        "nld": "dutch",
        
        # Arabic
        "ar": "arabic",
        "ara": "arabic",
        
        # Turkish
        "tr": "turkish",
        "tur": "turkish",
        
        # Auto/unknown
        "auto": "auto",
        "unknown": "auto",
    }
    
    def __init__(self, codec_language_id: Optional[Dict[str, int]] = None):
        """
        Initialize the language registry.
        
        Args:
            codec_language_id: Dictionary mapping language keys to language IDs.
                              If None, no languages are registered.
        """
        self.codec_language_id = codec_language_id or {}
    
    def normalize_language(self, language: str) -> str:
        """
        Normalize a language string to its canonical form.
        
        Handles various formats:
        - Language names: "Polish" -> "polish"
        - ISO codes: "pl", "pl-PL" -> "polish"
        - Case variations: "ENGLISH" -> "english"
        
        Args:
            language: Language string to normalize
            
        Returns:
            Normalized language key (lowercase)
        """
        if not language:
            return "auto"
        
        # Convert to lowercase for consistent handling
        lang_lower = language.lower().strip()
        
        # Check if empty after stripping
        if not lang_lower:
            return "auto"
        
        # Check if it's already a known code
        if lang_lower in self.LANGUAGE_CODE_MAP:
            return self.LANGUAGE_CODE_MAP[lang_lower]
        
        # Otherwise, return as-is (lowercase)
        return lang_lower
    
    def resolve_language(
        self, 
        language: str, 
        strict: bool = False
    ) -> Optional[int]:
        """
        Resolve a language string to its language ID.
        
        Args:
            language: Language string to resolve
            strict: If True, raise NotImplementedError for unknown languages.
                   If False, return None (fallback to auto behavior).
        
        Returns:
            Language ID if found, None if not found and strict=False
            
        Raises:
            NotImplementedError: If language is not found and strict=True
        """
        # Normalize the language string
        normalized = self.normalize_language(language)
        
        # Handle "auto" specially
        if normalized == "auto":
            return None
        
        # Look up the language ID
        if normalized in self.codec_language_id:
            return self.codec_language_id[normalized]
        
        # Handle unknown language based on strict mode
        if strict:
            raise NotImplementedError(f"Language {language} not implemented")
        
        # Fallback: treat unknown language as "auto" (no language token)
        return None


# Convenience functions for standalone use
def normalize_language(language: str, codec_language_id: Optional[Dict[str, int]] = None) -> str:
    """
    Normalize a language string to its canonical form.
    
    Args:
        language: Language string to normalize
        codec_language_id: Optional language ID mapping (for registry initialization)
        
    Returns:
        Normalized language key
    """
    registry = LanguageRegistry(codec_language_id)
    return registry.normalize_language(language)


def resolve_language(
    language: str,
    codec_language_id: Optional[Dict[str, int]] = None,
    strict: bool = False
) -> Optional[int]:
    """
    Resolve a language string to its language ID.
    
    Args:
        language: Language string to resolve
        codec_language_id: Dictionary mapping language keys to language IDs
        strict: If True, raise error for unknown languages
        
    Returns:
        Language ID if found, None if not found and strict=False
        
    Raises:
        NotImplementedError: If language is not found and strict=True
    """
    registry = LanguageRegistry(codec_language_id)
    return registry.resolve_language(language, strict=strict)
