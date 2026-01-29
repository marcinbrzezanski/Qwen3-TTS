# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0

"""Language registry for Qwen3-TTS."""

from .registry import LanguageRegistry, normalize_language, resolve_language

__all__ = ["LanguageRegistry", "normalize_language", "resolve_language"]
