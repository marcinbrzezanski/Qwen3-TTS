# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0

import pytest

from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel


class _DummyTalkerConfig:
    def __init__(self, codec_language_id):
        self.codec_language_id = codec_language_id


class _DummyConfig:
    def __init__(self, codec_language_id):
        self.talker_config = _DummyTalkerConfig(codec_language_id)


class _DummyModel:
    def __init__(self, codec_language_id):
        self.config = _DummyConfig(codec_language_id)

    def get_supported_languages(self):
        return list(self.config.talker_config.codec_language_id.keys())


@pytest.fixture
def wrapper():
    model = _DummyModel(
        {
            "english": 4201,
            "polish": 4206,
        }
    )
    return Qwen3TTSModel(model=model, processor=None)


def test_normalize_languages_accepts_polish_aliases(wrapper):
    normalized = wrapper._normalize_languages(["Polish", "pl", "pl-PL", "pol"])
    assert normalized == ["polish", "polish", "polish", "polish"]


def test_validate_languages_after_normalization(wrapper):
    normalized = wrapper._normalize_languages(["pl-PL"])
    wrapper._validate_languages(normalized)


def test_validate_languages_rejects_unknown_language(wrapper):
    normalized = wrapper._normalize_languages(["klingon"])
    with pytest.raises(ValueError, match="Unsupported languages"):
        wrapper._validate_languages(normalized)
