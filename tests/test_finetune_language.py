#!/usr/bin/env python
# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0

"""
Unit tests for the language fine-tuning script.

These tests validate the core functionality of finetune_language.py without
requiring actual model weights or GPUs.
"""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

# Check if torch is available
try:
    import torch
    from safetensors.torch import load_file

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# Skip all tests if torch is not available
pytestmark = pytest.mark.skipif(not TORCH_AVAILABLE, reason="torch not installed")


class MockProcessor:
    def __call__(self, text, return_tensors=None, padding=None):
        return {"input_ids": torch.tensor([[11, 12, 13, 14, 15]])}


class DummySpeakerEncoder(torch.nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.proj = torch.nn.Linear(128, hidden_size, bias=False)

    def forward(self, ref_mels):
        pooled = ref_mels.mean(dim=1)
        return self.proj(pooled)


class DummyCore(torch.nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.text_embedding = torch.nn.Embedding(256, hidden_size)
        self.codec_embedding = torch.nn.Embedding(vocab_size, hidden_size)
        self.layers = torch.nn.ModuleList([torch.nn.Linear(hidden_size, hidden_size)])
        self.config = SimpleNamespace(num_hidden_layers=1)


class DummyCodePredictor(torch.nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.embeddings = torch.nn.ModuleList(
            [torch.nn.Embedding(vocab_size, hidden_size) for _ in range(15)]
        )

    def get_input_embeddings(self):
        return self.embeddings


class DummyTalker(torch.nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.model = DummyCore(hidden_size=hidden_size, vocab_size=vocab_size)
        self.code_predictor = DummyCodePredictor(hidden_size=hidden_size, vocab_size=vocab_size)

    def forward(self, inputs_embeds, attention_mask, labels, output_hidden_states=True):
        hidden = self.model.layers[0](inputs_embeds)
        masked_hidden = hidden * attention_mask.unsqueeze(-1)
        loss = masked_hidden.sum()
        return SimpleNamespace(loss=loss, hidden_states=[[hidden]])

    def forward_sub_talker_finetune(self, talker_codec_ids, talker_hidden_states):
        if talker_hidden_states.numel() == 0:
            loss = talker_hidden_states.sum()
        else:
            loss = talker_hidden_states.sum() * 0.01
        return None, loss


class DummyTrainModel(torch.nn.Module):
    def __init__(self, config, hidden_size: int = 8, vocab_size: int = 5000):
        super().__init__()
        self.config = config
        self.dtype = torch.float32
        self.talker = DummyTalker(hidden_size=hidden_size, vocab_size=vocab_size)
        self.speaker_encoder = DummySpeakerEncoder(hidden_size=hidden_size)

    @property
    def device(self):
        return next(self.parameters()).device


class DummyInferenceModel(DummyTrainModel):
    def __init__(self, config, hidden_size: int = 8, vocab_size: int = 5000):
        super().__init__(config=config, hidden_size=hidden_size, vocab_size=vocab_size)
        self.tts_model_type = "base"
        self.tokenizer_type = "dummy"
        self.tts_model_size = "dummy"
        self.speaker_encoder_sample_rate = 24000
        self.generate_config = {}
        self.last_generate_languages = None
        self.speech_tokenizer = SimpleNamespace(
            decode=lambda items: ([torch.zeros(24, dtype=torch.float32).numpy() for _ in items], 24000)
        )

    def get_supported_languages(self):
        return ["auto", *self.config.talker_config.codec_language_id.keys()]

    def generate(self, input_ids, ref_ids=None, voice_clone_prompt=None, languages=None, non_streaming_mode=False, **kwargs):
        self.last_generate_languages = languages
        return [torch.zeros((2, 16), dtype=torch.long) for _ in languages], None


class DummyAccelerator:
    is_main_process = True

    @staticmethod
    def unwrap_model(model):
        return model


class DummyScheduler:
    def __init__(self):
        self._state = {"step": 0}

    def state_dict(self):
        return dict(self._state)


class DummyProcessorForInference:
    def __init__(self):
        self.tokenizer = SimpleNamespace(vocab_size=256)

    def __call__(self, text, return_tensors=None, padding=None):
        return {"input_ids": torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8]])}


class DummyRegisteredModel(DummyTrainModel):
    def __init__(self, config, hidden_size: int = 8, vocab_size: int = 5000):
        super().__init__(config=config, hidden_size=hidden_size, vocab_size=vocab_size)
        self.supported_languages = ["auto", *config.talker_config.codec_language_id.keys()]

    def save_pretrained(self, output_path: str):
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "config.json", "w", encoding="utf-8") as f:
            json.dump(config_to_dict(self.config), f, indent=2)
        for name in ("generation_config.json", "tokenizer_config.json"):
            (output_dir / name).write_text("{}", encoding="utf-8")



def make_config(codec_language_id=None):
    return SimpleNamespace(
        tts_pad_token_id=90,
        tts_bos_token_id=91,
        tts_eos_token_id=92,
        tts_model_type="base",
        tokenizer_type="dummy",
        tts_model_size="dummy",
        talker_config=SimpleNamespace(
            codec_nothink_id=4203,
            codec_think_bos_id=4204,
            codec_think_eos_id=4205,
            codec_think_id=4202,
            codec_pad_id=4196,
            codec_bos_id=4197,
            codec_eos_token_id=4198,
            codec_language_id=dict(codec_language_id or {"english": 4201}),
            vocab_size=5000,
            initializer_range=0.02,
        ),
        speaker_encoder_config=SimpleNamespace(sample_rate=24000),
    )



def config_to_dict(config):
    return {
        "tts_pad_token_id": config.tts_pad_token_id,
        "tts_bos_token_id": config.tts_bos_token_id,
        "tts_eos_token_id": config.tts_eos_token_id,
        "tts_model_type": config.tts_model_type,
        "tokenizer_type": config.tokenizer_type,
        "tts_model_size": config.tts_model_size,
        "talker_config": {
            "codec_nothink_id": config.talker_config.codec_nothink_id,
            "codec_think_bos_id": config.talker_config.codec_think_bos_id,
            "codec_think_eos_id": config.talker_config.codec_think_eos_id,
            "codec_think_id": config.talker_config.codec_think_id,
            "codec_pad_id": config.talker_config.codec_pad_id,
            "codec_bos_id": config.talker_config.codec_bos_id,
            "codec_eos_token_id": config.talker_config.codec_eos_token_id,
            "codec_language_id": dict(config.talker_config.codec_language_id),
            "vocab_size": config.talker_config.vocab_size,
            "initializer_range": config.talker_config.initializer_range,
        },
        "speaker_encoder_config": {"sample_rate": config.speaker_encoder_config.sample_rate},
    }



def make_collated_batch(language_id=None):
    from scripts.finetune_language import LanguageDataset

    dataset = LanguageDataset([], MockProcessor(), make_config({"english": 4201, "polish": 4206}))
    item = {
        "text_ids": torch.tensor([[11, 12, 13, 14, 15, 16]], dtype=torch.long),
        "audio_codes": torch.tensor(
            [[31 + i for i in range(16)], [47 + i for i in range(16)]],
            dtype=torch.long,
        ),
        "ref_mel": torch.ones((1, 3, 128), dtype=torch.float32),
        "language": "polish" if language_id is not None else "auto",
        "language_id": language_id,
    }
    return dataset.collate_fn([item])



def test_imports():
    """Test that all required imports work."""
    try:
        from scripts.finetune_language import LanguageDataset
        from scripts.finetune_language import freeze_all_except_language_embeddings
        from scripts.finetune_language import load_checkpoint
        from scripts.finetune_language import save_checkpoint

        assert True
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")



def test_peft_optional():
    """Test that PEFT is optional and script handles its absence gracefully."""
    from scripts.finetune_language import PEFT_AVAILABLE

    assert isinstance(PEFT_AVAILABLE, bool)



def test_dataset_structure():
    """Test that LanguageDataset has the expected structure."""
    from scripts.finetune_language import LanguageDataset

    assert hasattr(LanguageDataset, "__init__")
    assert hasattr(LanguageDataset, "__len__")
    assert hasattr(LanguageDataset, "__getitem__")
    assert hasattr(LanguageDataset, "collate_fn")
    assert hasattr(LanguageDataset, "_load_audio_to_np")
    assert hasattr(LanguageDataset, "_build_assistant_text")
    assert hasattr(LanguageDataset, "extract_mels")



def test_training_modes():
    """Test that supported training modes are correctly defined."""
    from scripts.finetune_language import freeze_all_except_language_embeddings

    assert callable(freeze_all_except_language_embeddings)



def test_assistant_text_formatting():
    """Test the assistant text formatting function."""
    from scripts.finetune_language import LanguageDataset

    dataset = LanguageDataset([], MockProcessor(), make_config())
    text = "Hello world"
    formatted = dataset._build_assistant_text(text)

    assert "<|im_start|>assistant" in formatted
    assert "<|im_end|>" in formatted
    assert "Hello world" in formatted



def test_smoke_test_data_exists():
    """Test that smoke test data file exists."""
    test_data_path = Path(__file__).parent / "smoke_test_data" / "sample_train.jsonl"

    assert test_data_path.exists(), f"Smoke test data not found at {test_data_path}"

    with open(test_data_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        assert len(lines) > 0, "Smoke test data is empty"

        for line in lines:
            if not line.strip():
                continue
            data = json.loads(line)
            assert "audio" in data
            assert "text" in data
            assert "ref_audio" in data
            assert "audio_codes" in data



def test_script_has_main():
    """Test that the script has a main entry point."""
    from scripts.finetune_language import train

    assert callable(train)



def test_checkpoint_functions_exist():
    """Test that checkpoint save/load functions exist."""
    from scripts.finetune_language import load_checkpoint, save_checkpoint

    assert callable(save_checkpoint)
    assert callable(load_checkpoint)



def test_language_embedding_gradient_mask_keeps_only_target_row():
    """Test that gradient masking keeps only the target embedding row gradients."""
    from scripts.finetune_language import register_language_embedding_gradient_mask

    class _MockCore(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.codec_embedding = torch.nn.Embedding(8, 4)
            self.layers = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(2)])
            self.config = SimpleNamespace(num_hidden_layers=2)

    class _MockTalker(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _MockCore()

    class _MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.talker = _MockTalker()

    model = _MockModel()
    handle = register_language_embedding_gradient_mask(model, language_id=3)

    emb = model.talker.model.codec_embedding
    emb.weight.grad = None
    emb.weight.sum().backward()

    grad = emb.weight.grad
    assert grad is not None
    non_zero_rows = (grad.abs().sum(dim=1) > 0).nonzero(as_tuple=False).flatten().tolist()
    assert non_zero_rows == [3]

    handle.remove()



def test_language_embedding_gradient_mask_raises_for_out_of_range_id():
    """Test out-of-range language_id validation."""
    from scripts.finetune_language import register_language_embedding_gradient_mask

    class _MockCore(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.codec_embedding = torch.nn.Embedding(4, 2)

    class _MockTalker(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _MockCore()

    class _MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.talker = _MockTalker()

    model = _MockModel()
    with pytest.raises(ValueError, match="out of codec_embedding range"):
        register_language_embedding_gradient_mask(model, language_id=10)



def test_create_optimizer_disables_decay_for_masked_embedding_row():
    """Embedding param group should have zero decay when language row masking is active."""
    from scripts.finetune_language import create_optimizer

    class _MockCore(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.codec_embedding = torch.nn.Embedding(8, 4)
            self.layers = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(2)])
            self.config = SimpleNamespace(num_hidden_layers=2)

    class _MockTalker(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = _MockCore()

    class _MockModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.talker = _MockTalker()

    model = _MockModel()
    model.talker.model.codec_embedding.weight.requires_grad = True
    for layer in model.talker.model.layers:
        for param in layer.parameters():
            param.requires_grad = True

    optimizer = create_optimizer(model, learning_rate=1e-4, weight_decay=0.1, language_id=3)

    embedding_param = model.talker.model.codec_embedding.weight
    embedding_group = next(
        group for group in optimizer.param_groups
        if any(id(param) == id(embedding_param) for param in group["params"])
    )
    assert embedding_group["weight_decay"] == 0.0

    non_embedding_groups = [
        group for group in optimizer.param_groups
        if len(group["params"]) > 0 and all(id(param) != id(embedding_param) for param in group["params"])
    ]
    assert non_embedding_groups



def test_lang_only_requires_registered_target_language():
    from scripts.finetune_language import validate_lang_only_target_language

    with pytest.raises(NotImplementedError, match="not implemented"):
        validate_lang_only_target_language(make_config({"english": 4201}), "polish")



def test_dataset_normalizes_language_using_registry():
    from scripts.finetune_language import resolve_training_language
    from qwen_tts.langs.registry import LanguageRegistry

    registry = LanguageRegistry({"english": 4201, "polish": 4206})
    normalized, language_id = resolve_training_language(registry, sample_language="pl-PL", target_language=None)

    assert normalized == "polish"
    assert language_id == 4206



def test_sequence_layout_with_language_token_uses_named_positions():
    from scripts.finetune_language import build_training_sequence_layout, build_codec_conditioning_tokens

    config = make_config({"english": 4201, "polish": 4206})
    prompt_tokens = build_codec_conditioning_tokens(config, 4206)
    layout = build_training_sequence_layout(text_ids_len=6, codec_ids_len=2, prompt_tokens=prompt_tokens)
    batch = make_collated_batch(language_id=4206)

    assert layout.prompt_tokens == tuple([4202, 4204, 4206, 4205, 0, 4196, 4197])
    assert layout.speaker_position == 7
    assert layout.text_start == 9
    assert batch["input_ids"][0, layout.prompt_start:layout.prompt_end, 1].tolist() == [4202, 4204, 4206, 4205, 0, 4196]
    assert batch["input_ids"][0, layout.text_start, 1].item() == 4196
    assert batch["input_ids"][0, layout.codec_bos_position, 1].item() == 4197
    assert batch["speaker_positions"].tolist() == [7]



def test_forward_uses_language_token_and_gives_target_codec_row_gradient():
    from scripts.finetune_language import forward_training_batch

    config = make_config({"english": 4201, "polish": 4206})
    model = DummyTrainModel(config=config)

    batch_with_language = make_collated_batch(language_id=4206)
    model.zero_grad(set_to_none=True)
    loss = forward_training_batch(model, batch_with_language)
    loss.backward()
    grad_with_language = model.talker.model.codec_embedding.weight.grad[4206].abs().sum().item()

    model.zero_grad(set_to_none=True)
    batch_without_language = make_collated_batch(language_id=None)
    loss_without_language = forward_training_batch(model, batch_without_language)
    loss_without_language.backward()
    grad_without_language = model.talker.model.codec_embedding.weight.grad[4206].abs().sum().item()

    assert grad_with_language > 0
    assert grad_without_language == 0



def test_integration_register_train_save_reload_and_infer_polish(tmp_path, monkeypatch):
    from qwen_tts.inference.qwen3_tts_model import Qwen3TTSModel
    from scripts.finetune_language import create_optimizer, forward_training_batch, save_checkpoint
    from scripts.register_language import register_language

    base_config = make_config({"english": 4201})

    def fake_auto_config_from_pretrained(model_path, trust_remote_code=True):
        return make_config({"english": 4201})

    def fake_model_from_pretrained(model_path, config=None, torch_dtype=None, low_cpu_mem_usage=None):
        return DummyRegisteredModel(config=config)

    monkeypatch.setattr("scripts.register_language.AutoConfig.from_pretrained", fake_auto_config_from_pretrained)
    monkeypatch.setattr(
        "scripts.register_language.Qwen3TTSForConditionalGeneration.from_pretrained",
        fake_model_from_pretrained,
    )

    registered_dir = tmp_path / "registered"
    result = register_language(
        model_path=str(tmp_path / "base"),
        language="polish",
        output_path=str(registered_dir),
    )
    assert result["language_id"] == 4206

    registered_config_dict = json.loads((registered_dir / "config.json").read_text(encoding="utf-8"))
    registered_config = make_config(registered_config_dict["talker_config"]["codec_language_id"])
    train_model = DummyTrainModel(config=registered_config)

    batch = make_collated_batch(language_id=4206)
    optimizer = create_optimizer(train_model, learning_rate=1e-3, weight_decay=0.0, language_id=4206)
    scheduler = DummyScheduler()
    train_model.zero_grad(set_to_none=True)
    loss = forward_training_batch(train_model, batch)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    args = SimpleNamespace(train_mode="full", target_language="polish")
    output_dir = tmp_path / "output"
    save_checkpoint(
        DummyAccelerator(),
        train_model,
        optimizer,
        scheduler,
        epoch=1,
        step=1,
        output_dir=str(output_dir),
        model_path=str(registered_dir),
        args=args,
    )

    checkpoint_dir = output_dir / "checkpoint-epoch-1"
    state_dict = load_file(checkpoint_dir / "model.safetensors")
    reloaded_config_dict = json.loads((checkpoint_dir / "config.json").read_text(encoding="utf-8"))
    reloaded_model = DummyInferenceModel(make_config(reloaded_config_dict["talker_config"]["codec_language_id"]))
    reloaded_model.load_state_dict(state_dict, strict=True)

    wrapper = Qwen3TTSModel(model=reloaded_model, processor=DummyProcessorForInference())
    voice_clone_prompt = {
        "ref_code": [None],
        "ref_spk_embedding": [torch.zeros(8)],
        "x_vector_only_mode": [True],
        "icl_mode": [False],
    }
    wavs, sr = wrapper.generate_voice_clone(
        text="Cześć świecie",
        language="polish",
        voice_clone_prompt=voice_clone_prompt,
    )

    assert sr == 24000
    assert len(wavs) == 1
    assert reloaded_model.last_generate_languages == ["polish"]
    assert "polish" in reloaded_config_dict["talker_config"]["codec_language_id"]
    assert any(key.startswith("speaker_encoder.") for key in state_dict.keys())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
