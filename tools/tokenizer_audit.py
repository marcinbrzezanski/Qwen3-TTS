#!/usr/bin/env python3
# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0

"""
Tokenizer Audit Script for Polish Language Support

This script audits the text tokenizer used by Qwen3-TTS to evaluate its suitability
for Polish language support. It analyzes:
1. Average tokens per character
2. Longest token sequences
3. Unknown/fallback token behavior
4. Compression efficiency

The goal is to determine if the tokenizer needs to be extended or if the baseline
tokenizer is sufficient for Polish language support.
"""

import argparse
import json
import os
import sys
from typing import Dict, List

import numpy as np

# Try to import transformers and related dependencies
try:
    from transformers import AutoProcessor
except ImportError:
    print("Error: transformers not installed. Please install: pip install transformers")
    sys.exit(1)


# Polish text samples for testing
# These cover various text types: news, conversational, numbers/dates, special characters
POLISH_TEXT_SAMPLES = {
    "news": [
        "Warszawa, 15 stycznia 2026 roku. Prezydent Rzeczypospolitej Polskiej ogłosił dzisiaj nowe reformy gospodarcze.",
        "Polski rząd wprowadził zmiany w systemie podatkowym, które mają na celu wspieranie małych i średnich przedsiębiorstw.",
        "Najnowsze badania pokazują, że Polska osiągnęła wzrost gospodarczy na poziomie 4,5% w ubiegłym kwartale.",
        "Ministerstwo Zdrowia poinformowało o uruchomieniu nowego programu szczepień przeciwko grypie dla osób starszych.",
    ],
    "conversational": [
        "Cześć! Jak się masz? Co słychać?",
        "Dzień dobry, panie profesorze. Czy mogę zadać pytanie?",
        "Świetnie! Kiedy się spotykamy? O której będziemy?",
        "Przepraszam, nie rozumiem. Czy może pan powtórzyć?",
        "Dziękuję bardzo za pomoc. To było naprawdę miłe z pana strony.",
        "Czy mógłbyś mi powiedzieć, jak dojść do dworca?",
    ],
    "numbers_and_dates": [
        "W roku 2026 obchodzimy 1050. rocznicę chrztu Polski.",
        "Cena wynosi 129,99 złotych. To jest rabat 25% od ceny podstawowej.",
        "Spotkanie odbędzie się 29 stycznia o godzinie 15:30 w sali konferencyjnej.",
        "Numer telefonu: +48 22 123 45 67, kod pocztowy: 00-001 Warszawa.",
        "W tym miesiącu temperatura oscyluje między -5°C a +10°C.",
    ],
    "special_characters": [
        "Polskie znaki diakrytyczne: ą, ć, ę, ł, ń, ó, ś, ź, ż.",
        "Używamy również znaków: !, @, #, $, %, &, *, (, ), -, _, +, =.",
        "Znaki interpunkcyjne: kropka, przecinek, średnik; dwukropek: pytajnik? wykrzyknik!",
        "Cudzysłowy: \"tekst w cudzysłowie\", 'tekst w apostrofach'.",
    ],
    "technical": [
        "Implementacja algorytmu wykorzystuje bibliotekę NumPy w wersji 1.24.3.",
        "Konfiguracja sieci neuronowej: 128 warstw, learning_rate=0.001, batch_size=32.",
        "Instalacja: pip install transformers torch torchaudio soundfile librosa",
        "URL dokumentacji: https://github.com/Qwen/Qwen3-TTS/blob/main/README.md",
    ],
    "mixed": [
        "Pan Kowalski mieszka przy ul. Długiej 15/3 w Krakowie od 2015 roku.",
        "E-mail: jan.kowalski@example.pl, telefon: 601-234-567.",
        "Zakupiłem laptop za 4999 zł (około $1200 USD) w promocji Black Friday.",
        "Spotkajmy się w kawiarni \"Pod Aniołem\" w centrum miasta o 18:00.",
    ],
}


def load_tokenizer(model_path: str = None):
    """
    Load the text tokenizer from the Qwen3-TTS model.
    
    Args:
        model_path: Path or HuggingFace model ID. If None, uses a default.
        
    Returns:
        The tokenizer object from the processor
    """
    if model_path is None:
        # Default to a publicly available model if available
        # For now, we'll need a model path to be provided
        raise ValueError(
            "Please provide a model path using --model-path argument. "
            "Example: --model-path Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        )
    
    print(f"Loading tokenizer from: {model_path}")
    try:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        tokenizer = processor.tokenizer
        print(f"Tokenizer loaded: {type(tokenizer).__name__}")
        print(f"Vocabulary size: {tokenizer.vocab_size}")
        return tokenizer
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        raise


def analyze_tokenization(
    tokenizer, 
    text: str
) -> Dict:
    """
    Analyze tokenization for a single text sample.
    
    Args:
        tokenizer: The tokenizer to use
        text: Text to tokenize
        
    Returns:
        Dictionary with analysis results
    """
    # Tokenize the text
    tokens = tokenizer.encode(text, add_special_tokens=False)
    decoded_text = tokenizer.decode(tokens)
    
    # Calculate metrics
    num_tokens = len(tokens)
    num_chars = len(text)
    tokens_per_char = num_tokens / num_chars if num_chars > 0 else 0
    
    # Check for unknown tokens
    # Most tokenizers use a special UNK token
    unk_token_id = tokenizer.unk_token_id if hasattr(tokenizer, 'unk_token_id') else None
    num_unk_tokens = tokens.count(unk_token_id) if unk_token_id is not None else 0
    
    # Get token strings
    token_strings = [tokenizer.decode([t]) for t in tokens]
    
    # Check if decoding is lossless
    is_lossless = (text == decoded_text)
    
    return {
        "text": text,
        "num_tokens": num_tokens,
        "num_chars": num_chars,
        "tokens_per_char": tokens_per_char,
        "num_unk_tokens": num_unk_tokens,
        "is_lossless": is_lossless,
        "token_ids": tokens,
        "token_strings": token_strings,
        "decoded_text": decoded_text,
    }


def analyze_category(
    tokenizer,
    category_name: str,
    texts: List[str]
) -> Dict:
    """
    Analyze tokenization for a category of texts.
    
    Args:
        tokenizer: The tokenizer to use
        category_name: Name of the category
        texts: List of texts to analyze
        
    Returns:
        Dictionary with category-level statistics
    """
    print(f"\nAnalyzing category: {category_name}")
    
    results = []
    for text in texts:
        result = analyze_tokenization(tokenizer, text)
        results.append(result)
    
    # Calculate aggregate statistics
    tokens_per_char_list = [r["tokens_per_char"] for r in results]
    num_tokens_list = [r["num_tokens"] for r in results]
    num_unk_tokens_list = [r["num_unk_tokens"] for r in results]
    
    stats = {
        "category": category_name,
        "num_samples": len(texts),
        "avg_tokens_per_char": np.mean(tokens_per_char_list),
        "std_tokens_per_char": np.std(tokens_per_char_list),
        "min_tokens_per_char": np.min(tokens_per_char_list),
        "max_tokens_per_char": np.max(tokens_per_char_list),
        "avg_num_tokens": np.mean(num_tokens_list),
        "max_num_tokens": np.max(num_tokens_list),
        "total_unk_tokens": sum(num_unk_tokens_list),
        "samples_with_unk": sum(1 for r in results if r["num_unk_tokens"] > 0),
        "lossless_count": sum(1 for r in results if r["is_lossless"]),
        "samples": results,
    }
    
    print(f"  Avg tokens/char: {stats['avg_tokens_per_char']:.3f} ± {stats['std_tokens_per_char']:.3f}")
    print(f"  Max tokens/char: {stats['max_tokens_per_char']:.3f}")
    print(f"  Max sequence length: {stats['max_num_tokens']} tokens")
    print(f"  Unknown tokens: {stats['total_unk_tokens']} total, {stats['samples_with_unk']} samples affected")
    print(f"  Lossless encoding: {stats['lossless_count']}/{stats['num_samples']} samples")
    
    return stats


def run_audit(tokenizer, output_file: str = None):
    """
    Run complete tokenizer audit on Polish text samples.
    
    Args:
        tokenizer: The tokenizer to audit
        output_file: Optional path to save JSON results
        
    Returns:
        Dictionary with complete audit results
    """
    print("="*80)
    print("POLISH TOKENIZER AUDIT")
    print("="*80)
    
    # Analyze each category
    category_results = {}
    for category_name, texts in POLISH_TEXT_SAMPLES.items():
        category_results[category_name] = analyze_category(tokenizer, category_name, texts)
    
    # Calculate overall statistics
    all_tokens_per_char = []
    all_num_tokens = []
    total_unk_tokens = 0
    total_samples = 0
    total_lossless = 0
    
    for category_stats in category_results.values():
        all_tokens_per_char.extend([s["tokens_per_char"] for s in category_stats["samples"]])
        all_num_tokens.extend([s["num_tokens"] for s in category_stats["samples"]])
        total_unk_tokens += category_stats["total_unk_tokens"]
        total_samples += category_stats["num_samples"]
        total_lossless += category_stats["lossless_count"]
    
    overall_stats = {
        "total_samples": total_samples,
        "avg_tokens_per_char": np.mean(all_tokens_per_char),
        "std_tokens_per_char": np.std(all_tokens_per_char),
        "min_tokens_per_char": np.min(all_tokens_per_char),
        "max_tokens_per_char": np.max(all_tokens_per_char),
        "avg_sequence_length": np.mean(all_num_tokens),
        "max_sequence_length": np.max(all_num_tokens),
        "total_unk_tokens": total_unk_tokens,
        "lossless_encoding_rate": total_lossless / total_samples if total_samples > 0 else 0,
    }
    
    print("\n" + "="*80)
    print("OVERALL STATISTICS")
    print("="*80)
    print(f"Total samples analyzed: {overall_stats['total_samples']}")
    print(f"Average tokens per character: {overall_stats['avg_tokens_per_char']:.3f} ± {overall_stats['std_tokens_per_char']:.3f}")
    print(f"Min/Max tokens per character: {overall_stats['min_tokens_per_char']:.3f} / {overall_stats['max_tokens_per_char']:.3f}")
    print(f"Average sequence length: {overall_stats['avg_sequence_length']:.1f} tokens")
    print(f"Maximum sequence length: {overall_stats['max_sequence_length']} tokens")
    print(f"Total unknown tokens: {overall_stats['total_unk_tokens']}")
    print(f"Lossless encoding rate: {overall_stats['lossless_encoding_rate']*100:.1f}%")
    
    # Compile full results
    results = {
        "tokenizer_info": {
            "name": type(tokenizer).__name__,
            "vocab_size": tokenizer.vocab_size,
            "model_max_length": getattr(tokenizer, 'model_max_length', None),
        },
        "overall_statistics": overall_stats,
        "category_results": {
            k: {
                "category": v["category"],
                "num_samples": v["num_samples"],
                "avg_tokens_per_char": v["avg_tokens_per_char"],
                "std_tokens_per_char": v["std_tokens_per_char"],
                "max_tokens_per_char": v["max_tokens_per_char"],
                "max_num_tokens": v["max_num_tokens"],
                "total_unk_tokens": v["total_unk_tokens"],
                "samples_with_unk": v["samples_with_unk"],
                "lossless_count": v["lossless_count"],
            }
            for k, v in category_results.items()
        },
        # Include detailed sample results for debugging
        "detailed_samples": {
            k: [
                {
                    "text": s["text"],
                    "num_tokens": s["num_tokens"],
                    "tokens_per_char": s["tokens_per_char"],
                    "num_unk_tokens": s["num_unk_tokens"],
                    "is_lossless": s["is_lossless"],
                    "token_strings": s["token_strings"],
                }
                for s in v["samples"]
            ]
            for k, v in category_results.items()
        }
    }
    
    # Save to file if requested
    if output_file:
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\n✓ Results saved to: {output_file}")
    
    print("\n" + "="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    
    # Provide recommendations based on results
    if overall_stats['avg_tokens_per_char'] > 0.5:
        print("⚠ WARNING: High tokens/char ratio suggests poor tokenization efficiency for Polish.")
        print("  → Consider extending the tokenizer with Polish-specific vocabulary.")
    elif overall_stats['avg_tokens_per_char'] > 0.35:
        print("⚠ MODERATE: Tokenization efficiency is acceptable but could be improved.")
        print("  → Baseline tokenizer may work, but extension could improve performance.")
    else:
        print("✓ GOOD: Tokenization efficiency is good for Polish text.")
        print("  → Baseline tokenizer appears sufficient for Polish support.")
    
    if overall_stats['total_unk_tokens'] > 0:
        print(f"\n⚠ WARNING: Found {overall_stats['total_unk_tokens']} unknown tokens.")
        print("  → This indicates characters or sequences not in the vocabulary.")
        print("  → Tokenizer extension is REQUIRED for proper Polish support.")
    else:
        print("\n✓ GOOD: No unknown tokens detected.")
        print("  → All Polish characters are properly handled by the tokenizer.")
    
    if overall_stats['lossless_encoding_rate'] < 1.0:
        print(f"\n⚠ WARNING: Only {overall_stats['lossless_encoding_rate']*100:.1f}% lossless encoding.")
        print("  → Some information is lost during tokenization/detokenization.")
        print("  → This may affect Polish text quality.")
    else:
        print("\n✓ GOOD: 100% lossless encoding - no information loss.")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Audit text tokenizer for Polish language support in Qwen3-TTS"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path or HuggingFace model ID (e.g., Qwen/Qwen3-TTS-12Hz-1.7B-Base)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tokenizer_audit_results.json",
        help="Output JSON file path (default: tokenizer_audit_results.json)"
    )
    
    args = parser.parse_args()
    
    try:
        # Load tokenizer
        tokenizer = load_tokenizer(args.model_path)
        
        # Run audit
        results = run_audit(tokenizer, args.output)
        
        print("\n✓ Audit completed successfully!")
        return 0
        
    except Exception as e:
        print(f"\n✗ Error during audit: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
