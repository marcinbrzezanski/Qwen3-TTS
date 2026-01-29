#!/usr/bin/env python3
# coding=utf-8
# Copyright 2026 The Alibaba Qwen team.
# SPDX-License-Identifier: Apache-2.0

"""
Example: Running Tokenizer Audit

This script demonstrates how to use the tokenizer audit tool to evaluate
text tokenizer performance on Polish language samples.
"""

import subprocess
import sys
import os


def main():
    """
    Run tokenizer audit example.
    
    This example shows:
    1. How to run the audit script
    2. How to interpret the results
    3. What the output looks like
    """
    
    print("=" * 80)
    print("TOKENIZER AUDIT EXAMPLE")
    print("=" * 80)
    print()
    
    # Check if model path is provided
    model_path = os.environ.get('MODEL_PATH', 'Qwen/Qwen3-TTS-12Hz-1.7B-Base')
    
    print(f"Model path: {model_path}")
    print()
    print("This script will:")
    print("1. Load the text tokenizer from the model")
    print("2. Tokenize Polish text samples")
    print("3. Calculate efficiency metrics")
    print("4. Provide recommendations")
    print()
    print("=" * 80)
    print()
    
    # Construct command
    cmd = [
        sys.executable,
        "tools/tokenizer_audit.py",
        "--model-path", model_path,
        "--output", "example_audit_results.json"
    ]
    
    print("Running command:")
    print(" ".join(cmd))
    print()
    print("=" * 80)
    print()
    
    # Run the audit
    try:
        result = subprocess.run(cmd, check=True)
        
        print()
        print("=" * 80)
        print("✓ Audit completed successfully!")
        print()
        print("Results saved to: example_audit_results.json")
        print()
        print("Next steps:")
        print("1. Review the console output above")
        print("2. Check the JSON file for detailed metrics")
        print("3. Follow the recommendations to decide on tokenizer strategy")
        print("=" * 80)
        
        return 0
        
    except subprocess.CalledProcessError as e:
        print()
        print("=" * 80)
        print("✗ Audit failed!")
        print()
        print("This may be because:")
        print("- The model is not publicly available")
        print("- Required dependencies are missing")
        print("- Network issues prevented model download")
        print()
        print("To run with a local model:")
        print("  export MODEL_PATH=/path/to/local/model")
        print("  python examples/example_tokenizer_audit.py")
        print("=" * 80)
        return 1
    
    except KeyboardInterrupt:
        print()
        print("Interrupted by user")
        return 130


if __name__ == "__main__":
    sys.exit(main())
