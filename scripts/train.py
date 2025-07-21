#!/usr/bin/env python3
"""
Main training script entry point.
Supports both original and enhanced models with proper imports.
"""

import os
import sys
import argparse

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def main():
    parser = argparse.ArgumentParser(description='Train Manim LLM models')
    parser.add_argument('--model', choices=['original', 'enhanced'], 
                       default='enhanced', help='Model type to train')
    parser.add_argument('--windows', action='store_true', 
                       help='Use Windows-compatible training')
    parser.add_argument('--auto', action='store_true',
                       help='Run automatically without prompts')
    parser.add_argument('--epochs', type=int, default=15,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int,
                       help='Training batch size (auto-detected if not specified)')
    parser.add_argument('--skip-data', action='store_true',
                       help='Skip data generation if data already exists')
    parser.add_argument('--skip-tokenizer', action='store_true',
                       help='Skip tokenizer building if tokenizer exists')
    
    args = parser.parse_args()
    
    if args.model == 'enhanced':
        if args.windows:
            # Use Windows-compatible training pipeline
            from training.windows_compatible_training import main as train_main
            # Override sys.argv to pass the arguments
            sys.argv = ['windows_compatible_training.py']
            if args.auto:
                sys.argv.append('--auto')
            if args.epochs != 15:
                sys.argv.extend(['--epochs', str(args.epochs)])
            if args.batch_size:
                sys.argv.extend(['--batch-size', str(args.batch_size)])
            if args.skip_data:
                sys.argv.append('--skip-data')
            if args.skip_tokenizer:
                sys.argv.append('--skip-tokenizer')
            train_main()
        else:
            from training.full_training_pipeline import main as train_main
            sys.argv = ['full_training_pipeline.py']
            if args.auto:
                sys.argv.append('--auto')
            if args.epochs != 15:
                sys.argv.extend(['--epochs', str(args.epochs)])
            if args.batch_size:
                sys.argv.extend(['--batch-size', str(args.batch_size)])
            if args.skip_data:
                sys.argv.append('--skip-data')
            if args.skip_tokenizer:
                sys.argv.append('--skip-tokenizer')
            train_main()
    else:
        # Original model training
        from training.train_model import main as train_main
        train_main()

if __name__ == "__main__":
    main()