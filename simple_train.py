#!/usr/bin/env python3
"""
Simple Standalone Training Script for Enhanced Multi-Domain LLM
This script trains the model without complex import dependencies.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
import time
import pickle
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import argparse

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

class SimpleMultiDomainDataset(Dataset):
    """Simple dataset for multi-domain training."""
    
    def __init__(self, data_file: str, vocab_size: int = 15000, max_length: int = 512):
        """Initialize dataset with simple tokenization."""
        print(f"📚 Loading training data from {data_file}...")
        
        if not os.path.exists(data_file):
            print(f"❌ Data file not found: {data_file}")
            print("Creating sample data...")
            self.data = self.create_sample_data()
        else:
            with open(data_file, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        
        self.vocab_size = vocab_size
        self.max_length = max_length
        
        # Build simple vocabulary
        self.build_vocab()
        
        print(f"✅ Loaded {len(self.data)} training samples")
        print(f"📊 Vocabulary size: {len(self.vocab)}")
    
    def create_sample_data(self):
        """Create sample training data if files don't exist."""
        sample_data = []
        
        # Manim samples
        manim_samples = [
            {
                "prompt": "Create a blue circle that rotates",
                "response": "from manim import *\n\nclass RotatingCircle(Scene):\n    def construct(self):\n        circle = Circle(color=BLUE)\n        self.add(circle)\n        self.play(Rotate(circle, 2*PI), run_time=3)",
                "domain": "manim"
            },
            {
                "prompt": "Draw a sine wave animation",
                "response": "from manim import *\n\nclass SineWave(Scene):\n    def construct(self):\n        axes = Axes()\n        sine_curve = axes.plot(lambda x: np.sin(x), color=YELLOW)\n        self.play(Create(axes), Create(sine_curve))",
                "domain": "manim"
            }
        ]
        
        # DSA samples
        dsa_samples = [
            {
                "prompt": "Explain binary search algorithm",
                "response": "Binary search is an efficient algorithm for finding an item from a sorted list. It works by repeatedly dividing the search interval in half.\n\ndef binary_search(arr, target):\n    left, right = 0, len(arr) - 1\n    while left <= right:\n        mid = (left + right) // 2\n        if arr[mid] == target:\n            return mid\n        elif arr[mid] < target:\n            left = mid + 1\n        else:\n            right = mid - 1\n    return -1\n\nTime Complexity: O(log n)",
                "domain": "dsa"
            },
            {
                "prompt": "What is a linked list?",
                "response": "A linked list is a linear data structure where elements are stored in nodes, and each node contains data and a reference to the next node.\n\nclass ListNode:\n    def __init__(self, val=0, next=None):\n        self.val = val\n        self.next = next\n\nAdvantages:\n- Dynamic size\n- Efficient insertion/deletion at beginning\n\nDisadvantages:\n- No random access\n- Extra memory for pointers",
                "domain": "dsa"
            }
        ]
        
        # System Design samples
        system_design_samples = [
            {
                "prompt": "Explain the singleton pattern",
                "response": "The Singleton pattern ensures a class has only one instance and provides a global point of access to it.\n\nclass Singleton:\n    _instance = None\n    \n    def __new__(cls):\n        if cls._instance is None:\n            cls._instance = super().__new__(cls)\n        return cls._instance\n\nUse cases:\n- Database connections\n- Logger instances\n- Configuration settings",
                "domain": "system_design"
            }
        ]
        
        # Combine samples and replicate to get more data
        all_samples = manim_samples + dsa_samples + system_design_samples
        for i in range(1000):  # Create 5000 samples
            sample_data.append(all_samples[i % len(all_samples)].copy())
        
        return sample_data
    
    def build_vocab(self):
        """Build vocabulary from training data."""
        # Special tokens
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1,
            '<BOS>': 2,
            '<EOS>': 3
        }
        
        # Collect all text
        all_text = []
        for item in self.data:
            # Handle different data formats
            prompt = item.get('prompt', item.get('request', ''))
            response = item.get('response', '')
            all_text.append(prompt)
            all_text.append(response)
        
        # Simple word-based tokenization
        word_freq = {}
        for text in all_text:
            words = text.lower().split()
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Build vocab from most frequent words
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        self.vocab = self.special_tokens.copy()
        
        for word, freq in sorted_words[:self.vocab_size - len(self.special_tokens)]:
            self.vocab[word] = len(self.vocab)
        
        self.id_to_token = {v: k for k, v in self.vocab.items()}
    
    def tokenize(self, text: str) -> List[int]:
        """Simple tokenization."""
        words = text.lower().split()
        tokens = [self.special_tokens['<BOS>']]
        
        for word in words:
            if word in self.vocab:
                tokens.append(self.vocab[word])
            else:
                tokens.append(self.special_tokens['<UNK>'])
        
        tokens.append(self.special_tokens['<EOS>'])
        
        # Pad or truncate
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        else:
            tokens.extend([self.special_tokens['<PAD>']] * (self.max_length - len(tokens)))
        
        return tokens
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Handle different data formats
        input_text = item.get('prompt', item.get('request', ''))
        target_text = item.get('response', '')
        
        input_ids = self.tokenize(input_text)
        labels = self.tokenize(target_text)
        
        # Domain mapping
        domain_map = {'manim': 0, 'dsa': 1, 'system_design': 2, 'general': 3}
        domain_label = domain_map.get(item.get('domain', 'general'), 3)
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'domain_label': torch.tensor(domain_label, dtype=torch.long)
        }


class SimpleTransformer(nn.Module):
    """Simple transformer model for multi-domain training."""
    
    def __init__(self, vocab_size: int, d_model: int = 512, n_heads: int = 8, 
                 n_layers: int = 6, d_ff: int = 2048, max_len: int = 512, num_domains: int = 4):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_len, d_model)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output heads
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.domain_head = nn.Linear(d_model, num_domains)
        
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, input_ids, return_domain_logits=False):
        batch_size, seq_len = input_ids.shape
        
        # Create position ids
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(batch_size, -1)
        
        # Embeddings
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding(positions)
        x = self.dropout(token_emb + pos_emb)
        
        # Create attention mask for padding
        pad_mask = (input_ids == 0)
        
        # Transformer
        x = self.transformer(x, src_key_padding_mask=pad_mask)
        
        # Language modeling head
        lm_logits = self.lm_head(x)
        
        outputs = {'logits': lm_logits}
        
        if return_domain_logits:
            # Use pooled representation for domain classification
            pooled = x.mean(dim=1)  # Simple mean pooling
            domain_logits = self.domain_head(pooled)
            outputs['domain_logits'] = domain_logits
        
        return outputs


def train_model(data_file: str, epochs: int = 15, batch_size: int = 2, learning_rate: float = 1e-4):
    """Train the simple multi-domain model."""
    
    print("🚀 Starting Simple Multi-Domain LLM Training")
    print("=" * 60)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Using device: {device}")
    
    # Dataset
    dataset = SimpleMultiDomainDataset(data_file)
    
    # Split into train/validation
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"📊 Training samples: {len(train_dataset)}")
    print(f"📊 Validation samples: {len(val_dataset)}")
    
    # Model
    model = SimpleTransformer(
        vocab_size=len(dataset.vocab),
        d_model=512,
        n_heads=8,
        n_layers=6,
        d_ff=2048,
        max_len=512,
        num_domains=4
    ).to(device)
    
    print(f"🤖 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    lm_criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore padding
    domain_criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        print(f"\n📅 Epoch {epoch + 1}/{epochs}")
        print("-" * 40)
        
        # Training
        model.train()
        train_loss = 0
        train_batches = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            input_ids = batch['input_ids'].to(device)
            labels = batch['labels'].to(device)
            domain_labels = batch['domain_label'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(input_ids, return_domain_logits=True)
            
            # Calculate losses
            lm_loss = lm_criterion(outputs['logits'].view(-1, model.vocab_size), labels.view(-1))
            domain_loss = domain_criterion(outputs['domain_logits'], domain_labels)
            loss = lm_loss + 0.1 * domain_loss
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_batches += 1
        
        avg_train_loss = train_loss / train_batches
        
        # Validation
        model.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(device)
                labels = batch['labels'].to(device)
                domain_labels = batch['domain_label'].to(device)
                
                outputs = model(input_ids, return_domain_logits=True)
                
                lm_loss = lm_criterion(outputs['logits'].view(-1, model.vocab_size), labels.view(-1))
                domain_loss = domain_criterion(outputs['domain_logits'], domain_labels)
                loss = lm_loss + 0.1 * domain_loss
                
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        
        print(f"📊 Train Loss: {avg_train_loss:.4f}")
        print(f"📊 Val Loss: {avg_val_loss:.4f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            
            # Create checkpoints directory
            os.makedirs('checkpoints', exist_ok=True)
            
            # Save model
            torch.save({
                'model_state_dict': model.state_dict(),
                'vocab': dataset.vocab,
                'config': {
                    'vocab_size': len(dataset.vocab),
                    'd_model': 512,
                    'n_heads': 8,
                    'n_layers': 6,
                    'max_len': 512,
                    'num_domains': 4
                }
            }, 'checkpoints/best_model_simple.pth')
            
            print(f"💾 Best model saved! (Val Loss: {avg_val_loss:.4f})")
    
    print("\n🎉 Training completed successfully!")
    print("📁 Model saved to: checkpoints/best_model_simple.pth")
    return model, dataset


def main():
    parser = argparse.ArgumentParser(description='Simple Multi-Domain LLM Training')
    parser.add_argument('--data', type=str, default='data/enhanced_training_data.json', 
                       help='Path to training data')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    
    args = parser.parse_args()
    
    # Train model
    model, dataset = train_model(
        data_file=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    print("\n🚀 To test your model:")
    print("   python test_simple_model.py")


if __name__ == "__main__":
    main()