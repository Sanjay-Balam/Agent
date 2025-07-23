#!/usr/bin/env python3
"""
Test script for the simple multi-domain model
"""

import torch
import torch.nn.functional as F
import json
import os

class SimpleTransformer(torch.nn.Module):
    """Simple transformer model for multi-domain training."""
    
    def __init__(self, vocab_size: int, d_model: int = 512, n_heads: int = 8, 
                 n_layers: int = 6, d_ff: int = 2048, max_len: int = 512, num_domains: int = 4):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Embeddings
        self.token_embedding = torch.nn.Embedding(vocab_size, d_model)
        self.position_embedding = torch.nn.Embedding(max_len, d_model)
        
        # Transformer layers
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = torch.nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output heads
        self.lm_head = torch.nn.Linear(d_model, vocab_size)
        self.domain_head = torch.nn.Linear(d_model, num_domains)
        
        self.dropout = torch.nn.Dropout(0.1)
    
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


def load_model_and_test():
    """Load and test the trained model."""
    
    model_path = 'checkpoints/best_model_simple.pth'
    
    if not os.path.exists(model_path):
        print("❌ Model not found. Please train the model first:")
        print("   python simple_train.py")
        return
    
    print("🚀 Loading trained model...")
    
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu')
    config = checkpoint['config']
    vocab = checkpoint['vocab']
    
    # Create model
    model = SimpleTransformer(**config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Create reverse vocab for decoding
    id_to_token = {v: k for k, v in vocab.items()}
    
    print("✅ Model loaded successfully!")
    print(f"📊 Vocabulary size: {len(vocab)}")
    print(f"🤖 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test queries
    test_queries = [
        "Create a blue circle that rotates",
        "Explain binary search algorithm", 
        "What is the singleton pattern?",
        "Draw a sine wave animation"
    ]
    
    print("\n🧪 Testing model with sample queries...")
    print("=" * 60)
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n🔹 Test {i}: {query}")
        print("-" * 40)
        
        # Simple tokenization
        words = query.lower().split()
        input_tokens = [vocab['<bos>']] if '<bos>' in vocab else [2]  # BOS token
        
        for word in words:
            if word in vocab:
                input_tokens.append(vocab[word])
            else:
                input_tokens.append(vocab['<unk>'] if '<unk>' in vocab else 1)  # UNK token
        
        # Pad to reasonable length
        max_len = 50
        if len(input_tokens) < max_len:
            input_tokens.extend([0] * (max_len - len(input_tokens)))  # PAD token
        
        input_tensor = torch.tensor([input_tokens[:max_len]], dtype=torch.long)
        
        # Generate response
        with torch.no_grad():
            outputs = model(input_tensor, return_domain_logits=True)
            
            # Get domain prediction
            domain_logits = outputs['domain_logits']
            domain_pred = torch.argmax(domain_logits, dim=1).item()
            domain_names = ['Manim', 'DSA', 'System Design', 'General']
            
            print(f"🎯 Predicted domain: {domain_names[domain_pred]}")
            
            # Simple greedy generation
            logits = outputs['logits'][0]  # First (and only) batch item
            
            # Take the last few predictions
            predicted_tokens = []
            for pos in range(min(10, logits.shape[0])):
                token_id = torch.argmax(logits[pos]).item()
                if token_id in id_to_token and token_id not in [0, 1, 2, 3]:  # Skip special tokens
                    predicted_tokens.append(id_to_token[token_id])
            
            if predicted_tokens:
                response = ' '.join(predicted_tokens[:20])  # Limit response length
                print(f"📝 Generated response: {response}")
            else:
                print("📝 Generated response: [Model needs more training]")
            
            # Domain confidence
            domain_probs = F.softmax(domain_logits, dim=1)[0]
            print(f"🔍 Domain confidence: {domain_probs[domain_pred]:.3f}")
    
    print("\n✅ Model testing completed!")
    print("\n💡 Note: This is a simple implementation for demonstration.")
    print("For better results, consider:")
    print("- More training data")
    print("- Longer training time") 
    print("- Better tokenization")
    print("- Beam search decoding")


if __name__ == "__main__":
    load_model_and_test()