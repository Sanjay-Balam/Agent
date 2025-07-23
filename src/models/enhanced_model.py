"""
Enhanced Multi-Domain LLM Model
Extends the original ManimLLM to handle multiple CS domains (Manim, DSA, System Design)
while maintaining backward compatibility.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict, List
from enum import Enum

# Import original model components
from .model import (
    PositionalEncoding, MultiHeadAttention, FeedForward, 
    TransformerBlock, ManimLLM, ManimLLMConfig
)

class Domain(Enum):
    MANIM = 0
    DSA = 1
    SYSTEM_DESIGN = 2
    GENERAL = 3  # For mixed or unclear domains

class DomainClassifier(nn.Module):
    """Domain classification head to identify the topic domain."""
    
    def __init__(self, d_model: int, num_domains: int = 4):
        super().__init__()
        self.d_model = d_model
        self.num_domains = num_domains
        
        # Classification layers
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model // 4, num_domains)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Domain classification logits of shape (batch_size, num_domains)
        """
        # Use mean pooling over sequence dimension for classification
        pooled = torch.mean(x, dim=1)  # (batch_size, d_model)
        return self.classifier(pooled)

class DomainAdaptiveLayer(nn.Module):
    """Domain-adaptive layer that adjusts representations based on domain."""
    
    def __init__(self, d_model: int, num_domains: int = 4):
        super().__init__()
        self.d_model = d_model
        self.num_domains = num_domains
        
        # Domain-specific adaptation layers
        self.domain_adapters = nn.ModuleDict({
            str(i): nn.Sequential(
                nn.Linear(d_model, d_model),
                nn.LayerNorm(d_model),
                nn.ReLU(),
                nn.Linear(d_model, d_model)
            ) for i in range(num_domains)
        })
        
        # Gating mechanism to blend domain-specific and general representations
        self.gate = nn.Sequential(
            nn.Linear(d_model + num_domains, d_model),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor, domain_probs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            domain_probs: Domain probabilities of shape (batch_size, num_domains)
        Returns:
            Domain-adapted representations of shape (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Apply domain-specific adapters
        adapted_outputs = []
        for i in range(self.num_domains):
            adapted = self.domain_adapters[str(i)](x)  # (batch_size, seq_len, d_model)
            adapted_outputs.append(adapted)
        
        # Stack domain-specific outputs
        adapted_stack = torch.stack(adapted_outputs, dim=-1)  # (batch_size, seq_len, d_model, num_domains)
        
        # Weight by domain probabilities
        domain_weights = domain_probs.unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, num_domains)
        weighted_adapted = torch.sum(adapted_stack * domain_weights, dim=-1)  # (batch_size, seq_len, d_model)
        
        # Gating mechanism
        domain_probs_expanded = domain_probs.unsqueeze(1).expand(-1, seq_len, -1)  # (batch_size, seq_len, num_domains)
        gate_input = torch.cat([x, domain_probs_expanded], dim=-1)  # (batch_size, seq_len, d_model + num_domains)
        gate_weights = self.gate(gate_input)  # (batch_size, seq_len, d_model)
        
        # Blend original and adapted representations
        output = gate_weights * weighted_adapted + (1 - gate_weights) * x
        
        return output

class EnhancedTransformerBlock(nn.Module):
    """Enhanced transformer block with domain adaptation."""
    
    def __init__(self, d_model: int, n_heads: int = 8, d_ff: int = 2048, num_domains: int = 4):
        super().__init__()
        
        # Standard transformer components
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)
        
        # Domain adaptation
        self.domain_adaptive = DomainAdaptiveLayer(d_model, num_domains)
    
    def forward(self, x: torch.Tensor, domain_probs: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            domain_probs: Domain probabilities of shape (batch_size, num_domains)
            mask: Attention mask
        Returns:
            Enhanced transformer output
        """
        # Self-attention with residual connection
        attn_output = self.attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Domain adaptation
        adapted_x = self.domain_adaptive(x, domain_probs)
        x = self.norm2(x + adapted_x)
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x

class MultiDomainLLM(nn.Module):
    """Enhanced LLM that can handle multiple CS domains."""
    
    def __init__(self, vocab_size: int, d_model: int = 512, n_heads: int = 8, 
                 n_layers: int = 6, d_ff: int = 2048, max_len: int = 2048,
                 num_domains: int = 4, enable_domain_adaptation: bool = True):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.num_domains = num_domains
        self.enable_domain_adaptation = enable_domain_adaptation
        
        # Embedding layers
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_len)
        
        # Domain classifier
        self.domain_classifier = DomainClassifier(d_model, num_domains)
        
        # Enhanced transformer blocks
        if enable_domain_adaptation:
            self.transformer_blocks = nn.ModuleList([
                EnhancedTransformerBlock(d_model, n_heads, d_ff, num_domains) 
                for _ in range(n_layers)
            ])
        else:
            # Fallback to original transformer blocks
            self.transformer_blocks = nn.ModuleList([
                TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)
            ])
        
        # Output layers
        self.layer_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Domain-specific output heads (optional)
        self.domain_specific_heads = nn.ModuleDict({
            'manim': nn.Linear(d_model, vocab_size),
            'dsa': nn.Linear(d_model, vocab_size),
            'system_design': nn.Linear(d_model, vocab_size),
            'general': nn.Linear(d_model, vocab_size)
        })
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize model parameters."""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                return_domain_logits: bool = False, use_domain_specific_head: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass with domain awareness.
        
        Args:
            input_ids: Input token ids of shape (batch_size, seq_len)
            attention_mask: Attention mask of shape (batch_size, seq_len)
            return_domain_logits: Whether to return domain classification logits
            use_domain_specific_head: Whether to use domain-specific output heads
        
        Returns:
            Dictionary containing:
            - logits: Output logits of shape (batch_size, seq_len, vocab_size)
            - domain_logits: Domain classification logits (if requested)
            - domain_probs: Domain probabilities
        """
        batch_size, seq_len = input_ids.size()
        
        # Token embeddings
        embeddings = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        
        # Add positional encoding
        embeddings = self.positional_encoding(embeddings.transpose(0, 1)).transpose(0, 1)
        
        # Create causal mask for autoregressive generation
        causal_mask = self._create_causal_mask(seq_len).to(input_ids.device)
        
        # Combine with attention mask if provided
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            causal_mask = causal_mask & attention_mask
        
        # Domain classification
        domain_logits = self.domain_classifier(embeddings)
        domain_probs = F.softmax(domain_logits, dim=-1)
        
        # Pass through transformer blocks
        x = embeddings
        if self.enable_domain_adaptation:
            for block in self.transformer_blocks:
                x = block(x, domain_probs, causal_mask)
        else:
            for block in self.transformer_blocks:
                x = block(x, causal_mask)
        
        # Layer normalization
        x = self.layer_norm(x)
        
        # Output projection
        if use_domain_specific_head:
            # Use domain-specific heads weighted by domain probabilities
            domain_outputs = {}
            for domain_name, head in self.domain_specific_heads.items():
                domain_outputs[domain_name] = head(x)
            
            # Weight by domain probabilities
            domain_names = ['manim', 'dsa', 'system_design', 'general']
            weighted_logits = torch.zeros_like(domain_outputs['general'])
            
            for i, domain_name in enumerate(domain_names):
                domain_weight = domain_probs[:, i].unsqueeze(1).unsqueeze(2)
                weighted_logits += domain_weight * domain_outputs[domain_name]
            
            logits = weighted_logits
        else:
            logits = self.output_projection(x)
        
        result = {
            'logits': logits,
            'domain_probs': domain_probs
        }
        
        if return_domain_logits:
            result['domain_logits'] = domain_logits
        
        return result
    
    def _create_causal_mask(self, seq_len: int) -> torch.Tensor:
        """Create causal mask for autoregressive generation."""
        mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
        return mask == 0
    
    def generate(self, input_ids: torch.Tensor, max_length: int = 512, 
                temperature: float = 1.0, top_k: int = 50, top_p: float = 0.9,
                use_domain_adaptation: bool = True) -> Dict[str, torch.Tensor]:
        """Generate text with domain awareness."""
        self.eval()
        
        generated_domains = []
        
        with torch.no_grad():
            for _ in range(max_length - input_ids.size(1)):
                # Get predictions for next token
                outputs = self.forward(input_ids, use_domain_specific_head=use_domain_adaptation)
                logits = outputs['logits']
                domain_probs = outputs['domain_probs']
                
                # Store domain information
                predicted_domain = torch.argmax(domain_probs, dim=1)
                generated_domains.append(predicted_domain.cpu().numpy())
                
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply top-k filtering
                if top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k)
                    next_token_logits = torch.full_like(next_token_logits, -float('inf'))
                    next_token_logits.scatter_(1, top_k_indices, top_k_logits)
                
                # Apply top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = -float('inf')
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to sequence
                input_ids = torch.cat([input_ids, next_token], dim=1)
        
        return {
            'generated_ids': input_ids,
            'domain_sequence': generated_domains
        }
    
    def get_model_size(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())
    
    def get_trainable_parameters(self) -> int:
        """Get number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

class MultiDomainLLMConfig:
    """Configuration for MultiDomainLLM."""
    
    def __init__(self, vocab_size: int = 10000, d_model: int = 512, n_heads: int = 8,
                 n_layers: int = 6, d_ff: int = 2048, max_len: int = 2048,
                 num_domains: int = 4, enable_domain_adaptation: bool = True):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.d_ff = d_ff
        self.max_len = max_len
        self.num_domains = num_domains
        self.enable_domain_adaptation = enable_domain_adaptation
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return {
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'n_heads': self.n_heads,
            'n_layers': self.n_layers,
            'd_ff': self.d_ff,
            'max_len': self.max_len,
            'num_domains': self.num_domains,
            'enable_domain_adaptation': self.enable_domain_adaptation
        }
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> 'MultiDomainLLMConfig':
        """Create config from dictionary."""
        return cls(**config_dict)

def create_enhanced_model(config: MultiDomainLLMConfig) -> MultiDomainLLM:
    """Create enhanced multi-domain LLM from configuration."""
    model = MultiDomainLLM(
        vocab_size=config.vocab_size,
        d_model=config.d_model,
        n_heads=config.n_heads,
        n_layers=config.n_layers,
        d_ff=config.d_ff,
        max_len=config.max_len,
        num_domains=config.num_domains,
        enable_domain_adaptation=config.enable_domain_adaptation
    )
    
    print(f"Created MultiDomainLLM with {model.get_model_size():,} parameters")
    print(f"Domain adaptation: {'Enabled' if config.enable_domain_adaptation else 'Disabled'}")
    print(f"Number of domains: {config.num_domains}")
    
    return model

def migrate_from_manim_model(manim_model: ManimLLM, config: MultiDomainLLMConfig) -> MultiDomainLLM:
    """Migrate from original ManimLLM to MultiDomainLLM."""
    # Create new model
    enhanced_model = create_enhanced_model(config)
    
    # Copy compatible parameters
    state_dict = manim_model.state_dict()
    enhanced_state_dict = enhanced_model.state_dict()
    
    # Copy matching parameters
    for name, param in state_dict.items():
        if name in enhanced_state_dict and enhanced_state_dict[name].shape == param.shape:
            enhanced_state_dict[name] = param
            print(f"Copied parameter: {name}")
    
    enhanced_model.load_state_dict(enhanced_state_dict, strict=False)
    print("Migration completed. New domain-specific parameters initialized randomly.")
    
    return enhanced_model

if __name__ == "__main__":
    # Test the enhanced model
    config = MultiDomainLLMConfig(
        vocab_size=5000, 
        d_model=256, 
        n_layers=4,
        num_domains=4,
        enable_domain_adaptation=True
    )
    
    model = create_enhanced_model(config)
    
    # Test forward pass
    batch_size = 2
    seq_len = 100
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    with torch.no_grad():
        outputs = model(input_ids, return_domain_logits=True)
        
        print(f"Input shape: {input_ids.shape}")
        print(f"Output logits shape: {outputs['logits'].shape}")
        print(f"Domain logits shape: {outputs['domain_logits'].shape}")
        print(f"Domain probabilities shape: {outputs['domain_probs'].shape}")
        print(f"Model parameters: {model.get_model_size():,}")
        
        # Test generation with domain tracking
        prompt = torch.randint(0, config.vocab_size, (1, 10))
        generated = model.generate(prompt, max_length=50, use_domain_adaptation=True)
        print(f"Generated sequence length: {generated['generated_ids'].shape[1]}")
        print(f"Domain sequence length: {len(generated['domain_sequence'])}")
        
        # Show domain predictions
        domain_names = ['MANIM', 'DSA', 'SYSTEM_DESIGN', 'GENERAL']
        for i, domain_idx in enumerate(generated['domain_sequence'][:5]):  # Show first 5
            domain_name = domain_names[domain_idx[0]]
            print(f"Token {i+1}: Predicted domain = {domain_name}")