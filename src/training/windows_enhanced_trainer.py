"""
Windows-Compatible Enhanced Multi-Domain Model Trainer
Trains the enhanced model with multi-domain data and domain adaptation
Fixed for Windows console encoding issues
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
import time
import os
from typing import List, Dict, Tuple, Optional

# Import enhanced components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.enhanced_model import MultiDomainLLM, MultiDomainLLMConfig, Domain
from models.windows_enhanced_tokenizer import EnhancedManimTokenizer
from knowledge.multi_domain_knowledge_base import get_knowledge_base

class EnhancedManimDataset(Dataset):
    """Enhanced dataset supporting multi-domain training."""
    
    def __init__(self, data_file: str, tokenizer: EnhancedManimTokenizer, 
                 max_length: int = 512, include_domain_labels: bool = True):
        """
        Initialize enhanced dataset.
        
        Args:
            data_file: Path to training data JSON
            tokenizer: Enhanced tokenizer
            max_length: Maximum sequence length
            include_domain_labels: Whether to include domain classification labels
        """
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_domain_labels = include_domain_labels
        
        # Load training data
        print(f"Loading training data from {data_file}...")
        with open(data_file, 'r') as f:
            self.data = json.load(f)
        
        print(f"Loaded {len(self.data)} samples")
        
        # Domain mapping
        self.domain_map = {
            'manim': Domain.MANIM.value,
            'dsa': Domain.DSA.value,
            'system_design': Domain.SYSTEM_DESIGN.value
        }
        
        # Initialize knowledge base for domain detection
        if include_domain_labels:
            self.kb = get_knowledge_base()
        
        # Preprocess data
        self.processed_data = self._preprocess_data()
        
    def _preprocess_data(self) -> List[Dict]:
        """Preprocess training data."""
        processed = []
        
        print("Processing training data...")
        
        for i, item in enumerate(self.data):
            if i % 1000 == 0:
                print(f"Processed {i}/{len(self.data)} samples")
                
            request = item.get('request', '')
            response = item.get('response', '')
            domain = item.get('domain', 'manim')
            
            # Create input-output pair
            input_text = f"Query: {request}\nResponse:"
            target_text = response
            
            # Tokenize
            input_ids = self.tokenizer.encode(input_text)
            target_ids = self.tokenizer.encode(target_text)
            
            # Combine input and target
            full_sequence = input_ids + target_ids + [self.tokenizer.special_tokens['<EOS>']]
            
            # Truncate if too long
            if len(full_sequence) > self.max_length:
                full_sequence = full_sequence[:self.max_length]
            
            # Pad if too short
            while len(full_sequence) < self.max_length:
                full_sequence.append(self.tokenizer.special_tokens['<PAD>'])
            
            # Create labels (for language modeling, labels are shifted by 1)
            labels = full_sequence[1:] + [self.tokenizer.special_tokens['<PAD>']]
            
            # Get domain label
            domain_label = self.domain_map.get(domain, Domain.MANIM.value)
            
            # Detect domain confidence if needed
            domain_confidence = 1.0
            if self.include_domain_labels and hasattr(self, 'kb'):
                try:
                    detected_domain, confidence = self.kb.detect_domain(request)
                    domain_confidence = confidence
                except:
                    domain_confidence = 1.0
            
            processed_item = {
                'input_ids': torch.tensor(full_sequence, dtype=torch.long),
                'labels': torch.tensor(labels, dtype=torch.long),
                'domain_label': torch.tensor(domain_label, dtype=torch.long),
                'domain_confidence': torch.tensor(domain_confidence, dtype=torch.float),
                'input_length': len(input_ids),
                'target_length': len(target_ids),
                'original_request': request,
                'original_response': response
            }
            
            processed.append(processed_item)
        
        print(f"Preprocessed {len(processed)} samples")
        return processed
    
    def __len__(self) -> int:
        return len(self.processed_data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.processed_data[idx]

class EnhancedModelTrainer:
    """Enhanced trainer for multi-domain model."""
    
    def __init__(self, config: MultiDomainLLMConfig, tokenizer: EnhancedManimTokenizer,
                 device: str = "auto"):
        """
        Initialize enhanced trainer.
        
        Args:
            config: Model configuration
            tokenizer: Enhanced tokenizer
            device: Device to use
        """
        # Set device
        if device == "auto":
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Using device: {self.device}")
        
        self.config = config
        self.tokenizer = tokenizer
        
        # Update config with tokenizer vocab size
        self.config.vocab_size = tokenizer.get_vocab_size()
        
        # Initialize model
        self.model = MultiDomainLLM(
            vocab_size=self.config.vocab_size,
            d_model=self.config.d_model,
            n_heads=self.config.n_heads,
            n_layers=self.config.n_layers,
            d_ff=self.config.d_ff,
            max_len=self.config.max_len,
            num_domains=self.config.num_domains,
            enable_domain_adaptation=self.config.enable_domain_adaptation
        )
        
        self.model.to(self.device)
        
        print(f"Model created with {self.model.get_model_size():,} parameters")
        
        # Training tracking
        self.train_losses = []
        self.val_losses = []
        self.domain_accuracies = []
        
    def create_data_loaders(self, train_data_file: str, val_data_file: str, 
                          batch_size: int = 8, max_length: int = 512) -> Tuple[DataLoader, DataLoader]:
        """Create training and validation data loaders."""
        
        # Create datasets
        train_dataset = EnhancedManimDataset(
            train_data_file, self.tokenizer, max_length, include_domain_labels=True
        )
        
        val_dataset = EnhancedManimDataset(
            val_data_file, self.tokenizer, max_length, include_domain_labels=True
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0,  # Set to 0 to avoid multiprocessing issues
            pin_memory=torch.cuda.is_available()
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0,
            pin_memory=torch.cuda.is_available()
        )
        
        print(f"Train samples: {len(train_dataset)}")
        print(f"Val samples: {len(val_dataset)}")
        print(f"Batch size: {batch_size}")
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        
        return train_loader, val_loader
    
    def train_epoch(self, train_loader: DataLoader, optimizer: optim.Optimizer, 
                   criterion: nn.Module, domain_criterion: nn.Module, 
                   epoch: int) -> Tuple[float, float, float]:
        """Train one epoch."""
        
        self.model.train()
        total_loss = 0.0
        total_lm_loss = 0.0
        total_domain_loss = 0.0
        num_batches = len(train_loader)
        
        print(f"Training epoch {epoch+1}...")
        
        for batch_idx, batch in enumerate(train_loader):
            if batch_idx % 50 == 0:
                print(f"  Batch {batch_idx}/{num_batches}")
                
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            domain_labels = batch['domain_label'].to(self.device)
            
            # Forward pass
            optimizer.zero_grad()
            
            outputs = self.model(
                input_ids, 
                return_domain_logits=True,
                use_domain_specific_head=True
            )
            
            logits = outputs['logits']
            domain_logits = outputs['domain_logits']
            
            # Language modeling loss
            lm_loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
            
            # Domain classification loss
            domain_loss = domain_criterion(domain_logits, domain_labels)
            
            # Combined loss (weighted)
            loss = lm_loss + 0.1 * domain_loss  # Domain loss has lower weight
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Track losses
            total_loss += loss.item()
            total_lm_loss += lm_loss.item()
            total_domain_loss += domain_loss.item()
        
        avg_loss = total_loss / num_batches
        avg_lm_loss = total_lm_loss / num_batches
        avg_domain_loss = total_domain_loss / num_batches
        
        return avg_loss, avg_lm_loss, avg_domain_loss
    
    def validate_epoch(self, val_loader: DataLoader, criterion: nn.Module, 
                      domain_criterion: nn.Module) -> Tuple[float, float, float, float]:
        """Validate one epoch."""
        
        self.model.eval()
        total_loss = 0.0
        total_lm_loss = 0.0
        total_domain_loss = 0.0
        correct_domain_predictions = 0
        total_predictions = 0
        
        print("Validating...")
        
        with torch.no_grad():
            for batch in val_loader:
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                domain_labels = batch['domain_label'].to(self.device)
                
                # Forward pass
                outputs = self.model(
                    input_ids,
                    return_domain_logits=True,
                    use_domain_specific_head=True
                )
                
                logits = outputs['logits']
                domain_logits = outputs['domain_logits']
                
                # Calculate losses
                lm_loss = criterion(logits.view(-1, logits.size(-1)), labels.view(-1))
                domain_loss = domain_criterion(domain_logits, domain_labels)
                loss = lm_loss + 0.1 * domain_loss
                
                total_loss += loss.item()
                total_lm_loss += lm_loss.item()
                total_domain_loss += domain_loss.item()
                
                # Domain accuracy
                domain_predictions = torch.argmax(domain_logits, dim=1)
                correct_domain_predictions += (domain_predictions == domain_labels).sum().item()
                total_predictions += domain_labels.size(0)
        
        avg_loss = total_loss / len(val_loader)
        avg_lm_loss = total_lm_loss / len(val_loader)
        avg_domain_loss = total_domain_loss / len(val_loader)
        domain_accuracy = correct_domain_predictions / total_predictions
        
        return avg_loss, avg_lm_loss, avg_domain_loss, domain_accuracy
    
    def train(self, train_data_file: str, val_data_file: str, 
              num_epochs: int = 15, batch_size: int = 8, learning_rate: float = 1e-4,
              save_dir: str = "enhanced_model_checkpoints") -> None:
        """Train the enhanced model."""
        
        print("Starting Enhanced Multi-Domain Training")
        print("=" * 60)
        
        # Create save directory
        os.makedirs(save_dir, exist_ok=True)
        
        # Create data loaders
        train_loader, val_loader = self.create_data_loaders(
            train_data_file, val_data_file, batch_size
        )
        
        # Initialize optimizer and loss functions
        optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate,
            weight_decay=0.01
        )
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_epochs
        )
        
        # Loss functions
        criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.special_tokens['<PAD>'])
        domain_criterion = nn.CrossEntropyLoss()
        
        print(f"Training for {num_epochs} epochs")
        print(f"Learning rate: {learning_rate}")
        print(f"Optimizer: AdamW with weight decay")
        
        best_val_loss = float('inf')
        patience = 5
        patience_counter = 0
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            print("-" * 40)
            
            # Training
            start_time = time.time()
            train_loss, train_lm_loss, train_domain_loss = self.train_epoch(
                train_loader, optimizer, criterion, domain_criterion, epoch
            )
            
            # Validation
            val_loss, val_lm_loss, val_domain_loss, domain_accuracy = self.validate_epoch(
                val_loader, criterion, domain_criterion
            )
            
            # Update scheduler
            scheduler.step()
            
            epoch_time = time.time() - start_time
            
            # Log results
            print(f"Epoch time: {epoch_time:.2f}s")
            print(f"Train Loss: {train_loss:.4f} (LM: {train_lm_loss:.4f}, Domain: {train_domain_loss:.4f})")
            print(f"Val Loss: {val_loss:.4f} (LM: {val_lm_loss:.4f}, Domain: {val_domain_loss:.4f})")
            print(f"Domain Accuracy: {domain_accuracy:.3f}")
            print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
            
            # Track metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.domain_accuracies.append(domain_accuracy)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                
                # Save model
                model_path = os.path.join(save_dir, f"enhanced_best_model.pth")
                torch.save(self.model.state_dict(), model_path)
                
                # Save config
                config_path = os.path.join(save_dir, "enhanced_model_config.json")
                with open(config_path, 'w') as f:
                    json.dump(self.config.to_dict(), f, indent=2)
                
                print(f"Best model saved! (Val Loss: {val_loss:.4f})")
            else:
                patience_counter += 1
                print(f"No improvement for {patience_counter}/{patience} epochs")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
            
            # Save checkpoint every 5 epochs
            if (epoch + 1) % 5 == 0:
                checkpoint_path = os.path.join(save_dir, f"enhanced_model_epoch_{epoch + 1}.pth")
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"Checkpoint saved: {checkpoint_path}")
        
        print("\nTraining completed!")
        print(f"Best validation loss: {best_val_loss:.4f}")

def main():
    """Main training function."""
    
    print("Enhanced Multi-Domain LLM Training")
    print("=" * 60)
    
    # Check if training data exists
    if not os.path.exists("enhanced_train_data.json"):
        print("ERROR: Training data not found. Please run:")
        print("   python start_training.py")
        return
    
    if not os.path.exists("enhanced_tokenizer.pkl"):
        print("ERROR: Enhanced tokenizer not found. Please run:")
        print("   python enhanced_tokenizer.py")
        return
    
    # Load tokenizer
    print("Loading enhanced tokenizer...")
    tokenizer = EnhancedManimTokenizer.load("enhanced_tokenizer.pkl")
    
    # Create enhanced model configuration
    config = MultiDomainLLMConfig(
        vocab_size=tokenizer.get_vocab_size(),
        d_model=512,
        n_heads=8,
        n_layers=8,  # Increased for better performance
        d_ff=2048,
        max_len=512,
        num_domains=4,
        enable_domain_adaptation=True
    )
    
    print(f"Model Configuration:")
    print(f"   Vocab Size: {config.vocab_size}")
    print(f"   Model Dim: {config.d_model}")
    print(f"   Layers: {config.n_layers}")
    print(f"   Heads: {config.n_heads}")
    print(f"   Domain Adaptation: {config.enable_domain_adaptation}")
    
    # Create trainer
    trainer = EnhancedModelTrainer(config, tokenizer)
    
    # Start training
    trainer.train(
        train_data_file="enhanced_train_data.json",
        val_data_file="enhanced_val_data.json",
        num_epochs=15,
        batch_size=2,  # Reduced for memory efficiency (CPU training)
        learning_rate=1e-4
    )
    
    print("\nTraining completed successfully!")
    print("To use the trained model:")
    print("   1. Copy enhanced_model_checkpoints/enhanced_best_model.pth to your working directory")
    print("   2. Copy enhanced_tokenizer.pkl if not already present")
    print("   3. Run: python fixed_api_server.py")

if __name__ == "__main__":
    main()