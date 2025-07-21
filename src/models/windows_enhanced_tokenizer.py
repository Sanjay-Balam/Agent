"""
Enhanced Multi-Domain Tokenizer (Windows Compatible)
Extended tokenizer to handle Manim, DSA, and System Design vocabularies
Windows-compatible version without Unicode emoji characters
"""

import json
import pickle
from collections import Counter
from typing import List, Dict, Set, Tuple
import re

class EnhancedManimTokenizer:
    """Enhanced tokenizer supporting multiple CS domains."""
    
    def __init__(self, vocab_size: int = 15000):
        """
        Initialize enhanced tokenizer.
        
        Args:
            vocab_size: Maximum vocabulary size
        """
        self.vocab_size = vocab_size
        self.word_to_id = {}
        self.id_to_word = {}
        self.special_tokens = {
            '<PAD>': 0,
            '<UNK>': 1, 
            '<SOS>': 2,
            '<EOS>': 3,
            '<MANIM>': 4,     # Domain tokens
            '<DSA>': 5,
            '<SYSDES>': 6,
            '<CODE>': 7,
            '<COMPLEXITY>': 8,
            '<PATTERN>': 9
        }
        
        # Domain-specific vocabulary
        self.domain_vocabularies = {
            'manim': self._get_manim_vocabulary(),
            'dsa': self._get_dsa_vocabulary(), 
            'system_design': self._get_system_design_vocabulary(),
            'programming': self._get_programming_vocabulary()
        }
        
        # Initialize with special tokens
        for token, idx in self.special_tokens.items():
            self.word_to_id[token] = idx
            self.id_to_word[idx] = token
    
    def _get_manim_vocabulary(self) -> Set[str]:
        """Get Manim-specific vocabulary."""
        return {
            # Core Manim
            'manim', 'scene', 'construct', 'animation', 'mobject',
            'play', 'wait', 'add', 'remove', 'transform',
            
            # Objects
            'circle', 'square', 'rectangle', 'line', 'arrow', 'dot',
            'text', 'mathtex', 'tex', 'polygon', 'arc', 'sector',
            
            # Animations  
            'create', 'write', 'fadein', 'fadeout', 'rotate', 'scale',
            'shift', 'move_to', 'animate', 'succession', 'lag_ratio',
            
            # Colors
            'blue', 'red', 'green', 'yellow', 'orange', 'purple', 'pink',
            'white', 'black', 'gray', 'light_gray', 'dark_gray',
            
            # Math
            'pi', 'tau', 'sin', 'cos', 'tan', 'log', 'exp', 'sqrt',
            'integral', 'derivative', 'limit', 'matrix', 'vector',
            
            # Positions
            'up', 'down', 'left', 'right', 'origin', 'center',
            'ul', 'ur', 'dl', 'dr'
        }
    
    def _get_dsa_vocabulary(self) -> Set[str]:
        """Get DSA-specific vocabulary."""
        return {
            # Data Structures
            'array', 'list', 'stack', 'queue', 'tree', 'graph',
            'heap', 'hash', 'table', 'linked', 'binary', 'node',
            'pointer', 'reference', 'index', 'key', 'value',
            
            # Tree Terms
            'root', 'leaf', 'parent', 'child', 'subtree', 'depth',
            'height', 'level', 'balanced', 'bst', 'avl', 'btree',
            
            # Graph Terms
            'vertex', 'edge', 'directed', 'undirected', 'weighted',
            'path', 'cycle', 'connected', 'component', 'traversal',
            'dfs', 'bfs', 'dijkstra', 'bellman', 'ford', 'kruskal',
            
            # Algorithms
            'sort', 'search', 'merge', 'quick', 'bubble', 'insertion',
            'selection', 'heap', 'radix', 'counting', 'bucket',
            'binary', 'linear', 'recursive', 'iterative', 'dynamic',
            'programming', 'greedy', 'backtrack', 'divide', 'conquer',
            
            # Complexity
            'time', 'space', 'complexity', 'big', 'omega', 'theta',
            'log', 'linear', 'quadratic', 'cubic', 'exponential',
            'polynomial', 'factorial', 'amortized', 'worst', 'best',
            'average', 'case'
        }
    
    def _get_system_design_vocabulary(self) -> Set[str]:
        """Get System Design vocabulary.""" 
        return {
            # Design Patterns
            'singleton', 'factory', 'observer', 'strategy', 'command',
            'adapter', 'decorator', 'facade', 'proxy', 'builder',
            'prototype', 'template', 'method', 'state', 'visitor',
            
            # Architecture
            'microservices', 'monolith', 'api', 'rest', 'graphql',
            'gateway', 'proxy', 'load', 'balancer', 'cache', 'cdn',
            'database', 'nosql', 'sql', 'acid', 'cap', 'theorem',
            
            # Scalability  
            'horizontal', 'vertical', 'scaling', 'sharding', 'partition',
            'replication', 'master', 'slave', 'consensus', 'raft',
            'eventual', 'consistency', 'availability', 'partition',
            'tolerance', 'latency', 'throughput', 'bottleneck',
            
            # Infrastructure
            'container', 'docker', 'kubernetes', 'orchestration',
            'deployment', 'pipeline', 'monitoring', 'logging',
            'metrics', 'alerting', 'circuit', 'breaker', 'retry',
            'timeout', 'bulkhead', 'rate', 'limiting',
            
            # Messaging
            'queue', 'topic', 'producer', 'consumer', 'kafka',
            'rabbitmq', 'pubsub', 'event', 'sourcing', 'cqrs',
            'saga', 'choreography', 'orchestration'
        }
    
    def _get_programming_vocabulary(self) -> Set[str]:
        """Get general programming vocabulary."""
        return {
            # Python Keywords
            'def', 'class', 'if', 'else', 'elif', 'for', 'while',
            'try', 'except', 'finally', 'with', 'import', 'from',
            'return', 'yield', 'lambda', 'self', 'super', 'init',
            
            # Common Functions
            'print', 'len', 'range', 'enumerate', 'zip', 'map',
            'filter', 'reduce', 'sorted', 'reversed', 'sum', 'max',
            'min', 'abs', 'round', 'isinstance', 'hasattr', 'getattr',
            
            # Data Types
            'int', 'float', 'str', 'bool', 'list', 'dict', 'set',
            'tuple', 'none', 'true', 'false', 'object', 'type',
            
            # Operators
            'and', 'or', 'not', 'in', 'is', 'append', 'extend',
            'insert', 'remove', 'pop', 'clear', 'copy', 'update',
            
            # Common Terms
            'function', 'method', 'variable', 'parameter', 'argument',
            'instance', 'attribute', 'property', 'module', 'package',
            'exception', 'error', 'debug', 'test', 'assert'
        }
    
    def build_vocabulary(self, texts: List[str]) -> None:
        """
        Build vocabulary from training texts.
        
        Args:
            texts: List of training texts
        """
        print("Building enhanced multi-domain vocabulary...")
        
        # Collect all words
        word_counter = Counter()
        
        for text in texts:
            words = self.tokenize_text(text)
            word_counter.update(words)
        
        # Add domain-specific vocabularies with high priority
        domain_words = set()
        for domain, vocab in self.domain_vocabularies.items():
            domain_words.update(vocab)
            # Give domain words high counts to ensure inclusion
            for word in vocab:
                if word in word_counter:
                    word_counter[word] += 100  # Boost domain words
                else:
                    word_counter[word] = 100
        
        print(f"Domain vocabulary: {len(domain_words)} words")
        print(f"Total unique words: {len(word_counter)}")
        
        # Select top words (excluding special tokens)
        available_slots = self.vocab_size - len(self.special_tokens)
        most_common = word_counter.most_common(available_slots)
        
        # Build vocabulary mappings
        current_id = len(self.special_tokens)
        
        for word, count in most_common:
            if word not in self.word_to_id:
                self.word_to_id[word] = current_id
                self.id_to_word[current_id] = word
                current_id += 1
        
        print(f"Vocabulary built: {len(self.word_to_id)} tokens")
        
        # Show domain coverage
        for domain, vocab in self.domain_vocabularies.items():
            covered = sum(1 for word in vocab if word in self.word_to_id)
            coverage = covered / len(vocab) * 100
            print(f"{domain.title()} coverage: {covered}/{len(vocab)} ({coverage:.1f}%)")
    
    def tokenize_text(self, text: str) -> List[str]:
        """
        Tokenize text into words.
        
        Args:
            text: Input text
            
        Returns:
            List of tokens
        """
        # Normalize text
        text = text.lower()
        
        # Handle code patterns
        text = re.sub(r'([(){}[\].,;:])', r' \1 ', text)  # Separate punctuation
        text = re.sub(r'([<>=!+\-*/])', r' \1 ', text)   # Separate operators
        text = re.sub(r'\s+', ' ', text)                  # Normalize whitespace
        
        # Split into tokens
        tokens = text.split()
        
        # Filter out empty tokens
        tokens = [token for token in tokens if token.strip()]
        
        return tokens
    
    def encode(self, text: str) -> List[int]:
        """
        Encode text to token IDs.
        
        Args:
            text: Input text
            
        Returns:
            List of token IDs
        """
        tokens = self.tokenize_text(text)
        
        # Convert to IDs
        token_ids = []
        for token in tokens:
            if token in self.word_to_id:
                token_ids.append(self.word_to_id[token])
            else:
                token_ids.append(self.special_tokens['<UNK>'])  # Unknown token
        
        return token_ids
    
    def decode(self, token_ids: List[int]) -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs
            
        Returns:
            Decoded text
        """
        tokens = []
        for token_id in token_ids:
            if token_id in self.id_to_word:
                token = self.id_to_word[token_id]
                # Skip special tokens in output
                if token not in ['<PAD>', '<SOS>', '<EOS>', '<UNK>']:
                    tokens.append(token)
        
        # Join tokens and clean up
        text = ' '.join(tokens)
        
        # Clean up punctuation spacing
        text = re.sub(r'\s+([(){}[\].,;:])', r'\1', text)  # Remove space before punctuation
        text = re.sub(r'([(){}[\]])\s+', r'\1', text)      # Clean bracket spacing
        
        return text
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.word_to_id)
    
    def save(self, filepath: str) -> None:
        """
        Save tokenizer to file.
        
        Args:
            filepath: Path to save tokenizer
        """
        tokenizer_data = {
            'word_to_id': self.word_to_id,
            'id_to_word': self.id_to_word,
            'vocab_size': self.vocab_size,
            'special_tokens': self.special_tokens
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(tokenizer_data, f)
        
        print(f"Enhanced tokenizer saved to {filepath}")
        print(f"Vocabulary size: {self.get_vocab_size()}")
    
    @classmethod
    def load(cls, filepath: str) -> 'EnhancedManimTokenizer':
        """
        Load tokenizer from file.
        
        Args:
            filepath: Path to tokenizer file
            
        Returns:
            Loaded tokenizer
        """
        with open(filepath, 'rb') as f:
            tokenizer_data = pickle.load(f)
        
        # Create new instance
        tokenizer = cls(tokenizer_data['vocab_size'])
        tokenizer.word_to_id = tokenizer_data['word_to_id']
        tokenizer.id_to_word = tokenizer_data['id_to_word']
        tokenizer.special_tokens = tokenizer_data.get('special_tokens', tokenizer.special_tokens)
        
        print(f"Enhanced tokenizer loaded from {filepath}")
        print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
        
        return tokenizer

def build_enhanced_tokenizer_from_data(data_file: str, output_file: str = "enhanced_tokenizer.pkl") -> None:
    """
    Build enhanced tokenizer from training data.
    
    Args:
        data_file: Path to training data JSON
        output_file: Output tokenizer file
    """
    print("Building Enhanced Multi-Domain Tokenizer")
    print("=" * 50)
    
    # Load training data
    print(f"Loading training data from {data_file}...")
    with open(data_file, 'r') as f:
        data = json.load(f)
    
    # Extract all texts
    texts = []
    for item in data:
        # Add both request and response texts
        texts.append(item.get('request', ''))
        texts.append(item.get('response', ''))
    
    print(f"Loaded {len(data)} samples, {len(texts)} texts")
    
    # Build tokenizer
    tokenizer = EnhancedManimTokenizer(vocab_size=15000)
    tokenizer.build_vocabulary(texts)
    
    # Save tokenizer
    tokenizer.save(output_file)
    
    # Test tokenizer
    print("\nTesting tokenizer...")
    test_texts = [
        "Create a blue circle that rotates",
        "Implement binary search algorithm",
        "Explain singleton design pattern"
    ]
    
    for text in test_texts:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        print(f"Original: {text}")
        print(f"Encoded:  {encoded[:10]}... ({len(encoded)} tokens)")
        print(f"Decoded:  {decoded}")
        print()
    
    print("Enhanced tokenizer ready!")
    return tokenizer

if __name__ == "__main__":
    # Build enhanced tokenizer from training data
    build_enhanced_tokenizer_from_data(
        data_file="enhanced_training_data.json",
        output_file="enhanced_tokenizer.pkl"
    )