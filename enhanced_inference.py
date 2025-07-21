"""
Enhanced Inference Pipeline
Handles multi-domain queries (Manim, DSA, System Design) with domain detection and routing.
Maintains backward compatibility with existing Manim inference.
"""

import torch
import pickle
import json
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

# Import model components
from enhanced_model import MultiDomainLLM, MultiDomainLLMConfig, Domain
from multi_domain_knowledge_base import MultiDomainKnowledgeBase, get_knowledge_base
from tokenizer import Tokenizer

class EnhancedInferenceEngine:
    """Enhanced inference engine supporting multiple CS domains."""
    
    def __init__(self, model_path: str, tokenizer_path: str, config_path: Optional[str] = None,
                 knowledge_base: Optional[MultiDomainKnowledgeBase] = None):
        """
        Initialize the enhanced inference engine.
        
        Args:
            model_path: Path to the trained model file
            tokenizer_path: Path to the tokenizer file
            config_path: Path to model configuration (optional)
            knowledge_base: Multi-domain knowledge base (optional, will create if None)
        """
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Load tokenizer
        self.tokenizer = self._load_tokenizer(tokenizer_path)
        print(f"Loaded tokenizer with vocab size: {self.tokenizer.vocab_size}")
        
        # Load model configuration
        if config_path and Path(config_path).exists():
            self.config = self._load_config(config_path)
        else:
            # Default configuration
            self.config = MultiDomainLLMConfig(
                vocab_size=self.tokenizer.vocab_size,
                d_model=512,
                n_heads=8,
                n_layers=6,
                enable_domain_adaptation=True
            )
        
        # Load model
        self.model = self._load_model(model_path)
        print(f"Loaded model with {self.model.get_model_size():,} parameters")
        
        # Initialize knowledge base
        self.knowledge_base = knowledge_base or get_knowledge_base()
        
        # Domain mapping
        self.domain_names = {
            0: 'manim',
            1: 'dsa', 
            2: 'system_design',
            3: 'general'
        }
        
        # Generation parameters
        self.generation_params = {
            'max_length': 512,
            'temperature': 0.8,
            'top_k': 50,
            'top_p': 0.9
        }
    
    def _load_tokenizer(self, tokenizer_path: str) -> Tokenizer:
        """Load the tokenizer from file."""
        with open(tokenizer_path, 'rb') as f:
            tokenizer = pickle.load(f)
        return tokenizer
    
    def _load_config(self, config_path: str) -> MultiDomainLLMConfig:
        """Load model configuration from file."""
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        return MultiDomainLLMConfig.from_dict(config_dict)
    
    def _load_model(self, model_path: str) -> MultiDomainLLM:
        """Load the trained model from file."""
        # Create model
        model = MultiDomainLLM(
            vocab_size=self.config.vocab_size,
            d_model=self.config.d_model,
            n_heads=self.config.n_heads,
            n_layers=self.config.n_layers,
            d_ff=self.config.d_ff,
            max_len=self.config.max_len,
            num_domains=self.config.num_domains,
            enable_domain_adaptation=self.config.enable_domain_adaptation
        )
        
        # Load state dict
        try:
            state_dict = torch.load(model_path, map_location=self.device)
            model.load_state_dict(state_dict, strict=False)
            print("Model loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load model state dict: {e}")
            print("Using randomly initialized model")
        
        model.to(self.device)
        model.eval()
        
        return model
    
    def preprocess_query(self, query: str) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Preprocess query and detect domain.
        
        Args:
            query: Input query string
            
        Returns:
            Tuple of (tokenized_input, query_info)
        """
        # Detect domain using knowledge base
        detected_domain, confidence = self.knowledge_base.detect_domain(query)
        
        # Search for relevant context
        search_results = self.knowledge_base.search_topics(query, max_results=3)
        
        # Create enhanced prompt based on domain
        enhanced_query = self._create_enhanced_prompt(query, detected_domain, search_results)
        
        # Tokenize
        input_ids = self.tokenizer.encode(enhanced_query)
        input_tensor = torch.tensor([input_ids], dtype=torch.long).to(self.device)
        
        query_info = {
            'original_query': query,
            'enhanced_query': enhanced_query,
            'detected_domain': detected_domain.value,
            'confidence': confidence,
            'search_results': search_results
        }
        
        return input_tensor, query_info
    
    def _create_enhanced_prompt(self, query: str, domain: Any, search_results: List[Dict]) -> str:
        """Create enhanced prompt based on detected domain and search results."""
        
        # Base prompt templates by domain
        domain_prompts = {
            'manim': f"Create a Manim animation script for: {query}\n\nScript:",
            'dsa': f"Explain the data structure/algorithm concept: {query}\n\nExplanation:",
            'system_design': f"Explain the system design concept: {query}\n\nDesign:",
            'unknown': f"Answer the computer science question: {query}\n\nAnswer:"
        }
        
        # Get domain-specific prompt
        domain_name = domain.value if hasattr(domain, 'value') else str(domain)
        base_prompt = domain_prompts.get(domain_name, domain_prompts['unknown'])
        
        # Add context from search results if available
        if search_results:
            context_info = []
            for result in search_results[:2]:  # Use top 2 results
                topic = result.get('topic', '').replace('_', ' ').title()
                context_info.append(f"Related: {topic}")
            
            if context_info:
                context = " | ".join(context_info)
                base_prompt = f"Context: {context}\n\n{base_prompt}"
        
        return base_prompt
    
    def generate_response(self, query: str, **generation_kwargs) -> Dict[str, Any]:
        """
        Generate response for a query with domain awareness.
        
        Args:
            query: Input query string
            **generation_kwargs: Override generation parameters
            
        Returns:
            Dictionary containing response and metadata
        """
        # Preprocess query
        input_tensor, query_info = self.preprocess_query(query)
        
        # Update generation parameters
        gen_params = {**self.generation_params, **generation_kwargs}
        
        # Generate response
        with torch.no_grad():
            generation_result = self.model.generate(
                input_tensor,
                max_length=gen_params['max_length'],
                temperature=gen_params['temperature'],
                top_k=gen_params['top_k'],
                top_p=gen_params['top_p'],
                use_domain_adaptation=True
            )
            
            generated_ids = generation_result['generated_ids']
            domain_sequence = generation_result['domain_sequence']
        
        # Decode response
        generated_text = self.tokenizer.decode(generated_ids[0].cpu().tolist())
        
        # Extract actual response (remove prompt)
        enhanced_query = query_info['enhanced_query']
        if generated_text.startswith(enhanced_query):
            response_text = generated_text[len(enhanced_query):].strip()
        else:
            response_text = generated_text.strip()
        
        # Analyze domain predictions during generation
        domain_analysis = self._analyze_domain_sequence(domain_sequence)
        
        # Get knowledge base suggestions
        kb_response = self.knowledge_base.generate_response(query)
        
        return {
            'query': query,
            'response': response_text,
            'domain_info': {
                'detected_domain': query_info['detected_domain'],
                'confidence': query_info['confidence'],
                'generation_domains': domain_analysis
            },
            'knowledge_base_results': kb_response['results'],
            'suggestions': kb_response['suggestions'],
            'metadata': {
                'model_used': 'MultiDomainLLM',
                'generation_params': gen_params,
                'input_length': len(enhanced_query),
                'output_length': len(response_text)
            }
        }
    
    def _analyze_domain_sequence(self, domain_sequence: List) -> Dict[str, Any]:
        """Analyze the sequence of domains predicted during generation."""
        if not domain_sequence:
            return {'primary_domain': 'unknown', 'domain_distribution': {}}
        
        # Flatten domain predictions
        domains = []
        for step in domain_sequence:
            if isinstance(step, (list, tuple)):
                domains.extend(step)
            else:
                domains.append(step)
        
        # Count domain occurrences
        domain_counts = {}
        for domain_idx in domains:
            domain_name = self.domain_names.get(int(domain_idx), 'unknown')
            domain_counts[domain_name] = domain_counts.get(domain_name, 0) + 1
        
        # Calculate distribution
        total_predictions = sum(domain_counts.values())
        domain_distribution = {
            domain: count / total_predictions 
            for domain, count in domain_counts.items()
        } if total_predictions > 0 else {}
        
        # Find primary domain
        primary_domain = max(domain_distribution, key=domain_distribution.get) if domain_distribution else 'unknown'
        
        return {
            'primary_domain': primary_domain,
            'domain_distribution': domain_distribution,
            'total_tokens': total_predictions
        }
    
    def batch_generate(self, queries: List[str], **generation_kwargs) -> List[Dict[str, Any]]:
        """Generate responses for multiple queries."""
        results = []
        for query in queries:
            try:
                result = self.generate_response(query, **generation_kwargs)
                results.append(result)
            except Exception as e:
                results.append({
                    'query': query,
                    'error': str(e),
                    'response': None
                })
        return results
    
    def get_domain_specific_examples(self, domain: str, num_examples: int = 5) -> List[str]:
        """Get example queries for a specific domain."""
        examples = {
            'manim': [
                "Create a blue circle that moves in a figure-8 pattern",
                "Show the mathematical formula E = mc²",
                "Animate a transformation from circle to square",
                "Create text that fades in and says 'Hello Manim'",
                "Draw a coordinate system with a moving dot"
            ],
            'dsa': [
                "Explain binary search algorithm",
                "Implement merge sort in Python",
                "What is the time complexity of quicksort?",
                "How does a hash table work?",
                "Show binary tree traversal methods"
            ],
            'system_design': [
                "Explain singleton design pattern",
                "How does a load balancer work?",
                "Design a scalable chat application",
                "What is microservices architecture?",
                "Explain database sharding"
            ]
        }
        
        return examples.get(domain.lower(), [])[:num_examples]
    
    def save_config(self, config_path: str):
        """Save current model configuration."""
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)
        print(f"Configuration saved to {config_path}")

# Convenience functions for backward compatibility
def create_inference_engine(model_path: str, tokenizer_path: str, 
                          config_path: Optional[str] = None) -> EnhancedInferenceEngine:
    """Create enhanced inference engine."""
    return EnhancedInferenceEngine(model_path, tokenizer_path, config_path)

def generate_response(query: str, model_path: str, tokenizer_path: str, **kwargs) -> str:
    """Quick response generation (simplified interface)."""
    engine = create_inference_engine(model_path, tokenizer_path)
    result = engine.generate_response(query, **kwargs)
    return result['response']

if __name__ == "__main__":
    # Test the enhanced inference engine
    
    # Note: These paths would need to point to actual trained models
    model_path = "best_model_epoch_10.pth"  # This should be an enhanced model
    tokenizer_path = "tokenizer.pkl"
    
    try:
        # Create inference engine
        engine = EnhancedInferenceEngine(model_path, tokenizer_path)
        
        # Test queries from different domains
        test_queries = [
            "Create a blue circle animation",  # Manim
            "Explain binary search",  # DSA  
            "What is singleton pattern?",  # System Design
            "How to optimize database queries?"  # General
        ]
        
        print("=== Enhanced Inference Engine Test ===")
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            
            # Generate response
            result = engine.generate_response(query, max_length=256)
            
            print(f"Detected Domain: {result['domain_info']['detected_domain']}")
            print(f"Confidence: {result['domain_info']['confidence']:.2f}")
            print(f"Response: {result['response'][:200]}...")
            
            if result['knowledge_base_results']:
                print(f"KB Results: {len(result['knowledge_base_results'])} found")
        
        # Test domain-specific examples
        print("\n=== Domain-Specific Examples ===")
        for domain in ['manim', 'dsa', 'system_design']:
            examples = engine.get_domain_specific_examples(domain, 3)
            print(f"{domain.upper()}: {examples}")
            
    except FileNotFoundError as e:
        print(f"Model files not found: {e}")
        print("This is expected if you haven't trained the enhanced model yet.")
        print("The inference engine is ready to use once you train the model.")
    except Exception as e:
        print(f"Error testing inference engine: {e}")
        print("This might be due to model compatibility issues.")