"""
Enhanced Model Evaluator
Evaluates trained multi-domain model performance across all domains
"""

import torch
import json
import time
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import numpy as np

from enhanced_model import MultiDomainLLM, MultiDomainLLMConfig
from enhanced_tokenizer import EnhancedManimTokenizer
from enhanced_inference import EnhancedInferenceEngine
from validator import ManimScriptValidator
from multi_domain_knowledge_base import Domain, get_knowledge_base

class EnhancedModelEvaluator:
    """Comprehensive evaluator for enhanced multi-domain model."""
    
    def __init__(self, model_path: str, tokenizer_path: str, config_path: Optional[str] = None):
        """
        Initialize evaluator.
        
        Args:
            model_path: Path to trained model
            tokenizer_path: Path to tokenizer
            config_path: Path to model config (optional)
        """
        print("🔍 Initializing Enhanced Model Evaluator")
        print("=" * 50)
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"🖥️ Using device: {self.device}")
        
        # Initialize inference engine
        try:
            self.inference_engine = EnhancedInferenceEngine(
                model_path, tokenizer_path, config_path
            )
            print("✅ Enhanced inference engine loaded")
        except Exception as e:
            print(f"❌ Failed to load inference engine: {e}")
            raise
        
        # Initialize validator for Manim scripts
        try:
            self.validator = ManimScriptValidator()
            print("✅ Script validator loaded")
        except Exception as e:
            print(f"⚠️ Validator failed to load: {e}")
            self.validator = None
        
        # Initialize knowledge base
        self.knowledge_base = get_knowledge_base()
        
        # Evaluation metrics
        self.results = {
            'manim': defaultdict(list),
            'dsa': defaultdict(list), 
            'system_design': defaultdict(list),
            'overall': defaultdict(list)
        }
    
    def create_evaluation_dataset(self) -> Dict[str, List[Dict]]:
        """Create comprehensive evaluation dataset."""
        
        eval_data = {
            'manim': [
                {
                    'query': 'Create a blue circle that rotates 360 degrees',
                    'expected_elements': ['Circle', 'BLUE', 'Rotate', 'from manim import'],
                    'difficulty': 'basic'
                },
                {
                    'query': 'Transform a red square into a green circle with smooth animation',
                    'expected_elements': ['Square', 'Circle', 'Transform', 'RED', 'GREEN'],
                    'difficulty': 'intermediate'
                },
                {
                    'query': 'Create a sine wave that draws itself from left to right with mathematical labels',
                    'expected_elements': ['sin', 'axes', 'graph', 'MathTex', 'animate'],
                    'difficulty': 'advanced'
                },
                {
                    'query': 'Show the equation E=mc² with a dramatic fade in effect',
                    'expected_elements': ['MathTex', 'E = mc^2', 'FadeIn'],
                    'difficulty': 'basic'
                },
                {
                    'query': 'Create multiple shapes arranged in a circle that rotate together',
                    'expected_elements': ['Circle', 'arrange', 'Rotate', 'Group'],
                    'difficulty': 'advanced'
                }
            ],
            
            'dsa': [
                {
                    'query': 'Explain binary search algorithm with implementation',
                    'expected_elements': ['binary search', 'O(log n)', 'sorted array', 'def binary_search'],
                    'difficulty': 'basic'
                },
                {
                    'query': 'Implement merge sort with detailed complexity analysis',
                    'expected_elements': ['merge sort', 'O(n log n)', 'divide and conquer', 'def merge_sort'],
                    'difficulty': 'intermediate'
                },
                {
                    'query': 'Compare time complexities of different sorting algorithms',
                    'expected_elements': ['O(n²)', 'O(n log n)', 'comparison', 'bubble', 'quick', 'merge'],
                    'difficulty': 'intermediate'
                },
                {
                    'query': 'Explain binary tree traversal methods with examples',
                    'expected_elements': ['inorder', 'preorder', 'postorder', 'binary tree', 'traversal'],
                    'difficulty': 'basic'
                },
                {
                    'query': 'Implement dynamic programming solution for longest common subsequence',
                    'expected_elements': ['dynamic programming', 'LCS', 'dp table', 'O(mn)'],
                    'difficulty': 'advanced'
                }
            ],
            
            'system_design': [
                {
                    'query': 'Explain singleton design pattern with thread-safe implementation',
                    'expected_elements': ['singleton', 'thread-safe', 'instance', 'design pattern'],
                    'difficulty': 'basic'
                },
                {
                    'query': 'How does a load balancer work in distributed systems?',
                    'expected_elements': ['load balancer', 'distributed', 'requests', 'servers'],
                    'difficulty': 'intermediate'
                },
                {
                    'query': 'Design a scalable chat application architecture',
                    'expected_elements': ['scalable', 'architecture', 'microservices', 'database'],
                    'difficulty': 'advanced'
                },
                {
                    'query': 'Compare microservices vs monolith architecture',
                    'expected_elements': ['microservices', 'monolith', 'pros', 'cons', 'scalability'],
                    'difficulty': 'intermediate'
                },
                {
                    'query': 'Explain database sharding strategies and trade-offs',
                    'expected_elements': ['sharding', 'database', 'partition', 'trade-offs'],
                    'difficulty': 'advanced'
                }
            ]
        }
        
        return eval_data
    
    def evaluate_response_quality(self, query: str, response: str, expected_elements: List[str], 
                                 domain: str) -> Dict[str, float]:
        """Evaluate response quality across multiple dimensions."""
        
        metrics = {}
        response_lower = response.lower()
        
        # 1. Content Coverage (0-1)
        covered_elements = sum(1 for elem in expected_elements if elem.lower() in response_lower)
        metrics['content_coverage'] = covered_elements / len(expected_elements)
        
        # 2. Response Length Score (normalized)
        length_score = min(len(response) / 500, 1.0)  # Normalize to 500 chars
        metrics['length_score'] = length_score
        
        # 3. Domain-specific Quality
        if domain == 'manim':
            # Check for valid Manim script structure
            manim_score = self._evaluate_manim_script(response)
            metrics['domain_score'] = manim_score
        elif domain == 'dsa':
            # Check for algorithmic content
            dsa_score = self._evaluate_dsa_content(response)
            metrics['domain_score'] = dsa_score
        elif domain == 'system_design':
            # Check for system design concepts
            sysdes_score = self._evaluate_system_design_content(response)
            metrics['domain_score'] = sysdes_score
        else:
            metrics['domain_score'] = 0.5
        
        # 4. Code Quality (if applicable)
        code_score = self._evaluate_code_quality(response)
        metrics['code_quality'] = code_score
        
        # 5. Overall Quality (weighted average)
        metrics['overall_quality'] = (
            0.3 * metrics['content_coverage'] +
            0.2 * metrics['length_score'] +
            0.3 * metrics['domain_score'] +
            0.2 * metrics['code_quality']
        )
        
        return metrics
    
    def _evaluate_manim_script(self, response: str) -> float:
        """Evaluate Manim script quality."""
        score = 0.0
        
        # Check for essential Manim components
        manim_indicators = [
            'from manim import',
            'class',
            'Scene',
            'def construct',
            'self.play',
            'self.wait'
        ]
        
        for indicator in manim_indicators:
            if indicator in response:
                score += 1.0 / len(manim_indicators)
        
        # Use validator if available
        if self.validator and 'from manim import' in response:
            try:
                is_valid, _, _ = self.validator.validate_and_fix(response)
                if is_valid:
                    score += 0.3  # Bonus for valid script
            except:
                pass
        
        return min(score, 1.0)
    
    def _evaluate_dsa_content(self, response: str) -> float:
        """Evaluate DSA content quality."""
        score = 0.0
        response_lower = response.lower()
        
        # Check for algorithmic concepts
        dsa_indicators = [
            'time complexity',
            'space complexity',
            'algorithm',
            'data structure',
            'o(',  # Big O notation
            'def ',  # Function definition
            'implementation'
        ]
        
        for indicator in dsa_indicators:
            if indicator in response_lower:
                score += 1.0 / len(dsa_indicators)
        
        # Bonus for code blocks
        if '```python' in response or 'def ' in response:
            score += 0.2
        
        return min(score, 1.0)
    
    def _evaluate_system_design_content(self, response: str) -> float:
        """Evaluate system design content quality."""
        score = 0.0
        response_lower = response.lower()
        
        # Check for system design concepts
        sysdes_indicators = [
            'architecture',
            'scalability', 
            'design pattern',
            'distributed',
            'microservices',
            'database',
            'load balancer',
            'benefits',
            'trade-offs'
        ]
        
        for indicator in sysdes_indicators:
            if indicator in response_lower:
                score += 1.0 / len(sysdes_indicators)
        
        # Bonus for structured explanation
        if 'definition:' in response_lower or 'benefits:' in response_lower:
            score += 0.2
        
        return min(score, 1.0)
    
    def _evaluate_code_quality(self, response: str) -> float:
        """Evaluate code quality in response."""
        score = 0.0
        
        # Check for code presence
        if 'def ' in response or 'class ' in response:
            score += 0.3
        
        # Check for code blocks
        if '```python' in response or '```' in response:
            score += 0.3
        
        # Check for comments
        if '#' in response:
            score += 0.2
        
        # Check for proper indentation patterns
        lines = response.split('\n')
        indented_lines = sum(1 for line in lines if line.startswith('    ') or line.startswith('\t'))
        if indented_lines > 0:
            score += 0.2
        
        return min(score, 1.0)
    
    def run_comprehensive_evaluation(self) -> Dict[str, Dict]:
        """Run comprehensive evaluation across all domains."""
        
        print("🧪 Running Comprehensive Multi-Domain Evaluation")
        print("=" * 60)
        
        eval_data = self.create_evaluation_dataset()
        
        overall_results = {}
        
        for domain, queries in eval_data.items():
            print(f"\n🔍 Evaluating {domain.upper()} domain...")
            
            domain_results = []
            
            for i, test_case in enumerate(queries):
                query = test_case['query']
                expected_elements = test_case['expected_elements']
                difficulty = test_case['difficulty']
                
                print(f"  📝 Test {i+1}/{len(queries)}: {difficulty} - {query[:50]}...")
                
                # Generate response
                start_time = time.time()
                try:
                    result = self.inference_engine.generate_response(
                        query, 
                        max_length=512,
                        temperature=0.7
                    )
                    response = result['response']
                    generation_time = time.time() - start_time
                    
                    # Evaluate response
                    metrics = self.evaluate_response_quality(
                        query, response, expected_elements, domain
                    )
                    
                    # Add metadata
                    metrics.update({
                        'query': query,
                        'response_length': len(response),
                        'generation_time': generation_time,
                        'difficulty': difficulty,
                        'domain_detected': result['domain_info']['detected_domain'],
                        'domain_confidence': result['domain_info']['confidence']
                    })
                    
                    domain_results.append(metrics)
                    
                    print(f"    ✅ Quality: {metrics['overall_quality']:.3f} | "
                          f"Coverage: {metrics['content_coverage']:.3f} | "
                          f"Time: {generation_time:.2f}s")
                    
                except Exception as e:
                    print(f"    ❌ Error: {e}")
                    domain_results.append({
                        'query': query,
                        'error': str(e),
                        'overall_quality': 0.0
                    })
            
            overall_results[domain] = domain_results
        
        # Calculate summary statistics
        summary = self._calculate_summary_stats(overall_results)
        
        # Save results
        self._save_evaluation_results(overall_results, summary)
        
        # Print summary
        self._print_evaluation_summary(summary)
        
        return overall_results
    
    def _calculate_summary_stats(self, results: Dict[str, List[Dict]]) -> Dict:
        """Calculate summary statistics."""
        
        summary = {}
        
        for domain, domain_results in results.items():
            valid_results = [r for r in domain_results if 'overall_quality' in r and r['overall_quality'] > 0]
            
            if valid_results:
                qualities = [r['overall_quality'] for r in valid_results]
                coverages = [r.get('content_coverage', 0) for r in valid_results]
                times = [r.get('generation_time', 0) for r in valid_results]
                
                summary[domain] = {
                    'avg_quality': np.mean(qualities),
                    'std_quality': np.std(qualities),
                    'avg_coverage': np.mean(coverages),
                    'avg_time': np.mean(times),
                    'success_rate': len(valid_results) / len(domain_results),
                    'total_tests': len(domain_results)
                }
        
        # Overall stats
        all_qualities = []
        all_times = []
        for domain_stats in summary.values():
            all_qualities.append(domain_stats['avg_quality'])
            all_times.append(domain_stats['avg_time'])
        
        summary['overall'] = {
            'avg_quality': np.mean(all_qualities),
            'avg_time': np.mean(all_times),
            'total_domains': len(summary)
        }
        
        return summary
    
    def _save_evaluation_results(self, results: Dict, summary: Dict) -> None:
        """Save evaluation results to files."""
        
        # Save detailed results
        with open('evaluation_results_detailed.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save summary
        with open('evaluation_results_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        print("\n💾 Results saved:")
        print("   📄 evaluation_results_detailed.json")
        print("   📊 evaluation_results_summary.json")
    
    def _print_evaluation_summary(self, summary: Dict) -> None:
        """Print evaluation summary."""
        
        print("\n📊 EVALUATION SUMMARY")
        print("=" * 60)
        
        for domain, stats in summary.items():
            if domain == 'overall':
                continue
                
            print(f"\n🔹 {domain.upper()}:")
            print(f"   Quality Score: {stats['avg_quality']:.3f} (±{stats['std_quality']:.3f})")
            print(f"   Content Coverage: {stats['avg_coverage']:.3f}")
            print(f"   Success Rate: {stats['success_rate']:.1%}")
            print(f"   Avg Generation Time: {stats['avg_time']:.2f}s")
            print(f"   Tests Run: {stats['total_tests']}")
        
        print(f"\n🏆 OVERALL PERFORMANCE:")
        print(f"   Average Quality: {summary['overall']['avg_quality']:.3f}")
        print(f"   Average Speed: {summary['overall']['avg_time']:.2f}s")
        print(f"   Domains Evaluated: {summary['overall']['total_domains']}")
        
        # Performance rating
        overall_quality = summary['overall']['avg_quality']
        if overall_quality >= 0.8:
            rating = "🌟 EXCELLENT"
        elif overall_quality >= 0.7:
            rating = "✅ GOOD"
        elif overall_quality >= 0.6:
            rating = "⚠️ FAIR"
        else:
            rating = "❌ NEEDS IMPROVEMENT"
        
        print(f"   Performance Rating: {rating}")

def main():
    """Main evaluation function."""
    
    # Check if model exists
    model_path = "enhanced_model_checkpoints/enhanced_best_model.pth"
    tokenizer_path = "enhanced_tokenizer.pkl"
    config_path = "enhanced_model_checkpoints/enhanced_model_config.json"
    
    if not os.path.exists(model_path):
        print("❌ Trained model not found. Please train the model first:")
        print("   python3 enhanced_trainer.py")
        return
    
    if not os.path.exists(tokenizer_path):
        print("❌ Enhanced tokenizer not found. Please run:")
        print("   python3 enhanced_tokenizer.py")
        return
    
    print("🔍 Starting Enhanced Model Evaluation")
    
    # Create evaluator
    evaluator = EnhancedModelEvaluator(model_path, tokenizer_path, config_path)
    
    # Run evaluation
    results = evaluator.run_comprehensive_evaluation()
    
    print("\n🎉 Evaluation completed successfully!")
    print("Check the generated JSON files for detailed results.")

if __name__ == "__main__":
    import os
    main()