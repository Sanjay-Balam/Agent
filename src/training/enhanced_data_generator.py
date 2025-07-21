"""
Enhanced Multi-Domain Data Generator
Extends the original ManimDataGenerator to support DSA and System Design domains
while maintaining all existing Manim functionality.
"""

import random
import json
from typing import List, Dict, Tuple, Optional
from .data_generator import ManimDataGenerator
from ..knowledge.multi_domain_knowledge_base import MultiDomainKnowledgeBase, Domain
from ..knowledge.dsa_knowledge_base import DATA_STRUCTURES, ALGORITHMS, DSA_PATTERNS, COMMON_QUESTIONS
from ..knowledge.system_design_knowledge_base import LLD_PATTERNS, HLD_COMPONENTS, SYSTEM_DESIGN_PATTERNS

class DSADataGenerator:
    """Generates training data for DSA topics."""
    
    def __init__(self):
        self.data_structures = list(DATA_STRUCTURES.keys())
        self.algorithms = list(ALGORITHMS.keys())
        self.patterns = list(DSA_PATTERNS.keys())
        self.questions = list(COMMON_QUESTIONS.keys())
        
        # Request templates for DSA
        self.request_templates = [
            "Explain {data_structure} data structure",
            "Implement {algorithm} algorithm",
            "What is the time complexity of {algorithm}?",
            "Show me {data_structure} implementation in Python",
            "How does {algorithm} work?",
            "Explain {pattern} pattern with example",
            "Solve {question} problem",
            "What are the use cases for {data_structure}?",
            "Compare {data_structure} and {data_structure2}",
            "Optimize this {algorithm} solution",
            "When should I use {data_structure}?",
            "Implement {data_structure} with all operations",
            "Show step-by-step execution of {algorithm}",
            "What is the space complexity of {data_structure}?",
            "Explain the difference between {algorithm} and {algorithm2}"
        ]
    
    def generate_single_request_response(self) -> Tuple[str, str]:
        """Generate a single DSA request-response pair."""
        template = random.choice(self.request_templates)
        
        # Fill template with random values
        request = template.format(
            data_structure=random.choice(self.data_structures),
            data_structure2=random.choice(self.data_structures),
            algorithm=random.choice(self.algorithms),
            algorithm2=random.choice(self.algorithms),
            pattern=random.choice(self.patterns),
            question=random.choice(self.questions)
        )
        
        # Generate corresponding response
        response = self._generate_response_for_request(request)
        
        return request, response
    
    def _generate_response_for_request(self, request: str) -> str:
        """Generate DSA response based on the request."""
        request_lower = request.lower()
        
        # Determine what the request is about
        if "implement" in request_lower or "implementation" in request_lower:
            return self._generate_implementation_response(request_lower)
        elif "time complexity" in request_lower:
            return self._generate_complexity_response(request_lower, "time")
        elif "space complexity" in request_lower:
            return self._generate_complexity_response(request_lower, "space")
        elif "explain" in request_lower:
            return self._generate_explanation_response(request_lower)
        elif "solve" in request_lower:
            return self._generate_solution_response(request_lower)
        else:
            return self._generate_general_response(request_lower)
    
    def _generate_implementation_response(self, request: str) -> str:
        """Generate implementation response."""
        # Find which data structure or algorithm is requested
        for ds in self.data_structures:
            if ds in request:
                ds_info = DATA_STRUCTURES[ds]
                return f"""Here's a Python implementation of {ds.title()}:

{ds_info['code_template']}

Time Complexity:
{self._format_complexity(ds_info.get('time_complexity', {}))}

Space Complexity: {ds_info.get('space_complexity', 'O(n)')}

Use Cases:
{self._format_use_cases(ds_info.get('use_cases', []))}
"""
        
        for algo in self.algorithms:
            if algo in request:
                algo_info = ALGORITHMS[algo]
                return f"""Here's a Python implementation of {algo.title().replace('_', ' ')}:

{algo_info['code_template']}

Time Complexity:
{self._format_complexity(algo_info.get('time_complexity', {}))}

Space Complexity: {algo_info.get('space_complexity', 'O(1)')}

Use Cases:
{self._format_use_cases(algo_info.get('use_cases', []))}
"""
        
        return "I'll provide a Python implementation with time/space complexity analysis and use cases."
    
    def _generate_complexity_response(self, request: str, complexity_type: str) -> str:
        """Generate complexity analysis response."""
        for item in self.data_structures + self.algorithms:
            if item in request:
                if item in DATA_STRUCTURES:
                    info = DATA_STRUCTURES[item]
                else:
                    info = ALGORITHMS[item]
                
                complexity = info.get(f'{complexity_type}_complexity', {})
                if isinstance(complexity, dict):
                    return f"The {complexity_type} complexity of {item.title().replace('_', ' ')} is:\n{self._format_complexity(complexity)}"
                else:
                    return f"The {complexity_type} complexity of {item.title().replace('_', ' ')} is: {complexity}"
        
        return f"I'll analyze the {complexity_type} complexity with detailed explanation."
    
    def _generate_explanation_response(self, request: str) -> str:
        """Generate explanation response."""
        for ds in self.data_structures:
            if ds in request:
                ds_info = DATA_STRUCTURES[ds]
                return f"""{ds.title().replace('_', ' ')}:

Definition: {ds_info.get('definition', 'A fundamental data structure')}

Key Characteristics:
- Time Complexity: {self._format_complexity(ds_info.get('time_complexity', {}))}
- Space Complexity: {ds_info.get('space_complexity', 'O(n)')}

Common Use Cases:
{self._format_use_cases(ds_info.get('use_cases', []))}

Implementation details and examples would be provided with code snippets.
"""
        
        for algo in self.algorithms:
            if algo in request:
                algo_info = ALGORITHMS[algo]
                return f"""{algo.title().replace('_', ' ')}:

Definition: {algo_info.get('definition', 'An important algorithm')}

Algorithm Characteristics:
- Time Complexity: {self._format_complexity(algo_info.get('time_complexity', {}))}
- Space Complexity: {algo_info.get('space_complexity', 'O(1)')}
- Stable: {algo_info.get('stable', 'N/A')}

Use Cases:
{self._format_use_cases(algo_info.get('use_cases', []))}

Step-by-step explanation and implementation would follow.
"""
        
        return "I'll provide a detailed explanation with examples and complexity analysis."
    
    def _generate_solution_response(self, request: str) -> str:
        """Generate solution response for coding problems."""
        for question in self.questions:
            if question in request:
                q_info = COMMON_QUESTIONS[question]
                return f"""Problem: {q_info['question']}

Solution:
{q_info['solution']}

Difficulty: {q_info['difficulty']}
Topics: {', '.join(q_info['topics'])}

Explanation:
This solution uses {', '.join(q_info['topics']).lower()} concepts to solve the problem efficiently.
"""
        
        return "I'll provide a complete solution with explanation, time/space complexity analysis, and test cases."
    
    def _generate_general_response(self, request: str) -> str:
        """Generate general response."""
        return "I'll provide a comprehensive explanation with code examples, complexity analysis, and practical applications."
    
    def _format_complexity(self, complexity: Dict) -> str:
        """Format complexity dictionary into readable string."""
        if not complexity:
            return "O(n)"
        
        if isinstance(complexity, str):
            return complexity
        
        formatted = []
        for case, value in complexity.items():
            formatted.append(f"- {case.title()}: {value}")
        return '\n'.join(formatted)
    
    def _format_use_cases(self, use_cases: List) -> str:
        """Format use cases list into readable string."""
        if not use_cases:
            return "- General purpose applications"
        
        return '\n'.join(f"- {case}" for case in use_cases)

class SystemDesignDataGenerator:
    """Generates training data for System Design topics."""
    
    def __init__(self):
        self.lld_patterns = list(LLD_PATTERNS.keys())
        self.hld_components = list(HLD_COMPONENTS.keys())
        self.system_patterns = list(SYSTEM_DESIGN_PATTERNS.keys())
        
        # Request templates for System Design
        self.request_templates = [
            "Explain {lld_pattern} design pattern",
            "How does {hld_component} work in distributed systems?",
            "Design a {system} using {system_pattern}",
            "When should I use {lld_pattern} pattern?",
            "Compare {lld_pattern} and {lld_pattern2} patterns",
            "Implement {lld_pattern} pattern in Python",
            "What are the benefits of {hld_component}?",
            "Design a scalable {system} architecture",
            "Explain {system_pattern} with real-world example",
            "How to implement {hld_component} in microservices?",
            "What are the trade-offs of {system_pattern}?",
            "Design {system} to handle {scale} users",
            "How does {hld_component} improve system performance?",
            "Implement {system_pattern} for {use_case}",
            "What are the challenges with {system_pattern}?"
        ]
        
        # Sample systems and use cases
        self.systems = [
            "chat application", "e-commerce platform", "social media", "video streaming",
            "online banking", "ride sharing", "food delivery", "search engine",
            "content management", "gaming platform"
        ]
        
        self.scales = ["1 million", "10 million", "100 million", "1 billion"]
        self.use_cases = [
            "high availability", "real-time processing", "data consistency",
            "fault tolerance", "scalability", "performance optimization"
        ]
    
    def generate_single_request_response(self) -> Tuple[str, str]:
        """Generate a single System Design request-response pair."""
        template = random.choice(self.request_templates)
        
        # Fill template with random values
        request = template.format(
            lld_pattern=random.choice(self.lld_patterns),
            lld_pattern2=random.choice(self.lld_patterns),
            hld_component=random.choice(self.hld_components),
            system_pattern=random.choice(self.system_patterns),
            system=random.choice(self.systems),
            scale=random.choice(self.scales),
            use_case=random.choice(self.use_cases)
        )
        
        # Generate corresponding response
        response = self._generate_response_for_request(request)
        
        return request, response
    
    def _generate_response_for_request(self, request: str) -> str:
        """Generate System Design response based on the request."""
        request_lower = request.lower()
        
        if "implement" in request_lower or "implementation" in request_lower:
            return self._generate_implementation_response(request_lower)
        elif "explain" in request_lower:
            return self._generate_explanation_response(request_lower)
        elif "design" in request_lower:
            return self._generate_design_response(request_lower)
        elif "compare" in request_lower:
            return self._generate_comparison_response(request_lower)
        elif "benefits" in request_lower or "advantages" in request_lower:
            return self._generate_benefits_response(request_lower)
        elif "challenges" in request_lower or "trade-offs" in request_lower:
            return self._generate_challenges_response(request_lower)
        else:
            return self._generate_general_response(request_lower)
    
    def _generate_implementation_response(self, request: str) -> str:
        """Generate implementation response."""
        for pattern in self.lld_patterns:
            if pattern in request:
                pattern_info = LLD_PATTERNS[pattern]
                return f"""Implementation of {pattern.title()} Pattern:

{pattern_info['code_template']}

Benefits:
{self._format_list(pattern_info.get('pros', []))}

Drawbacks:
{self._format_list(pattern_info.get('cons', []))}

Use Cases:
{self._format_list(pattern_info.get('use_cases', []))}
"""
        
        return "I'll provide a complete implementation with code examples, benefits, and use cases."
    
    def _generate_explanation_response(self, request: str) -> str:
        """Generate explanation response."""
        # Check LLD patterns
        for pattern in self.lld_patterns:
            if pattern in request:
                pattern_info = LLD_PATTERNS[pattern]
                return f"""{pattern.title()} Design Pattern:

Definition: {pattern_info.get('definition', 'A design pattern for software architecture')}

Key Benefits:
{self._format_list(pattern_info.get('pros', []))}

Potential Drawbacks:
{self._format_list(pattern_info.get('cons', []))}

Common Use Cases:
{self._format_list(pattern_info.get('use_cases', []))}

Implementation example and detailed explanation would follow.
"""
        
        # Check HLD components
        for component in self.hld_components:
            if component in request:
                component_info = HLD_COMPONENTS[component]
                return f"""{component.title().replace('_', ' ')}:

Definition: {component_info.get('definition', 'A system component for distributed architecture')}

Key Features and Benefits:
{self._format_benefits(component_info)}

Use Cases:
{self._format_use_cases(component_info)}

Detailed architecture diagrams and implementation details would be provided.
"""
        
        return "I'll provide a comprehensive explanation with architecture diagrams and examples."
    
    def _generate_design_response(self, request: str) -> str:
        """Generate system design response."""
        return """System Design Approach:

1. Requirements Gathering
   - Functional requirements
   - Non-functional requirements (scale, performance, availability)

2. High-Level Architecture
   - Component identification
   - Data flow design
   - Technology stack selection

3. Detailed Design
   - Database schema
   - API design
   - Service interactions

4. Scalability Considerations
   - Load balancing
   - Caching strategies
   - Database scaling

5. Monitoring and Maintenance
   - Logging and metrics
   - Error handling
   - Deployment strategy

Detailed diagrams and implementation details would follow.
"""
    
    def _generate_comparison_response(self, request: str) -> str:
        """Generate comparison response."""
        return "I'll provide a detailed comparison covering definitions, use cases, pros/cons, and implementation examples for each approach."
    
    def _generate_benefits_response(self, request: str) -> str:
        """Generate benefits response."""
        return "I'll explain the key benefits including scalability improvements, performance gains, reliability enhancements, and maintenance advantages."
    
    def _generate_challenges_response(self, request: str) -> str:
        """Generate challenges response."""
        return "I'll discuss the main challenges including complexity increases, potential performance trade-offs, implementation difficulties, and maintenance considerations."
    
    def _generate_general_response(self, request: str) -> str:
        """Generate general response."""
        return "I'll provide a comprehensive system design explanation with architecture diagrams, trade-off analysis, and implementation guidelines."
    
    def _format_list(self, items: List) -> str:
        """Format list into readable string."""
        if not items:
            return "- Not specified"
        return '\n'.join(f"- {item}" for item in items)
    
    def _format_benefits(self, component_info: Dict) -> str:
        """Format component benefits."""
        benefits = component_info.get('benefits', [])
        if benefits:
            return self._format_list(benefits)
        return "- Improved system architecture"
    
    def _format_use_cases(self, component_info: Dict) -> str:
        """Format component use cases."""
        use_cases = component_info.get('use_cases', [])
        if use_cases:
            return self._format_list(use_cases)
        return "- General distributed systems"

class EnhancedDataGenerator:
    """Enhanced data generator that supports all domains: Manim, DSA, and System Design."""
    
    def __init__(self):
        self.manim_generator = ManimDataGenerator()
        self.dsa_generator = DSADataGenerator()
        self.system_design_generator = SystemDesignDataGenerator()
        self.knowledge_base = MultiDomainKnowledgeBase()
    
    def generate_single_request_response(self, domain: Optional[Domain] = None) -> Tuple[str, str, str]:
        """Generate a single request-response pair, optionally for a specific domain."""
        if domain is None:
            # Randomly choose domain
            domain = random.choice([Domain.MANIM, Domain.DSA, Domain.SYSTEM_DESIGN])
        
        if domain == Domain.MANIM:
            request, response = self.manim_generator.generate_single_request_response()
        elif domain == Domain.DSA:
            request, response = self.dsa_generator.generate_single_request_response()
        elif domain == Domain.SYSTEM_DESIGN:
            request, response = self.system_design_generator.generate_single_request_response()
        else:
            raise ValueError(f"Unsupported domain: {domain}")
        
        return request, response, domain.value
    
    def generate_training_data(self, num_samples: int = 1000, domain_distribution: Optional[Dict[Domain, float]] = None) -> List[Dict]:
        """
        Generate training dataset with specified distribution across domains.
        
        Args:
            num_samples: Total number of samples to generate
            domain_distribution: Dict mapping Domain to proportion (should sum to 1.0)
                                Default: equal distribution across all domains
        """
        if domain_distribution is None:
            # Equal distribution
            domain_distribution = {
                Domain.MANIM: 0.33,
                Domain.DSA: 0.34,
                Domain.SYSTEM_DESIGN: 0.33
            }
        
        # Validate distribution
        if abs(sum(domain_distribution.values()) - 1.0) > 0.01:
            raise ValueError("Domain distribution must sum to 1.0")
        
        training_data = []
        
        print(f"Generating {num_samples} training samples across multiple domains...")
        
        # Calculate samples per domain
        samples_per_domain = {
            domain: int(num_samples * proportion) 
            for domain, proportion in domain_distribution.items()
        }
        
        # Adjust for rounding errors
        total_assigned = sum(samples_per_domain.values())
        if total_assigned < num_samples:
            # Add remaining to largest domain
            largest_domain = max(samples_per_domain, key=samples_per_domain.get)
            samples_per_domain[largest_domain] += num_samples - total_assigned
        
        sample_id = 0
        for domain, num_domain_samples in samples_per_domain.items():
            print(f"Generating {num_domain_samples} samples for {domain.value}...")
            
            for i in range(num_domain_samples):
                if sample_id % 100 == 0:
                    print(f"Generated {sample_id}/{num_samples} total samples")
                
                request, response, domain_name = self.generate_single_request_response(domain)
                
                training_data.append({
                    "id": sample_id,
                    "domain": domain_name,
                    "request": request,
                    "response": response,
                    "length": len(response)
                })
                
                sample_id += 1
        
        print(f"Generated {len(training_data)} training samples across all domains")
        return training_data
    
    def save_training_data(self, data: List[Dict], filename: str = "enhanced_training_data.json"):
        """Save training data to file."""
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Enhanced training data saved to {filename}")
        
        # Print statistics
        domain_counts = {}
        for item in data:
            domain = item['domain']
            domain_counts[domain] = domain_counts.get(domain, 0) + 1
        
        print("Domain distribution:")
        for domain, count in domain_counts.items():
            percentage = (count / len(data)) * 100
            print(f"  {domain}: {count} samples ({percentage:.1f}%)")
    
    def load_training_data(self, filename: str = "enhanced_training_data.json") -> List[Dict]:
        """Load training data from file."""
        with open(filename, 'r') as f:
            data = json.load(f)
        print(f"Loaded {len(data)} training samples from {filename}")
        return data
    
    def generate_and_save_dataset(self, num_samples: int = 1000, filename: str = "enhanced_training_data.json",
                                  domain_distribution: Optional[Dict[Domain, float]] = None):
        """Generate and save enhanced training dataset."""
        # Generate synthetic data
        training_data = self.generate_training_data(num_samples, domain_distribution)
        
        # Save to file
        self.save_training_data(training_data, filename)
        
        return training_data
    
    def create_validation_split(self, data: List[Dict], split_ratio: float = 0.8) -> Tuple[List[Dict], List[Dict]]:
        """Split data into training and validation sets while maintaining domain distribution."""
        # Separate by domain
        domain_data = {}
        for item in data:
            domain = item['domain']
            if domain not in domain_data:
                domain_data[domain] = []
            domain_data[domain].append(item)
        
        train_data = []
        val_data = []
        
        # Split each domain separately to maintain distribution
        for domain, items in domain_data.items():
            random.shuffle(items)
            split_idx = int(len(items) * split_ratio)
            
            train_data.extend(items[:split_idx])
            val_data.extend(items[split_idx:])
        
        # Shuffle the final datasets
        random.shuffle(train_data)
        random.shuffle(val_data)
        
        print(f"Training samples: {len(train_data)}")
        print(f"Validation samples: {len(val_data)}")
        
        # Print domain distribution for both sets
        def print_distribution(dataset, name):
            domain_counts = {}
            for item in dataset:
                domain = item['domain']
                domain_counts[domain] = domain_counts.get(domain, 0) + 1
            
            print(f"{name} domain distribution:")
            for domain, count in domain_counts.items():
                percentage = (count / len(dataset)) * 100
                print(f"  {domain}: {count} samples ({percentage:.1f}%)")
        
        print_distribution(train_data, "Training")
        print_distribution(val_data, "Validation")
        
        return train_data, val_data

if __name__ == "__main__":
    # Test the enhanced data generator
    generator = EnhancedDataGenerator()
    
    # Test single sample generation for each domain
    print("=== Testing Single Sample Generation ===")
    for domain in [Domain.MANIM, Domain.DSA, Domain.SYSTEM_DESIGN]:
        request, response, domain_name = generator.generate_single_request_response(domain)
        print(f"\nDomain: {domain_name}")
        print(f"Request: {request}")
        print(f"Response length: {len(response)} characters")
        print(f"Response preview: {response[:100]}...")
    
    # Generate a small dataset
    print("\n=== Generating Enhanced Dataset ===")
    data = generator.generate_and_save_dataset(
        num_samples=150,  # Small sample for testing
        filename="test_enhanced_training_data.json",
        domain_distribution={
            Domain.MANIM: 0.4,
            Domain.DSA: 0.4,
            Domain.SYSTEM_DESIGN: 0.2
        }
    )
    
    # Create train/validation split
    train_data, val_data = generator.create_validation_split(data)
    
    # Save splits
    generator.save_training_data(train_data, "test_train_data.json")
    generator.save_training_data(val_data, "test_val_data.json")
    
    print("\nEnhanced dataset generation complete!")
    print(f"Total samples: {len(data)}")
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")