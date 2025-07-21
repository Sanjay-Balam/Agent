"""
Enhanced ManimAgent - AI Agent that supports multiple CS domains while maintaining backward compatibility.
This agent can use both the old Manim-only inference and the new multi-domain inference.
"""

import os
import json
import re
from typing import Dict, List, Optional, Tuple, Union

# Optional dotenv import
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Import original components for backward compatibility
from ..knowledge.knowledge_base import (
    MANIM_OBJECTS, MANIM_ANIMATIONS, MANIM_COLORS, MANIM_POSITIONS,
    COMMON_PATTERNS, EXAMPLE_SCRIPTS, get_object_code, get_animation_code,
    get_pattern_code, get_full_script
)

# Import enhanced components
from ..knowledge.multi_domain_knowledge_base import MultiDomainKnowledgeBase, Domain, get_knowledge_base

# Environment variables already loaded above if available

# Try to import inference engines
ORIGINAL_LLM_AVAILABLE = False
ENHANCED_LLM_AVAILABLE = False

try:
    from .inference import ManimInferenceEngine
    ORIGINAL_LLM_AVAILABLE = True
except ImportError:
    print("Original inference engine not available.")

try:
    from .enhanced_inference import EnhancedInferenceEngine
    ENHANCED_LLM_AVAILABLE = True
except ImportError:
    print("Enhanced inference engine not available.")

# Import external APIs (optional)
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

class EnhancedManimAgent:
    """Enhanced AI Agent supporting multiple CS domains while maintaining Manim compatibility."""
    
    def __init__(self, llm_provider="enhanced", model_path=None, tokenizer_path=None, 
                 model_name=None, api_key=None, use_enhanced_inference=True):
        """
        Initialize the Enhanced Manim Agent.
        
        Args:
            llm_provider: "enhanced", "custom", "openai", or "anthropic"
            model_path: Path to trained model
            tokenizer_path: Path to tokenizer
            model_name: Specific model to use (for external APIs)
            api_key: API key for external LLM providers
            use_enhanced_inference: Whether to use enhanced multi-domain inference
        """
        self.llm_provider = llm_provider.lower()
        self.use_enhanced_inference = use_enhanced_inference
        
        # Initialize knowledge base
        self.knowledge_base = get_knowledge_base()
        
        # Initialize based on provider
        if self.llm_provider in ["enhanced", "custom"]:
            self._initialize_local_llm(model_path, tokenizer_path)
        else:
            self.model_name = model_name or self._get_default_model()
            self.api_key = api_key or self._get_api_key()
            self.client = self._initialize_llm_client()
            self.system_prompt = self._create_system_prompt()
    
    def _initialize_local_llm(self, model_path: str = None, tokenizer_path: str = None):
        """Initialize the local LLM (enhanced or original)."""
        # Default paths
        if model_path is None:
            model_path = "best_model_epoch_10.pth"
        if tokenizer_path is None:
            tokenizer_path = "tokenizer.pkl"
        
        # Check if files exist
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
        
        # Try to initialize enhanced inference first
        if self.use_enhanced_inference and ENHANCED_LLM_AVAILABLE:
            try:
                self.inference_engine = EnhancedInferenceEngine(model_path, tokenizer_path)
                self.inference_type = "enhanced"
                print(f"Enhanced multi-domain LLM initialized with model: {model_path}")
                return
            except Exception as e:
                print(f"Failed to initialize enhanced inference: {e}")
                print("Falling back to original inference...")
        
        # Fall back to original inference
        if ORIGINAL_LLM_AVAILABLE:
            try:
                from inference import ManimInferenceEngine
                self.inference_engine = ManimInferenceEngine(model_path, tokenizer_path)
                self.inference_type = "original"
                print(f"Original Manim LLM initialized with model: {model_path}")
                return
            except Exception as e:
                print(f"Failed to initialize original inference: {e}")
        
        # If both fail, create a mock interface
        self.inference_engine = None
        self.inference_type = "mock"
        print("Warning: No inference engine available. Using template-based generation.")
    
    def _get_default_model(self) -> str:
        """Get default model for the provider."""
        if self.llm_provider == "openai":
            return "gpt-4"
        elif self.llm_provider == "anthropic":
            return "claude-3-sonnet-20240229"
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
    
    def _get_api_key(self) -> str:
        """Get API key from environment variables."""
        if self.llm_provider == "openai":
            return os.getenv("OPENAI_API_KEY")
        elif self.llm_provider == "anthropic":
            return os.getenv("ANTHROPIC_API_KEY")
        else:
            raise ValueError(f"Unsupported LLM provider: {self.llm_provider}")
    
    def _initialize_llm_client(self):
        """Initialize the external LLM client."""
        try:
            if self.llm_provider == "openai":
                if not OPENAI_AVAILABLE:
                    raise ImportError("OpenAI package not installed")
                import openai
                return openai.OpenAI(api_key=self.api_key)
            elif self.llm_provider == "anthropic":
                if not ANTHROPIC_AVAILABLE:
                    raise ImportError("Anthropic package not installed")
                import anthropic
                return anthropic.Anthropic(api_key=self.api_key)
        except ImportError as e:
            raise ImportError(f"Required package not installed: {e}")
        except Exception as e:
            raise Exception(f"Failed to initialize LLM client: {e}")
    
    def _create_system_prompt(self) -> str:
        """Create enhanced system prompt supporting multiple domains."""
        return f"""You are a specialized AI agent for generating content across multiple computer science domains:

1. **MANIM ANIMATIONS**: Generate Python Manim scripts for mathematical visualizations
2. **DATA STRUCTURES & ALGORITHMS**: Explain concepts, provide implementations, analyze complexity  
3. **SYSTEM DESIGN**: Explain architectural patterns, design principles, scalability concepts

For Manim scripts, follow these guidelines:
- ALWAYS include proper imports: `from manim import *` and `import numpy as np`
- Create a Scene class with descriptive name
- Use the construct() method for animation logic
- Available objects: {list(MANIM_OBJECTS.keys())}
- Available animations: {list(MANIM_ANIMATIONS.keys())}
- Available colors: {MANIM_COLORS}

For DSA topics:
- Provide clear explanations with code examples
- Include time/space complexity analysis
- Show practical use cases and variations

For System Design:
- Explain concepts with real-world examples
- Discuss trade-offs and best practices
- Include implementation guidance

Always generate complete, working examples. Detect the domain from the user's request and respond accordingly."""
    
    def generate_script(self, user_request: str) -> str:
        """
        Generate response based on user request with domain detection.
        
        Args:
            user_request: User's description of desired content
            
        Returns:
            Generated content (script, explanation, or code)
        """
        try:
            # Detect domain if using enhanced inference
            if self.inference_type == "enhanced":
                return self._generate_with_enhanced_inference(user_request)
            elif self.inference_type == "original":
                return self._generate_with_original_inference(user_request)
            elif self.inference_type == "mock":
                return self._generate_with_templates(user_request)
            else:
                return self._generate_with_external_llm(user_request)
            
        except Exception as e:
            return f"Error generating content: {str(e)}"
    
    def _generate_with_enhanced_inference(self, user_request: str) -> str:
        """Generate using enhanced multi-domain inference."""
        try:
            result = self.inference_engine.generate_response(user_request)
            return result['response']
        except Exception as e:
            print(f"Enhanced inference failed: {e}")
            # Fall back to template-based generation
            return self._generate_with_templates(user_request)
    
    def _generate_with_original_inference(self, user_request: str) -> str:
        """Generate using original Manim-only inference."""
        try:
            result = self.inference_engine.generate_and_validate(user_request)
            return result.get('best_script', result.get('script', 'Generation failed'))
        except Exception as e:
            print(f"Original inference failed: {e}")
            return self._generate_with_templates(user_request)
    
    def _generate_with_templates(self, user_request: str) -> str:
        """Generate using template-based approach as fallback."""
        # Detect domain using knowledge base
        domain, confidence = self.knowledge_base.detect_domain(user_request)
        
        if domain == Domain.MANIM:
            return self._generate_manim_template(user_request)
        elif domain == Domain.DSA:
            return self._generate_dsa_template(user_request)
        elif domain == Domain.SYSTEM_DESIGN:
            return self._generate_system_design_template(user_request)
        else:
            return self._generate_general_template(user_request)
    
    def _generate_manim_template(self, user_request: str) -> str:
        """Generate Manim script using templates."""
        request_lower = user_request.lower()
        
        # Determine object type
        obj_type = "circle"
        for obj in MANIM_OBJECTS:
            if obj in request_lower:
                obj_type = obj
                break
        
        # Determine color
        color = "BLUE"
        for c in MANIM_COLORS:
            if c.lower() in request_lower:
                color = c
                break
        
        # Generate based on request type
        if "rotate" in request_lower:
            return f"""from manim import *
import numpy as np

class RotatingScene(Scene):
    def construct(self):
        # Create {obj_type}
        obj = {MANIM_OBJECTS[obj_type]['code'].format(**MANIM_OBJECTS[obj_type]['default_params'])}
        obj.set_color({color})
        
        # Show and rotate
        self.play(Create(obj))
        self.play(Rotate(obj, angle=2*PI))
        self.wait(1)"""
        
        elif "transform" in request_lower:
            return f"""from manim import *
import numpy as np

class TransformScene(Scene):
    def construct(self):
        # Create first shape
        shape1 = Circle(radius=1)
        shape1.set_color({color})
        
        # Create second shape  
        shape2 = Square(side_length=2)
        shape2.set_color(RED)
        
        # Show transformation
        self.play(Create(shape1))
        self.wait(1)
        self.play(Transform(shape1, shape2))
        self.wait(1)"""
        
        else:
            # Basic animation
            return f"""from manim import *
import numpy as np

class BasicScene(Scene):
    def construct(self):
        # Create {obj_type}
        obj = {MANIM_OBJECTS[obj_type]['code'].format(**MANIM_OBJECTS[obj_type]['default_params'])}
        obj.set_color({color})
        
        # Animate
        self.play(Create(obj))
        self.wait(1)"""
    
    def _generate_dsa_template(self, user_request: str) -> str:
        """Generate DSA explanation using templates."""
        request_lower = user_request.lower()
        
        if "binary search" in request_lower:
            return """Binary Search Algorithm:

```python
def binary_search(arr, target):
    left, right = 0, len(arr) - 1
    
    while left <= right:
        mid = (left + right) // 2
        
        if arr[mid] == target:
            return mid
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1  # Target not found

# Example usage
arr = [1, 3, 5, 7, 9, 11, 13, 15]
target = 7
result = binary_search(arr, target)
print(f"Element found at index: {result}")
```

Time Complexity: O(log n)
Space Complexity: O(1)

Binary search works by repeatedly dividing the search interval in half. It requires the array to be sorted."""
        
        elif "merge sort" in request_lower:
            return """Merge Sort Algorithm:

```python
def merge_sort(arr):
    if len(arr) <= 1:
        return arr
    
    # Divide
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    
    # Conquer
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    
    result.extend(left[i:])
    result.extend(right[j:])
    return result

# Example usage
arr = [38, 27, 43, 3, 9, 82, 10]
sorted_arr = merge_sort(arr)
print(f"Sorted array: {sorted_arr}")
```

Time Complexity: O(n log n)
Space Complexity: O(n)

Merge sort is a stable, divide-and-conquer algorithm that consistently performs well."""
        
        else:
            return f"DSA topic requested: {user_request}\n\nI can help explain data structures and algorithms. Try asking about specific topics like 'binary search', 'merge sort', 'binary trees', etc."
    
    def _generate_system_design_template(self, user_request: str) -> str:
        """Generate system design explanation using templates."""
        request_lower = user_request.lower()
        
        if "singleton" in request_lower:
            return """Singleton Design Pattern:

**Definition**: Ensures a class has only one instance and provides global access to it.

```python
class Singleton:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(Singleton, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._initialized = True
            self.data = "Singleton Instance"

# Usage
s1 = Singleton()
s2 = Singleton()
print(s1 is s2)  # True - same instance
```

**Benefits**:
- Controlled access to sole instance
- Reduced memory usage
- Global access point

**Use Cases**:
- Database connections
- Logging systems
- Configuration settings"""
        
        elif "load balancer" in request_lower:
            return """Load Balancer in System Design:

**Definition**: Distributes incoming requests across multiple servers to ensure no single server is overwhelmed.

**Types**:
1. **Layer 4 (Transport)**: Routes based on IP and port
2. **Layer 7 (Application)**: Routes based on HTTP headers, URLs

**Algorithms**:
- **Round Robin**: Equal distribution in rotation
- **Least Connections**: Route to server with fewest active connections
- **Weighted**: Assign weights based on server capacity

**Benefits**:
- High availability
- Improved performance
- Fault tolerance
- Scalability

**Popular Solutions**:
- nginx
- HAProxy
- AWS Application Load Balancer
- Google Cloud Load Balancing

Load balancers are essential for distributed systems handling high traffic volumes."""
        
        else:
            return f"System Design topic requested: {user_request}\n\nI can help explain system design concepts. Try asking about 'load balancer', 'singleton pattern', 'microservices', 'database sharding', etc."
    
    def _generate_general_template(self, user_request: str) -> str:
        """Generate general CS response."""
        return f"Computer Science topic requested: {user_request}\n\nI can help with:\n- Manim animations (e.g., 'create a rotating circle')\n- Data Structures & Algorithms (e.g., 'explain binary search')\n- System Design (e.g., 'what is load balancing')\n\nPlease be more specific about what you'd like to learn!"
    
    def _generate_with_external_llm(self, user_request: str) -> str:
        """Generate using external LLM API."""
        # Detect domain for better prompting
        domain, confidence = self.knowledge_base.detect_domain(user_request)
        
        if domain == Domain.MANIM:
            prompt = f"""Generate a complete Manim script for: {user_request}

Remember to:
1. Include proper imports (from manim import *)
2. Create a Scene class with descriptive name
3. Use the construct() method
4. Make it complete and runnable
5. Add appropriate colors and animations

Script:"""
        else:
            prompt = f"""Answer this computer science question: {user_request}

Provide a comprehensive explanation with code examples where appropriate."""

        try:
            response = self._call_llm(prompt)
            if domain == Domain.MANIM:
                return self._extract_code(response)
            return response
        except Exception as e:
            return f"Error calling external LLM: {str(e)}"
    
    def _call_llm(self, prompt: str) -> str:
        """Call the external LLM with the given prompt."""
        try:
            if self.llm_provider == "openai":
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=1500
                )
                return response.choices[0].message.content
            
            elif self.llm_provider == "anthropic":
                response = self.client.messages.create(
                    model=self.model_name,
                    max_tokens=1500,
                    temperature=0.7,
                    system=self.system_prompt,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.content[0].text
                
        except Exception as e:
            raise Exception(f"LLM call failed: {str(e)}")
    
    def _extract_code(self, response: str) -> str:
        """Extract Python code from LLM response."""
        # Try to extract code between triple backticks
        code_pattern = r'```python\n(.*?)\n```'
        match = re.search(code_pattern, response, re.DOTALL)
        
        if match:
            return match.group(1).strip()
        
        # If no code blocks found, try to extract from response
        lines = response.split('\n')
        code_lines = []
        in_code = False
        
        for line in lines:
            if line.strip().startswith('from manim import') or line.strip().startswith('import'):
                in_code = True
            
            if in_code:
                code_lines.append(line)
        
        if code_lines:
            return '\n'.join(code_lines)
        
        # Return the full response if no code extraction worked
        return response.strip()
    
    def explain_script(self, script: str) -> str:
        """Generate explanation for a script/code."""
        if self.inference_type == "enhanced":
            try:
                # Use enhanced inference for explanations
                explanation_request = f"Explain this code:\n\n{script}"
                result = self.inference_engine.generate_response(explanation_request)
                return result['response']
            except Exception as e:
                print(f"Enhanced explanation failed: {e}")
        
        # Fall back to simple explanation
        return self._generate_simple_explanation(script)
    
    def _generate_simple_explanation(self, script: str) -> str:
        """Generate a simple explanation for the script."""
        explanation = []
        lines = script.split('\n')
        
        explanation.append("This Manim script contains:")
        
        # Check for imports
        if any('import' in line for line in lines):
            explanation.append("- Proper imports for Manim functionality")
        
        # Check for class definition
        for line in lines:
            if line.strip().startswith('class ') and 'Scene' in line:
                class_name = line.split('class ')[1].split('(')[0]
                explanation.append(f"- A scene class named '{class_name}'")
        
        # Check for objects
        for obj_type in MANIM_OBJECTS:
            if obj_type.title() in script:
                explanation.append(f"- {obj_type.title()} object creation")
        
        # Check for animations
        for anim_type in MANIM_ANIMATIONS:
            if anim_type.title() in script:
                explanation.append(f"- {anim_type.title()} animation")
        
        # Check for colors
        for color in MANIM_COLORS:
            if color in script:
                explanation.append(f"- Uses {color} color")
        
        if len(explanation) == 1:
            explanation.append("- Basic Manim structure")
        
        return '\n'.join(explanation)
    
    def improve_script(self, script: str, improvement_request: str) -> str:
        """Improve an existing script based on feedback."""
        if self.inference_type == "enhanced":
            try:
                full_request = f"Improve this code based on the request:\n\nCode:\n{script}\n\nImprovement: {improvement_request}"
                result = self.inference_engine.generate_response(full_request)
                return result['response']
            except Exception as e:
                print(f"Enhanced improvement failed: {e}")
        
        # Fall back to template-based improvement
        return f"Improved version based on: {improvement_request}\n\n{script}\n\n# TODO: Apply requested improvements"
    
    def get_model_info(self) -> Dict[str, str]:
        """Get information about the current model."""
        if hasattr(self, 'inference_type'):
            return {
                "provider": f"Local LLM ({self.inference_type})",
                "model": "Enhanced Multi-Domain Transformer" if self.inference_type == "enhanced" else "Custom Manim-trained transformer",
                "status": "Local model - no API keys required",
                "domains": "Manim, DSA, System Design" if self.inference_type == "enhanced" else "Manim only"
            }
        else:
            return {
                "provider": self.llm_provider,
                "model": self.model_name,
                "status": "External API",
                "domains": "All CS domains"
            }
    
    # Backward compatibility methods
    def generate_with_template(self, pattern_name: str, **params) -> str:
        """Generate script using a predefined template (backward compatibility)."""
        if pattern_name not in COMMON_PATTERNS:
            return f"Error: Pattern '{pattern_name}' not found"
        
        try:
            pattern_code = get_pattern_code(pattern_name, **params)
            scene_name = params.get('scene_name', f'{pattern_name.title()}Scene')
            script = get_full_script(scene_name, pattern_code)
            return script
        except Exception as e:
            return f"Error generating template script: {str(e)}"
    
    def get_example_script(self, example_name: str) -> str:
        """Get a predefined example script (backward compatibility)."""
        if example_name not in EXAMPLE_SCRIPTS:
            available = list(EXAMPLE_SCRIPTS.keys())
            return f"Error: Example '{example_name}' not found. Available: {available}"
        return EXAMPLE_SCRIPTS[example_name]
    
    def list_available_objects(self) -> Dict[str, str]:
        """Get list of available Manim objects."""
        return {obj: info["description"] for obj, info in MANIM_OBJECTS.items()}
    
    def list_available_animations(self) -> Dict[str, str]:
        """Get list of available Manim animations."""
        return {anim: info["description"] for anim, info in MANIM_ANIMATIONS.items()}

# Backward compatibility - create alias
ManimAgent = EnhancedManimAgent

if __name__ == "__main__":
    # Test the enhanced agent
    try:
        agent = EnhancedManimAgent(llm_provider="enhanced")
        
        test_queries = [
            "Create a blue circle that rotates 360 degrees",
            "Explain binary search algorithm",
            "What is singleton design pattern?"
        ]
        
        for query in test_queries:
            print(f"\n=== Query: {query} ===")
            result = agent.generate_script(query)
            print(f"Response: {result[:200]}...")
        
        print(f"\nModel info: {agent.get_model_info()}")
        
    except Exception as e:
        print(f"Test failed: {e}")
        print("This is expected if the model files are not available.")