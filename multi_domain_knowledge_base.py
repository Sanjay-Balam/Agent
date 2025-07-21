"""
Multi-Domain Knowledge Base Manager
Combines Manim, DSA, and System Design knowledge bases into a unified interface.
Provides domain detection and routing capabilities.
"""

import re
from typing import Dict, Any, List, Optional, Tuple
from enum import Enum

# Import all domain-specific knowledge bases
from knowledge_base import (
    MANIM_OBJECTS, MANIM_ANIMATIONS, MANIM_COLORS, MANIM_POSITIONS,
    COMMON_PATTERNS as MANIM_PATTERNS, EXAMPLE_SCRIPTS,
    get_object_code, get_animation_code, get_pattern_code, get_full_script
)
from dsa_knowledge_base import (
    DATA_STRUCTURES, ALGORITHMS, DSA_PATTERNS, COMMON_QUESTIONS,
    get_data_structure_info, get_algorithm_info, get_pattern_info, get_question_solution
)
from system_design_knowledge_base import (
    LLD_PATTERNS, HLD_COMPONENTS, SYSTEM_DESIGN_PATTERNS, SCALABILITY_PATTERNS,
    get_lld_pattern_info, get_hld_component_info, get_system_pattern_info, get_scalability_info
)

class Domain(Enum):
    MANIM = "manim"
    DSA = "dsa"
    SYSTEM_DESIGN = "system_design"
    UNKNOWN = "unknown"

class MultiDomainKnowledgeBase:
    """Unified knowledge base manager for all CS domains."""
    
    def __init__(self):
        # Domain detection keywords
        self.domain_keywords = {
            Domain.MANIM: [
                'manim', 'animation', 'scene', 'circle', 'square', 'text', 'mathtext',
                'create', 'write', 'fadein', 'fadeout', 'transform', 'rotate', 'scale',
                'move', 'color', 'blue', 'red', 'green', 'mathematical', 'formula',
                'visualize', 'animate', 'draw', 'show', 'display'
            ],
            Domain.DSA: [
                'algorithm', 'data structure', 'array', 'linked list', 'tree', 'graph',
                'stack', 'queue', 'heap', 'hash table', 'sort', 'search', 'binary search',
                'merge sort', 'quick sort', 'bubble sort', 'dynamic programming', 'dp',
                'recursion', 'iteration', 'complexity', 'big o', 'time complexity',
                'space complexity', 'leetcode', 'coding', 'implementation'
            ],
            Domain.SYSTEM_DESIGN: [
                'system design', 'architecture', 'scalability', 'load balancer', 
                'database', 'caching', 'microservices', 'api', 'distributed',
                'high level design', 'low level design', 'hld', 'lld',
                'design pattern', 'singleton', 'factory', 'observer', 'strategy',
                'circuit breaker', 'event sourcing', 'cqrs', 'message queue',
                'horizontal scaling', 'vertical scaling', 'sharding', 'replication'
            ]
        }
        
        # Initialize combined knowledge base
        self.all_topics = self._build_combined_topics()
    
    def _build_combined_topics(self) -> Dict[str, Dict]:
        """Build a combined dictionary of all topics from all domains."""
        combined = {}
        
        # Add Manim topics
        for topic in MANIM_OBJECTS:
            combined[f"manim_{topic}"] = {
                "domain": Domain.MANIM,
                "type": "object",
                "data": MANIM_OBJECTS[topic]
            }
        
        for topic in MANIM_ANIMATIONS:
            combined[f"manim_{topic}"] = {
                "domain": Domain.MANIM,
                "type": "animation", 
                "data": MANIM_ANIMATIONS[topic]
            }
        
        # Add DSA topics
        for topic in DATA_STRUCTURES:
            combined[f"dsa_{topic}"] = {
                "domain": Domain.DSA,
                "type": "data_structure",
                "data": DATA_STRUCTURES[topic]
            }
        
        for topic in ALGORITHMS:
            combined[f"dsa_{topic}"] = {
                "domain": Domain.DSA,
                "type": "algorithm",
                "data": ALGORITHMS[topic]
            }
        
        # Add System Design topics
        for topic in LLD_PATTERNS:
            combined[f"lld_{topic}"] = {
                "domain": Domain.SYSTEM_DESIGN,
                "type": "lld_pattern",
                "data": LLD_PATTERNS[topic]
            }
        
        for topic in HLD_COMPONENTS:
            combined[f"hld_{topic}"] = {
                "domain": Domain.SYSTEM_DESIGN,
                "type": "hld_component",
                "data": HLD_COMPONENTS[topic]
            }
        
        return combined
    
    def detect_domain(self, query: str) -> Tuple[Domain, float]:
        """
        Detect the most likely domain for a query.
        Returns domain and confidence score.
        """
        query_lower = query.lower()
        domain_scores = {domain: 0 for domain in Domain}
        
        for domain, keywords in self.domain_keywords.items():
            for keyword in keywords:
                if keyword in query_lower:
                    # Give higher weight to exact matches
                    if keyword == query_lower.strip():
                        domain_scores[domain] += 10
                    else:
                        domain_scores[domain] += 1
        
        # Find domain with highest score
        best_domain = max(domain_scores, key=domain_scores.get)
        max_score = domain_scores[best_domain]
        
        # Calculate confidence (normalize by total keywords found)
        total_score = sum(domain_scores.values())
        confidence = max_score / total_score if total_score > 0 else 0
        
        return best_domain if max_score > 0 else Domain.UNKNOWN, confidence
    
    def get_topic_info(self, topic_name: str, domain_hint: Optional[Domain] = None) -> Optional[Dict]:
        """Get information about a specific topic, optionally with domain hint."""
        topic_lower = topic_name.lower().replace(' ', '_')
        
        # If domain hint provided, search in that domain first
        if domain_hint:
            if domain_hint == Domain.MANIM:
                # Search Manim objects and animations
                if topic_lower in MANIM_OBJECTS:
                    return {"domain": Domain.MANIM, "type": "object", "data": MANIM_OBJECTS[topic_lower]}
                if topic_lower in MANIM_ANIMATIONS:
                    return {"domain": Domain.MANIM, "type": "animation", "data": MANIM_ANIMATIONS[topic_lower]}
            
            elif domain_hint == Domain.DSA:
                # Search DSA structures and algorithms
                if topic_lower in DATA_STRUCTURES:
                    return {"domain": Domain.DSA, "type": "data_structure", "data": DATA_STRUCTURES[topic_lower]}
                if topic_lower in ALGORITHMS:
                    return {"domain": Domain.DSA, "type": "algorithm", "data": ALGORITHMS[topic_lower]}
                if topic_lower in COMMON_QUESTIONS:
                    return {"domain": Domain.DSA, "type": "question", "data": COMMON_QUESTIONS[topic_lower]}
            
            elif domain_hint == Domain.SYSTEM_DESIGN:
                # Search System Design patterns and components
                if topic_lower in LLD_PATTERNS:
                    return {"domain": Domain.SYSTEM_DESIGN, "type": "lld_pattern", "data": LLD_PATTERNS[topic_lower]}
                if topic_lower in HLD_COMPONENTS:
                    return {"domain": Domain.SYSTEM_DESIGN, "type": "hld_component", "data": HLD_COMPONENTS[topic_lower]}
        
        # Search all domains
        for full_topic_name, topic_data in self.all_topics.items():
            if topic_lower in full_topic_name or full_topic_name.endswith(topic_lower):
                return topic_data
        
        return None
    
    def search_topics(self, query: str, max_results: int = 10) -> List[Dict]:
        """Search for topics across all domains matching the query."""
        query_lower = query.lower()
        results = []
        
        for topic_name, topic_data in self.all_topics.items():
            # Check if query matches topic name or description
            score = 0
            
            # Direct name match
            if query_lower in topic_name.lower():
                score += 10
            
            # Check in description/definition
            data = topic_data.get("data", {})
            if isinstance(data, dict):
                definition = data.get("definition", "").lower()
                description = data.get("description", "").lower()
                
                if query_lower in definition:
                    score += 5
                if query_lower in description:
                    score += 3
                
                # Check use cases
                use_cases = data.get("use_cases", [])
                if isinstance(use_cases, list):
                    for use_case in use_cases:
                        if query_lower in str(use_case).lower():
                            score += 2
            
            if score > 0:
                results.append({
                    "topic": topic_name,
                    "score": score,
                    "domain": topic_data["domain"].value,
                    "type": topic_data["type"],
                    "data": topic_data["data"]
                })
        
        # Sort by score and return top results
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:max_results]
    
    def generate_response(self, query: str) -> Dict[str, Any]:
        """Generate a comprehensive response for a query."""
        # Detect domain
        domain, confidence = self.detect_domain(query)
        
        # Search for relevant topics
        search_results = self.search_topics(query)
        
        response = {
            "query": query,
            "detected_domain": domain.value,
            "confidence": confidence,
            "results": search_results,
            "suggestions": []
        }
        
        # Add domain-specific suggestions
        if domain == Domain.MANIM:
            response["suggestions"] = [
                "Try asking for specific animations like 'create a blue circle'",
                "Ask for mathematical formulas like 'show equation E=mc²'",
                "Request transformations like 'transform circle to square'"
            ]
        elif domain == Domain.DSA:
            response["suggestions"] = [
                "Ask about data structures like 'explain binary tree'",
                "Request algorithm implementations like 'show merge sort'",
                "Ask for complexity analysis like 'time complexity of quick sort'"
            ]
        elif domain == Domain.SYSTEM_DESIGN:
            response["suggestions"] = [
                "Ask about design patterns like 'explain singleton pattern'",
                "Request system components like 'how does load balancer work'",
                "Ask about scalability like 'horizontal vs vertical scaling'"
            ]
        
        return response
    
    def get_domain_summary(self, domain: Domain) -> Dict[str, Any]:
        """Get a summary of available topics in a domain."""
        if domain == Domain.MANIM:
            return {
                "domain": domain.value,
                "objects": list(MANIM_OBJECTS.keys()),
                "animations": list(MANIM_ANIMATIONS.keys()),
                "patterns": list(MANIM_PATTERNS.keys()),
                "total_topics": len(MANIM_OBJECTS) + len(MANIM_ANIMATIONS) + len(MANIM_PATTERNS)
            }
        elif domain == Domain.DSA:
            return {
                "domain": domain.value,
                "data_structures": list(DATA_STRUCTURES.keys()),
                "algorithms": list(ALGORITHMS.keys()),
                "patterns": list(DSA_PATTERNS.keys()),
                "questions": list(COMMON_QUESTIONS.keys()),
                "total_topics": len(DATA_STRUCTURES) + len(ALGORITHMS) + len(DSA_PATTERNS) + len(COMMON_QUESTIONS)
            }
        elif domain == Domain.SYSTEM_DESIGN:
            return {
                "domain": domain.value,
                "lld_patterns": list(LLD_PATTERNS.keys()),
                "hld_components": list(HLD_COMPONENTS.keys()),
                "system_patterns": list(SYSTEM_DESIGN_PATTERNS.keys()),
                "scalability_patterns": list(SCALABILITY_PATTERNS.keys()),
                "total_topics": len(LLD_PATTERNS) + len(HLD_COMPONENTS) + len(SYSTEM_DESIGN_PATTERNS) + len(SCALABILITY_PATTERNS)
            }
        
        return {"domain": domain.value, "error": "Domain not supported"}
    
    def get_all_domains_summary(self) -> Dict[str, Any]:
        """Get summary of all available domains and their topics."""
        return {
            "total_domains": len(Domain) - 1,  # Exclude UNKNOWN
            "domains": {
                Domain.MANIM.value: self.get_domain_summary(Domain.MANIM),
                Domain.DSA.value: self.get_domain_summary(Domain.DSA),
                Domain.SYSTEM_DESIGN.value: self.get_domain_summary(Domain.SYSTEM_DESIGN)
            },
            "total_topics": len(self.all_topics)
        }

# Convenience functions for backward compatibility and easy access
def detect_query_domain(query: str) -> Tuple[str, float]:
    """Detect domain for a query. Returns domain name and confidence."""
    kb = MultiDomainKnowledgeBase()
    domain, confidence = kb.detect_domain(query)
    return domain.value, confidence

def search_all_domains(query: str, max_results: int = 10) -> List[Dict]:
    """Search across all domains for a query."""
    kb = MultiDomainKnowledgeBase()
    return kb.search_topics(query, max_results)

def get_comprehensive_response(query: str) -> Dict[str, Any]:
    """Get a comprehensive response including domain detection and search results."""
    kb = MultiDomainKnowledgeBase()
    return kb.generate_response(query)

# Global instance for easy access
_knowledge_base_instance = None

def get_knowledge_base() -> MultiDomainKnowledgeBase:
    """Get the global knowledge base instance (singleton pattern)."""
    global _knowledge_base_instance
    if _knowledge_base_instance is None:
        _knowledge_base_instance = MultiDomainKnowledgeBase()
    return _knowledge_base_instance

if __name__ == "__main__":
    # Test the multi-domain knowledge base
    kb = MultiDomainKnowledgeBase()
    
    # Test domain detection
    test_queries = [
        "Create a blue circle animation",
        "Explain binary search algorithm",
        "How does load balancer work in system design",
        "Show me merge sort implementation",
        "What is singleton design pattern"
    ]
    
    print("=== Domain Detection Test ===")
    for query in test_queries:
        domain, confidence = kb.detect_domain(query)
        print(f"Query: '{query}'")
        print(f"Domain: {domain.value}, Confidence: {confidence:.2f}")
        print()
    
    # Test search functionality
    print("=== Search Test ===")
    search_query = "sorting"
    results = kb.search_topics(search_query)
    print(f"Search results for '{search_query}':")
    for result in results[:3]:  # Show top 3
        print(f"- {result['topic']} (Domain: {result['domain']}, Score: {result['score']})")
    print()
    
    # Test comprehensive response
    print("=== Comprehensive Response Test ===")
    response = kb.generate_response("binary tree")
    print(f"Query: {response['query']}")
    print(f"Detected Domain: {response['detected_domain']}")
    print(f"Confidence: {response['confidence']:.2f}")
    print(f"Found {len(response['results'])} results")
    
    # Test domain summary
    print("\n=== Domain Summary ===")
    summary = kb.get_all_domains_summary()
    print(f"Total domains: {summary['total_domains']}")
    print(f"Total topics: {summary['total_topics']}")
    for domain_name, domain_info in summary['domains'].items():
        print(f"{domain_name}: {domain_info['total_topics']} topics")