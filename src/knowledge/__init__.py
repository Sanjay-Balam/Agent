"""
Knowledge base components for different domains.
Includes Manim, DSA, System Design, and unified knowledge bases.
"""

from .multi_domain_knowledge_base import get_knowledge_base, MultiDomainKnowledgeBase
from .dsa_knowledge_base import DSAKnowledgeBase
from .system_design_knowledge_base import SystemDesignKnowledgeBase
from .knowledge_base import KnowledgeBase

__all__ = [
    'get_knowledge_base', 'MultiDomainKnowledgeBase',
    'DSAKnowledgeBase', 'SystemDesignKnowledgeBase', 'KnowledgeBase'
]