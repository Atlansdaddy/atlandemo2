"""
Expert Modules for Wave-based Cognition
Plug-and-play expertise modules that integrate with the temporal resonance engine.
"""

from .base_expert import BaseExpertModule, ExpertResponse
from .registry import ExpertRegistry
from .logic_expert import LogicExpertModule  
from .math_expert import MathExpertModule

__all__ = [
    'BaseExpertModule',
    'ExpertResponse', 
    'ExpertRegistry',
    'LogicExpertModule',
    'MathExpertModule'
]