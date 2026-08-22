"""
LTTS Miners System

Specialized data extraction modules that process raw context data on-demand.
Refined from the original collectors with improved architecture alignment.
"""

from .attention_miner import AttentionMiner
from .moe_miner import MoEMiner
from .ffn_miner import FFNMiner
from .embeddings_miner import EmbeddingsMiner
from .normalization_miner import NormalizationMiner

# Register all miners with the global registry
from scripts.core.miners_registry import register_miner

register_miner("attention", AttentionMiner)
register_miner("moe", MoEMiner)
register_miner("ffn", FFNMiner)
register_miner("embeddings", EmbeddingsMiner)
register_miner("normalization", NormalizationMiner)

__all__ = [
    'AttentionMiner',
    'MoEMiner',
    'FFNMiner',
    'EmbeddingsMiner',
    'NormalizationMiner'
]
