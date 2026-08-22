from pydantic import BaseModel
from typing import Optional, List, Union, Any, Dict
import torch
import torch.nn as nn
import json
import logging

from scripts.core.ltts_model import LTTSModel
from scripts.models.architecture_adapter import get_architecture_adapter


class ModelInput(BaseModel):
    id: str
    engine: str


class ModelStruct:
    def __init__(self, name: str, type: str, layers: Optional[List['ModelStruct']] = None):
        self.name: str = name
        self.type: str = type
        self.layers: List['ModelStruct'] = layers if layers is not None else []

        # Optional attributes for UI display
        self.parameters: Optional[int] = None
        self.inputShape: Optional[List[Union[int, str]]] = None
        self.outputShape: Optional[List[Union[int, str]]] = None
        self.numHeads: Optional[int] = None
        self.attentionType: Optional[str] = None
        self.hasResidual: Optional[bool] = None
        self.blockType: Optional[str] = None
        self.moduleRole: Optional[str] = None
        self.architectureStage: Optional[str] = None
        self.layerIndex: Optional[int] = None
        self.attentionKind: Optional[str] = None
        self.attachable: Optional[bool] = None

        # Model-level architecture metadata
        self.architecture: Optional[str] = None
        self.architectureFamily: Optional[str] = None
        self.architectureTopology: Optional[str] = None
        self.modelType: Optional[str] = None
        self.capabilities: Optional[Dict[str, bool]] = None

        # Layer inspector information
        self.inspector_info: Optional[Dict[str, Any]] = None

        # Breadcrumb navigation support
        self.breadcrumb: Optional[List[Dict[str, str]]] = None
        self.fullPath: Optional[str] = None
        self.depth: Optional[int] = None

        # Hierarchy information for visual indentation
        self.hierarchy: Optional[Dict[str, Any]] = None

        # Aggregated metadata
        self.architecturalLayerCount: Optional[int] = None
        self.detailedLayerCount: Optional[int] = None

    def add_child(self, node):
        if not isinstance(node, ModelStruct):
            raise ValueError("not a ModelStruct instance")
        self.layers.append(node)

    def to_dict(self):
        result = {
            "name": self.name,
            "type": self.type,
            "layers": [child.to_dict() for child in self.layers],
        }

        # Add optional attributes if they exist
        optional_attrs = [
            'parameters',
            'inputShape',
            'outputShape',
            'numHeads',
            'attentionType',
            'hasResidual',
            'blockType',
            'moduleRole',
            'architectureStage',
            'layerIndex',
            'attentionKind',
            'attachable',
            'architecture',
            'architectureFamily',
            'architectureTopology',
            'modelType',
            'capabilities',
            'categorization',
            'breadcrumb',
            'fullPath',
            'depth',
            'inspector_info',
            'hierarchy',
            'architecturalLayerCount',
            'detailedLayerCount',
        ]
        for attr in optional_attrs:
            if hasattr(self, attr) and getattr(self, attr) is not None:
                result[attr] = getattr(self, attr)

        return result


def traverse_model(model: nn.Module, parent_name: str = "model", max_traversal: bool = True) -> ModelStruct:
    """
    Unified model structure analysis with maximum traversal.

    Builds the model-layer tree used by the UI and analysis modules.
    All modules are flattened and categorized for easy access.

    Args:
        model: The model to analyze (LTTSModel or nn.Module)
        parent_name: Name for the root node
        max_traversal: If True, shows all layers flattened. If False, preserves hierarchy.

    Returns:
        ModelStruct containing the layer tree
    """
    # Unwrap LTTSModel to access the underlying nn.Module
    base_model = model.original_model if isinstance(model, LTTSModel) else model
    architecture_adapter = (
        model.architecture_adapter
        if isinstance(model, LTTSModel) and hasattr(model, "architecture_adapter")
        else get_architecture_adapter(base_model)
    )

    # Root node
    root = ModelStruct(parent_name, type(base_model).__name__)
    architecture = architecture_adapter.descriptor
    root.architecture = architecture.display_name
    root.architectureFamily = architecture.family
    root.architectureTopology = architecture.topology
    root.modelType = architecture.model_type
    root.capabilities = architecture.capabilities.to_dict()

    # The UI decides module compatibility from this map, and the backend
    # rejects an attachment using its own richer one. Anything the backend
    # evaluates but this map omits reads as `undefined` in the browser and the
    # check silently passes - the user picks a module and the run fails with a
    # 422 instead. Merge the runtime-derived verdicts in so both sides see the
    # same keys.
    try:
        from scripts.services.model_compatibility import inspect_model_capabilities

        for name, verdict in inspect_model_capabilities(model).items():
            root.capabilities.setdefault(name, bool(verdict.get("supported")))
    except Exception as capability_error:  # pragma: no cover - defensive
        logging.getLogger(__name__).debug(
            "Could not merge runtime capabilities: %s", capability_error
        )

    # Add model-level metadata
    try:
        root.parameters = sum(p.numel() for p in base_model.parameters())
        root.hasResidual = "residual" in type(base_model).__name__.lower()
    except Exception as e:
        logging.getLogger(__name__).debug("Error counting model parameters: %s", e)

    if max_traversal:
        # Maximum traversal: flatten all modules for UI display
        return _build_flat_structure(
            base_model,
            root,
            model if isinstance(model, LTTSModel) else None,
            architecture_adapter=architecture_adapter,
        )
    else:
        # Hierarchical structure (legacy mode)
        return _build_hierarchical_structure(base_model, root)

def _build_flat_structure(
    base_model: nn.Module,
    root: ModelStruct,
    ltts_model=None,
    architecture_adapter=None,
) -> ModelStruct:
    """Build flat structure with hierarchical names for terminal (leaf) layers only."""

    # Collect all terminal (leaf) modules AND important parent modules
    modules_to_include: List[tuple[str, nn.Module, bool]] = []
    architectural_blocks: set[str] = set()

    for name, module in base_model.named_modules():
        if name == "":
            continue  # skip root

        # Check if this module has any children
        has_children = len(list(module.children())) > 0
        descriptor = architecture_adapter.describe_module(name, module)
        if descriptor.is_block:
            architectural_blocks.add(name)

        # Include terminal modules (no children) - these are ALL the leaf layers we need to show
        if not has_children:
            modules_to_include.append((name, module, has_children))
        # Also include meaningful parent components (blocks, attention, MLP/MoE).
        # Leaf-only trees hide the architecture and make attachment targets
        # impossible to understand before an LTTS wrapper has been created.
        elif descriptor.attachable:
            modules_to_include.append((name, module, has_children))

    # Get layer inspector data if available
    inspector_data = {}
    if ltts_model and hasattr(ltts_model, 'context') and ltts_model.context:
        model_state = getattr(ltts_model.context, 'model_state', {})
        inspector_data = model_state.get('layer_inspector_data', {})

    # Add all modules to the root as flat structure
    # but preserve hierarchical names for easy filtering
    hierarchy_reference = [(module_name, module_obj) for module_name, module_obj, _ in modules_to_include]

    for name, module, has_children in modules_to_include:
        module_info = _analyze_module(
            name, module, ltts_model, architecture_adapter=architecture_adapter
        )

        if has_children:
            module_info['parameters'] = 0
            module_info['trainable_parameters'] = 0
            module_info['is_wrapper_group'] = True

        # Add hierarchy information
        module_info['hierarchy'] = _calculate_hierarchy_info(name, hierarchy_reference)

        layer_struct = _create_layer_struct(module_info)

        # Inject inspector info if available for this layer
        if name in inspector_data:
            layer_struct.inspector_info = inspector_data[name]

        root.add_child(layer_struct)

    if architectural_blocks:
        root.architecturalLayerCount = len(architectural_blocks)
    root.detailedLayerCount = len(root.layers)

    return root


def _calculate_hierarchy_info(name: str, all_modules: list) -> dict:
    """Calculate hierarchy information for a module."""
    parts = name.split('.')
    depth = len(parts)

    # Create a set of all module names for faster lookup
    module_names = set()
    for entry in all_modules:
        if isinstance(entry, tuple) and entry:
            module_names.add(entry[0])
        else:
            module_names.add(entry)

    # Find immediate parent (one level up)
    parent_name = None
    if depth > 1:
        parent_name = '.'.join(parts[:-1])
        # Only consider it a parent if it actually exists in our displayed modules
        if parent_name not in module_names:
            parent_name = None

    # Find direct children (one level down)
    children = []
    for module_name in module_names:
        if (module_name.startswith(name + '.') and
            module_name.count('.') == name.count('.') + 1):
            children.append(module_name)

    # A module is a child if its immediate parent exists in our displayed list
    # This creates the natural HuggingFace-style hierarchy where:
    # - Parent modules (like self_attn) show with their children indented under them
    # - Child modules (like q_proj, k_proj) are visually distinguished as children
    is_child = parent_name is not None and parent_name in module_names


    return {
        'depth': depth,
        'parent': parent_name,
        'children': children,
        'is_child': is_child,
        'has_children': len(children) > 0,
        'is_leaf': len(children) == 0
    }


def _extract_architecture_block(name: str) -> Optional[str]:
    """Extract block-level identifier (e.g., transformer.layers.0) from a module path."""
    parts = name.split('.')
    for idx, part in enumerate(parts):
        if part.isdigit():
            return '.'.join(parts[: idx + 1])
    return None


def _analyze_module(
    name: str,
    module: torch.nn.Module,
    ltts_model=None,
    architecture_adapter=None,
) -> dict:
    """Build module metadata for the shared structure."""

    module_type = type(module).__name__

    # Calculate parameters
    try:
        param_count = sum(p.numel() for p in module.parameters())
        trainable_params = sum(p.numel() for p in module.parameters() if p.requires_grad)
    except Exception:
        param_count = 0
        trainable_params = 0

    # Check if wrapped (for LTTSModel)
    is_wrapped = False
    if ltts_model and hasattr(ltts_model, 'layer_wrappers'):
        is_wrapped = name in ltts_model.layer_wrappers

    # Get shape information
    shapes = {}
    try:
        if hasattr(module, 'weight') and module.weight is not None:
            shapes["weight_shape"] = list(module.weight.shape)
        if hasattr(module, 'bias') and module.bias is not None:
            shapes["bias_shape"] = list(module.bias.shape)
        if hasattr(module, 'num_heads'):
            shapes["num_heads"] = module.num_heads
    except Exception:
        pass  # Shape extraction is best-effort for display only

    descriptor = (
        architecture_adapter.describe_module(name, module)
        if architecture_adapter is not None
        else None
    )

    return {
        "name": name,
        "type": module_type,
        "parameters": param_count,
        "trainable_parameters": trainable_params,
        "is_wrapped": is_wrapped,
        "has_children": len(list(module.children())) > 0,
        "shapes": shapes,
        "module": module,  # Keep reference for detailed analysis
        "descriptor": descriptor.to_dict() if descriptor is not None else {},
    }


def _create_layer_struct(layer_info: dict) -> ModelStruct:
    """Create ModelStruct from analyzed layer info."""

    struct = ModelStruct(layer_info["name"], layer_info["type"])

    # Add module metadata
    struct.parameters = layer_info["parameters"]
    struct.hasResidual = "residual" in layer_info["type"].lower()
    descriptor = layer_info.get("descriptor") or {}
    struct.moduleRole = descriptor.get("role")
    struct.architectureStage = descriptor.get("stage")
    struct.layerIndex = descriptor.get("layer_index")
    struct.attentionKind = descriptor.get("attention_kind")
    struct.attachable = bool(descriptor.get("attachable"))

    # Add hierarchical navigation support
    struct.fullPath = layer_info["name"]
    struct.breadcrumb = _create_breadcrumb(layer_info["name"])
    struct.depth = len(struct.breadcrumb) if struct.breadcrumb else 0

    # Add shape information
    shapes = layer_info["shapes"]
    if "weight_shape" in shapes:
        struct.inputShape = shapes["weight_shape"]
        struct.outputShape = shapes["weight_shape"]
    if "num_heads" in shapes:
        struct.numHeads = shapes["num_heads"]
        struct.attentionType = "multi-head"

    role_to_block_type = {
        "block": "transformer",
        "self_attention": "attention",
        "cross_attention": "cross_attention",
        "attention_projection": "attention_projection",
        "mlp": "feed_forward",
        "moe_router": "moe_router",
        "moe_expert": "moe_expert",
        "embedding": "embedding",
        "positional_embedding": "embedding",
        "normalization": "normalization",
        "activation": "activation",
        "output": "output",
        "dropout": "dropout",
        "other": "other",
    }
    struct.blockType = role_to_block_type.get(struct.moduleRole or "other", "other")

    # Add hierarchy information if available
    if "hierarchy" in layer_info and layer_info["hierarchy"]:
        struct.hierarchy = layer_info["hierarchy"]

    return struct


def _create_breadcrumb(module_path: str) -> List[Dict[str, str]]:
    """Create breadcrumb navigation from hierarchical module path."""
    if not module_path:
        return []

    breadcrumb = []
    path_parts = module_path.split('.')

    # Create clickable breadcrumb segments
    for i, part in enumerate(path_parts):
        # Build partial path for filtering
        partial_path = '.'.join(path_parts[:i+1])

        # Create user-friendly display names
        display_name = _get_breadcrumb_display_name(part, i, path_parts)

        breadcrumb.append({
            "name": display_name,
            "path": partial_path,
            "level": i,
            "isLeaf": i == len(path_parts) - 1
        })

    return breadcrumb


def _get_breadcrumb_display_name(part: str, level: int, full_path: List[str]) -> str:
    """Generate user-friendly display names for breadcrumb parts."""
    # Common transformer component name mappings
    name_mappings = {
        'model': '🏗️ Model',
        'transformer': '🔄 Transformer',
        'encoder': '📥 Encoder',
        'decoder': '📤 Decoder',
        'layers': '📚 Layers',
        'h': '🔢 Blocks',
        'attention': '👁️ Attention',
        'self_attn': '👁️ Self-Attn',
        'cross_attn': '🔄 Cross-Attn',
        'mlp': '🔗 MLP',
        'feed_forward': '🔗 FFN',
        'norm': '📏 Norm',
        'layernorm': '📏 LayerNorm',
        'rmsnorm': '📏 RMSNorm',
        'embedding': '📝 Embeddings',
        'embeddings': '📝 Embeddings',
        'lm_head': '🎯 LM Head',
        'output': '🎯 Output',
        'q_proj': 'Q',
        'k_proj': 'K',
        'v_proj': 'V',
        'o_proj': 'O',
        'out_proj': 'O',
        'fc1': 'FC1',
        'fc2': 'FC2',
        'gate_proj': 'Gate',
        'up_proj': 'Up',
        'down_proj': 'Down'
    }

    # Check if it's a numeric index (like h.0, layers.12)
    if part.isdigit():
        # Get parent name for context
        parent_name = full_path[level-1] if level > 0 else ''
        if parent_name in ['h', 'layers', 'layer']:
            return f"#{part}"
        else:
            return f"[{part}]"

    # Apply mapping if available
    if part.lower() in name_mappings:
        return name_mappings[part.lower()]

    # Default: capitalize and clean up
    return part.replace('_', ' ').title()

def _build_hierarchical_structure(base_model: nn.Module, root: ModelStruct) -> ModelStruct:
    """Build traditional hierarchical structure (legacy mode)."""

    # Build nested dict tree from dotted module names
    tree = {}
    for name, module in base_model.named_modules():
        if name == "":
            continue  # skip root

        parts = name.split(".")
        cursor = tree
        for i, part in enumerate(parts):
            if part not in cursor:
                cursor[part] = {"__module__": None, "__children__": {}}
            if i == len(parts) - 1:
                cursor[part]["__module__"] = module
            cursor = cursor[part]["__children__"]

    def build_struct(name: str, node: dict) -> ModelStruct:
        mod = node.get("__module__")
        mtype = type(mod).__name__ if mod is not None else "Module"
        ms = ModelStruct(name, mtype)
        for child_name, child_node in node.get("__children__", {}).items():
            ms.add_child(build_struct(child_name, child_node))
        return ms

    for child_name, child_node in tree.items():
        root.add_child(build_struct(child_name, child_node))

    return root


class LayerCategorizer:
    """
    Dynamic layer categorization system that analyzes model structure
    and creates meaningful categories based on actual layer patterns.
    """

    def __init__(self):
        # Define category patterns based on common transformer architectures
        self.category_patterns = {
            'embeddings': {
                'name': 'Embedding Layers',
                'description': 'Vector representations for tokens and positional encodings in high-dimensional space',
                'priority': 1,
                'patterns': [
                    'embedding', 'embed', 'token', 'position', 'pos', 'wte', 'wpe',
                    'word_embeddings', 'position_embeddings'
                ],
                'icon': 'E',
                'color': '#8B9DC3'  # Muted blue-gray
            },
            'input_processing': {
                'name': 'Input Processing Layers',
                'description': 'Preprocessing operations including normalization and regularization of input sequences',
                'priority': 2,
                'patterns': ['input_dropout', 'input_norm'],
                'icon': 'I',
                'color': '#9AAFB8'  # Muted teal-gray
            },
            'attention_mechanisms': {
                'name': 'Attention Mechanisms',
                'description': 'Multi-head self-attention and cross-attention layers implementing scaled dot-product attention',
                'priority': 3,
                'patterns': [
                    'attention', 'attn', 'self_attention', 'cross_attention',
                    'multihead_attention', 'mha', 'attention_layer',
                    'q_proj', 'k_proj', 'v_proj', 'out_proj', 'o_proj',
                    'rotary', 'rope', 'rotaryembedding', 'q_norm', 'k_norm'
                ],
                'icon': 'A',
                'color': '#A8A8A8'  # Muted gray
            },
            'transformer_blocks': {
                'name': 'Transformer Blocks',
                'description': 'Complete transformer encoder/decoder blocks with residual connections and layer normalization',
                'priority': 4,
                'patterns': [
                    'transformer', 'encoder', 'decoder', 'block', 'layer',
                    'bert_layer', 'gpt_block', 't5_block', 'h.', 'layers.'
                ],
                'icon': 'T',
                'color': '#B0B0B0'  # Muted silver-gray
            },
            'feed_forward': {
                'name': 'Feed-Forward Networks',
                'description': 'Multi-layer perceptrons with non-linear activations for position-wise transformations',
                'priority': 5,
                'patterns': [
                    'feed_forward', 'ffn', 'mlp', 'dense', 'linear',
                    'intermediate', 'output', 'projection', 'fc1', 'fc2'
                ],
                'icon': 'F',
                'color': '#A0A0A0'  # Muted dark gray
            },
            'normalization': {
                'name': 'Normalization Layers',
                'description': 'Statistical normalization techniques including LayerNorm, RMSNorm, and BatchNorm',
                'priority': 6,
                'patterns': [
                    'norm', 'layernorm', 'rmsnorm', 'groupnorm', 'batchnorm',
                    'instancenorm', 'ln', 'rms', 'layer_norm'
                ],
                'icon': 'N',
                'color': '#C0C0C0'  # Muted light gray
            },
            'activation_functions': {
                'name': 'Activation Functions',
                'description': 'Non-linear activation functions and regularization layers including ReLU, GELU, and Dropout',
                'priority': 7,
                'patterns': [
                    'activation', 'relu', 'gelu', 'sigmoid', 'tanh',
                    'softmax', 'dropout'
                ],
                'icon': 'R',
                'color': '#B8B8B8'  # Muted medium gray
            },
            'moe_layers': {
                'name': 'Mixture of Experts',
                'description': 'Sparse expert routing layers with gating mechanisms for conditional computation',
                'priority': 8,
                'patterns': ['moe', 'expert', 'mixture', 'router', 'gate'],
                'icon': 'M',
                'color': '#A8A8A8'  # Muted darker gray
            },
            'output_processing': {
                'name': 'Output Processing Layers',
                'description': 'Final prediction layers including classification heads and language modeling outputs',
                'priority': 9,
                'patterns': [
                    'output', 'prediction', 'classifier', 'lm_head',
                    'pooler', 'sequence_summary', 'logits'
                ],
                'icon': 'O',
                'color': '#9B9B9B'  # Muted slate gray
            }
        }

    def categorize_layers(self, layers):
        """
        Categorize layers based on their types and names.

        Args:
            layers: List of ModelStruct objects

        Returns:
            dict: Categories with their layers
        """
        categories = {}

        # Initialize all categories
        for cat_key, cat_info in self.category_patterns.items():
            categories[cat_key] = {
                **cat_info,
                'layers': [],
                'totalCount': 0,
                'totalParameters': 0
            }

        # Add default category for uncategorized layers
        categories['other'] = {
            'name': '📦 Other Layers',
            'description': 'Uncategorized layers',
            'priority': 999,
            'patterns': [],
            'icon': '📦',
            'color': '#9E9E9E',
            'layers': [],
            'totalCount': 0,
            'totalParameters': 0
        }

        # Categorize each layer
        for layer in layers:
            category = self._get_layer_category(layer)
            if category not in categories:
                # Create category if it doesn't exist (shouldn't happen with proper patterns)
                categories[category] = {
                    'name': f'📦 {category.title()}',
                    'description': f'Auto-detected {category} layers',
                    'priority': 998,
                    'patterns': [],
                    'icon': '📦',
                    'color': '#9E9E9E',
                    'layers': [],
                    'totalCount': 0,
                    'totalParameters': 0
                }

            # Store layer reference instead of object to avoid circular references.
            # The UI renders its layer cards from these entries, so anything the
            # compatibility check needs has to travel with them: without
            # moduleRole the card cannot tell which modules target this layer and
            # offers every module on every layer.
            layer_ref = {
                'name': layer.name,
                'type': layer.type,
                'parameters': layer.parameters,
                'index': len(categories[category]['layers']),
                'breadcrumb': layer.breadcrumb,
                'fullPath': layer.fullPath,
                'depth': layer.depth,
                'blockType': layer.blockType,
                'moduleRole': layer.moduleRole,
                'architectureStage': layer.architectureStage,
                'layerIndex': layer.layerIndex,
                'attachable': getattr(layer, 'attachable', None),
                # Displayed on the card when present. They lived only on the
                # flat layer list, so the card's head count, attention type,
                # tensor shapes and residual flag never rendered for any model.
                'numHeads': getattr(layer, 'numHeads', None),
                'attentionType': getattr(layer, 'attentionType', None),
                'attentionKind': getattr(layer, 'attentionKind', None),
                'hasResidual': getattr(layer, 'hasResidual', None),
                'inputShape': getattr(layer, 'inputShape', None),
                'outputShape': getattr(layer, 'outputShape', None),
            }
            layer_ref = {k: v for k, v in layer_ref.items() if v is not None}

            # Preserve hierarchy information if available
            if hasattr(layer, 'hierarchy') and layer.hierarchy:
                layer_ref['hierarchy'] = layer.hierarchy
            categories[category]['layers'].append(layer_ref)
            categories[category]['totalCount'] += 1
            categories[category]['totalParameters'] += (layer.parameters or 0)

        # Remove empty categories
        categories = {k: v for k, v in categories.items() if v['totalCount'] > 0}

        return categories

    def _get_layer_category(self, layer):
        """
        Determine the category for a single layer.

        Args:
            layer: ModelStruct object

        Returns:
            str: Category key
        """
        search_text = f"{layer.type} {layer.name}".lower()

        # Sort categories by priority (lower number = higher priority)
        sorted_categories = sorted(
            self.category_patterns.items(),
            key=lambda x: x[1].get('priority', 999)
        )

        # Check each category for matches in priority order
        for cat_key, cat_info in sorted_categories:
            for pattern in cat_info['patterns']:
                if self._pattern_matches(pattern.lower(), search_text):
                    return cat_key

        # Special handling for transformer block patterns
        if self._is_transformer_block(layer):
            return 'transformer_blocks'

        return 'other'

    def _pattern_matches(self, pattern, search_text):
        """
        Check if pattern matches search_text with improved logic to avoid false positives.

        Args:
            pattern: Pattern to match
            search_text: Text to search in

        Returns:
            bool: True if pattern matches appropriately
        """
        # For embedding patterns, be more specific to avoid matching rotary_embed, etc.
        if pattern in ['embed', 'embedding']:
            # Only match if it's a standalone word or part of known embedding terms
            embedding_terms = ['embed_tokens', 'word_embed', 'token_embed', 'position_embed',
                             'patch_embedding', 'embeddings.', '.embeddings', 'embedding.weight']
            return any(term in search_text for term in embedding_terms)

        # For attention patterns, ensure we catch rotary and other attention-related embeds
        if pattern in ['attention', 'attn', 'rotary', 'rope', 'rotaryembedding', 'q_norm', 'k_norm']:
            # Match attention patterns including rotary/rope which are attention-related
            attention_terms = ['attention', 'attn', 'rotary', 'rope', 'rotaryembedding', 'q_norm', 'k_norm']
            return any(term in search_text for term in attention_terms)

        # Default substring matching for other patterns
        return pattern in search_text

    def _is_transformer_block(self, layer):
        """
        Check if a layer represents a complete transformer block.

        Args:
            layer: ModelStruct object

        Returns:
            bool: True if it's a transformer block
        """
        name_lower = layer.name.lower()
        type_lower = layer.type.lower()

        # Simple string matching for transformer block patterns
        if ('h.' in name_lower and any(char.isdigit() for char in name_lower)) or \
           ('layers.' in name_lower and any(char.isdigit() for char in name_lower)) or \
           ('layer_' in name_lower and any(char.isdigit() for char in name_lower)) or \
           ('block_' in name_lower and any(char.isdigit() for char in name_lower)) or \
           ('transformer_' in name_lower and any(char.isdigit() for char in name_lower)):
            return True

        # Check if it contains typical transformer block keywords
        transformer_keywords = ['transformer', 'encoder', 'decoder', 'block']
        if any(keyword in name_lower or keyword in type_lower for keyword in transformer_keywords):
            return True

        return False

    def get_navigation_data(self, categories):
        """
        Generate navigation data for the frontend.

        Args:
            categories: Dict of categorized layers

        Returns:
            list: Navigation items sorted by priority
        """
        navigation = []

        for cat_key, category in categories.items():
            navigation.append({
                'id': cat_key,
                'name': category['name'],
                'icon': category['icon'],
                'count': category['totalCount'],
                'parameters': category['totalParameters'],
                'color': category['color'],
                'description': category['description']
            })

        # Sort by priority
        navigation.sort(key=lambda x: x.get('priority', 999))

        return navigation

    def detect_architecture_type(self, layers):
        """
        Detect the transformer architecture type based on layer patterns.

        Args:
            layers: List of ModelStruct objects

        Returns:
            str: Detected architecture type
        """
        layer_text = ' '.join([f"{l.type} {l.name}" for l in layers]).lower()

        architecture_patterns = {
            'BERT': ['bert', 'encoder', 'pooler', 'cls'],
            'GPT': ['gpt', 'decoder', 'lm_head', 'wte', 'wpe'],
            'T5': ['t5', 'encoder', 'decoder', 'shared'],
            'BART': ['bart', 'encoder', 'decoder'],
            'RoBERTa': ['roberta', 'encoder'],
            'LLaMA': ['llama', 'rmsnorm', 'rope', 'swiglu'],
            'Falcon': ['falcon', 'alibi', 'rotary'],
            'Mistral': ['mistral', 'sliding_window', 'rmsnorm'],
            'Generic Transformer': ['transformer', 'attention', 'feed_forward']
        }

        for arch, patterns in architecture_patterns.items():
            if all(pattern in layer_text for pattern in patterns):
                return arch

        return 'Unknown Architecture'


# Global categorizer instance
layer_categorizer = LayerCategorizer()


def categorize_model_structure(model_struct):
    """
    Categorize a model structure into meaningful groups.

    Args:
        model_struct: ModelStruct root object

    Returns:
        dict: Categorized model structure with navigation data
    """
    if not model_struct or not model_struct.layers:
        return {
            'categories': {},
            'navigation': [],
            'architecture': 'Unknown',
            'totalLayers': 0,
            'breadcrumbPaths': []
        }

    # Categorize layers
    categories = layer_categorizer.categorize_layers(model_struct.layers)

    # Generate navigation data
    navigation = layer_categorizer.get_navigation_data(categories)

    # Prefer the adapter descriptor; keep heuristic detection for legacy structs.
    architecture = (
        getattr(model_struct, "architecture", None)
        or layer_categorizer.detect_architecture_type(model_struct.layers)
    )

    # Generate unique breadcrumb paths for filtering
    breadcrumb_paths = _extract_breadcrumb_paths(model_struct.layers)

    return {
        'categories': categories,
        'navigation': navigation,
        'architecture': architecture,
        'architectureFamily': getattr(model_struct, 'architectureFamily', None),
        'architectureTopology': getattr(model_struct, 'architectureTopology', None),
        'capabilities': getattr(model_struct, 'capabilities', None) or {},
        'totalLayers': len(model_struct.layers),
        'breadcrumbPaths': breadcrumb_paths
    }


def _extract_breadcrumb_paths(layers):
    """Extract all unique breadcrumb paths for filtering."""
    paths = set()

    for layer in layers:
        if hasattr(layer, 'breadcrumb') and layer.breadcrumb:
            # Add each partial path from the breadcrumb
            for crumb in layer.breadcrumb:
                paths.add(crumb['path'])

    # Convert to sorted list with metadata
    path_list = []
    for path in sorted(paths):
        depth = len(path.split('.'))
        path_list.append({
            'path': path,
            'displayName': _get_path_display_name(path),
            'depth': depth,
            'segments': path.split('.')
        })

    return path_list


def _get_path_display_name(path):
    """Generate display name for a breadcrumb path."""
    segments = path.split('.')
    display_segments = []

    for i, segment in enumerate(segments):
        display_name = _get_breadcrumb_display_name(segment, i, segments)
        display_segments.append(display_name)

    return ' → '.join(display_segments)


def model_to_json(root):
    root.type = LTTSModel.__name__

    # Always add categorization (overwrite if exists to ensure it's fresh)
    root.categorization = categorize_model_structure(root)

    # Debug hierarchy data in JSON conversion
    hierarchy_count = 0
    for layer in root.layers:
        if hasattr(layer, 'hierarchy') and layer.hierarchy and 'self_attn' in layer.name and 'layers.0' in layer.name:
            hierarchy_count += 1

    result_dict = root.to_dict()

    # Check hierarchy in result dict
    for layer_dict in result_dict.get('layers', []):
        if 'hierarchy' in layer_dict and 'self_attn' in layer_dict.get('name', '') and 'layers.0' in layer_dict.get('name', ''):
            hierarchy_count += 1

    return json.dumps(result_dict, indent=2)
