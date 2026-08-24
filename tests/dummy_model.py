from typing import Dict, Any, Optional, Tuple, Callable
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast
from scripts.core.ltts_model import LTTSModel
from scripts.models.model_struct import ModelStruct, categorize_model_structure


class DummyTokenizer:
    """A dummy tokenizer for testing purposes"""

    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        self.pad_token = "<pad>"
        self.eos_token = "<eos>"
        self.bos_token = "<bos>"
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = 2

    def encode(self, text: str, **kwargs) -> list[int]:
        """Simple encoding that converts characters to ASCII values"""
        return [ord(c) % self.vocab_size for c in text]

    def decode(self, token_ids: list[int], **kwargs) -> str:
        """Simple decoding that converts token IDs back to characters"""
        return "".join([chr(tid) if tid < 128 else "?" for tid in token_ids])

    def __call__(self, text: str, **kwargs) -> Dict[str, Any]:
        """Tokenize text and return format compatible with transformers"""
        token_ids = self.encode(text, **kwargs)
        return {
            "input_ids": torch.tensor([token_ids]),
            "attention_mask": torch.ones(1, len(token_ids)),
        }


class DummyModel(nn.Module):
    """A dummy transformer model for testing purposes"""

    def __init__(
        self,
        vocab_size: int = 32000,
        hidden_size: int = 768,
        num_layers: int = 32,
        use_moe: bool = False,
        use_rms_norm: bool = False
    ):
        super().__init__()

        # Create a simple config object
        class Config:
            def __init__(self):
                self.vocab_size = vocab_size
                self.hidden_size = hidden_size
                self.num_hidden_layers = num_layers
                self.num_attention_heads = 12
                self.intermediate_size = hidden_size * 4
                self.max_position_embeddings = 2048
                self.pad_token_id = 0
                self.eos_token_id = 1
                self.bos_token_id = 2

        self.config = Config()

        # Embeddings
        self.embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_embeddings = nn.Embedding(2048, hidden_size)

        # Transformer layers
        if use_moe:
            self.layers = nn.ModuleList(
                [DummyTransformerLayerWithMoE(hidden_size, use_rms_norm) for _ in range(num_layers)]
            )
        else:
            self.layers = nn.ModuleList(
                [DummyTransformerLayer(hidden_size, use_rms_norm) for _ in range(num_layers)]
            )

        # Output head
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

        # Layer norm
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, input_ids: torch.Tensor, **kwargs) -> Dict[str, Any]:
        batch_size, seq_len = input_ids.shape
        device = input_ids.device

        # Create position IDs
        position_ids = (
            torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        )

        # Get embeddings
        token_embeddings = self.embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        # Combine embeddings
        hidden_states = token_embeddings + position_embeddings

        # Pass through transformer layers
        all_hidden_states = (
            [hidden_states] if kwargs.get("output_hidden_states", False) else None
        )

        for layer in self.layers:
            hidden_states = layer(hidden_states)
            if all_hidden_states is not None:
                all_hidden_states.append(hidden_states)

        # Final layer norm
        hidden_states = self.norm(hidden_states)

        # Output head
        logits = self.lm_head(hidden_states)

        return {"logits": logits, "hidden_states": all_hidden_states}

    def generate(
        self, input_ids: torch.Tensor, max_length: int = 100, **kwargs
    ) -> torch.Tensor:
        """Simple greedy generation"""
        current_ids = input_ids.clone()

        for _ in range(max_length - input_ids.shape[1]):
            outputs = self.forward(current_ids)
            next_token = outputs["logits"][:, -1, :].argmax(dim=-1, keepdim=True)
            current_ids = torch.cat([current_ids, next_token], dim=1)

            # Stop if we hit EOS token
            if (next_token == self.config.eos_token_id).any():
                break

        return current_ids


class DummyTransformerLayer(nn.Module):
    """A dummy transformer layer for testing"""

    def __init__(self, hidden_size: int, use_rms_norm: bool = False):
        super().__init__()
        self.attention = DummyAttention(hidden_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.ReLU(),
            nn.Linear(hidden_size * 4, hidden_size),
        )

        if use_rms_norm:
            self.norm1 = RMSNorm(hidden_size)
            self.norm2 = RMSNorm(hidden_size)
        else:
            self.norm1 = nn.LayerNorm(hidden_size)
            self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Self-attention
        attention_output = self.attention(hidden_states)
        hidden_states = self.norm1(hidden_states + attention_output)

        # Feed-forward
        ff_output = self.feed_forward(hidden_states)
        hidden_states = self.norm2(hidden_states + ff_output)

        return hidden_states


class DummyAttention(nn.Module):
    """A dummy attention mechanism for testing with proper scaling and masking"""

    def __init__(self, hidden_size: int, num_heads: int = 12, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.hidden_size = hidden_size
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)

        # Register causal mask buffer
        self.register_buffer("causal_mask", None, persistent=False)

    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # Project to Q, K, V
        q = (
            self.q_proj(hidden_states)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(hidden_states)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(hidden_states)
            .view(batch_size, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale

        # Apply causal mask for autoregressive generation
        if self.causal_mask is None or self.causal_mask.size(-1) < seq_len:
            self.causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=hidden_states.device, dtype=torch.bool))

        causal_mask = self.causal_mask[:seq_len, :seq_len]
        scores = scores.masked_fill(~causal_mask, float('-inf'))

        # Apply attention mask if provided
        if attention_mask is not None:
            # Expand attention mask to match scores shape
            expanded_mask = attention_mask[:, None, None, :].expand(batch_size, self.num_heads, seq_len, seq_len)
            scores = scores.masked_fill(~expanded_mask, float('-inf'))

        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # Apply attention
        context = torch.matmul(attention_weights, v)
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.hidden_size)
        )

        # Output projection
        return self.o_proj(context)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (used in LLaMA, Mistral)"""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class DummyMoELayer(nn.Module):
    """Dummy Mixture of Experts layer for testing MoE architectures"""

    def __init__(self, hidden_size: int, num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.hidden_size = hidden_size

        # Router/gating network
        self.router = nn.Linear(hidden_size, num_experts, bias=False)

        # Expert networks (simple FFNs)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.ReLU(),
                nn.Linear(hidden_size * 4, hidden_size)
            ) for _ in range(num_experts)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, hidden_size = x.shape
        x_flat = x.view(-1, hidden_size)  # [batch_size * seq_len, hidden_size]

        # Router logits
        router_logits = self.router(x_flat)  # [batch_size * seq_len, num_experts]
        routing_weights = F.softmax(router_logits, dim=-1)

        # Select top-k experts
        top_k_weights, top_k_indices = torch.topk(routing_weights, self.top_k, dim=-1)
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)  # Normalize

        # Initialize output
        output = torch.zeros_like(x_flat)

        # Process through selected experts
        for i in range(self.top_k):
            expert_indices = top_k_indices[:, i]
            expert_weights = top_k_weights[:, i:i+1]

            for expert_id in range(self.num_experts):
                expert_mask = (expert_indices == expert_id)
                if expert_mask.any():
                    expert_input = x_flat[expert_mask]
                    expert_output = self.experts[expert_id](expert_input)
                    output[expert_mask] += expert_weights[expert_mask] * expert_output

        return output.view(batch_size, seq_len, hidden_size)


class DummyTransformerLayerWithMoE(nn.Module):
    """Transformer layer with MoE instead of regular FFN"""

    def __init__(self, hidden_size: int, use_rms_norm: bool = False):
        super().__init__()
        self.attention = DummyAttention(hidden_size)
        self.moe = DummyMoELayer(hidden_size)

        if use_rms_norm:
            self.norm1 = RMSNorm(hidden_size)
            self.norm2 = RMSNorm(hidden_size)
        else:
            self.norm1 = nn.LayerNorm(hidden_size)
            self.norm2 = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Self-attention
        attention_output = self.attention(hidden_states)
        hidden_states = self.norm1(hidden_states + attention_output)

        # MoE
        moe_output = self.moe(hidden_states)
        hidden_states = self.norm2(hidden_states + moe_output)

        return hidden_states


def create_dummy_model_struct() -> ModelStruct:
    """Create a dummy model structure for testing with realistic transformer components"""
    root = ModelStruct(name="dummy", type="DummyModel")
    root.architecture = "Debug decoder transformer"
    root.architectureFamily = "debug_transformer"
    root.architectureTopology = "decoder_only"
    root.modelType = "causal_lm"
    root.capabilities = {
        "hidden_states": True,
        "gradients": True,
        "tokenizer_decoding": True,
        "unembedding": True,
        "generation": True,
        "eager_attention": False,
    }

    # Add embeddings
    embeddings = ModelStruct(name="embeddings", type="Embedding")
    embeddings.parameters = 24576000  # vocab_size * hidden_size
    embeddings.inputShape = [1, "seq_len"]
    embeddings.outputShape = [1, "seq_len", 768]
    root.add_child(embeddings)

    pos_embeddings = ModelStruct(name="position_embeddings", type="Embedding")
    pos_embeddings.parameters = 1572864  # 2048 * 768
    pos_embeddings.inputShape = [1, "seq_len"]
    pos_embeddings.outputShape = [1, "seq_len", 768]
    root.add_child(pos_embeddings)

    # Add transformer layers with realistic sub-components
    for i in range(4):  # Use 4 layers for better UI testing
        # Create transformer layer container
        transformer_layer = ModelStruct(name=f"layers.{i}", type="DummyTransformerLayer")
        transformer_layer.parameters = 2359296  # Approximate params per layer
        transformer_layer.inputShape = [1, "seq_len", 768]
        transformer_layer.outputShape = [1, "seq_len", 768]
        transformer_layer.hasResidual = True
        transformer_layer.blockType = "decoder"
        transformer_layer.moduleRole = "block"
        transformer_layer.architectureStage = "decoder"
        transformer_layer.layerIndex = i

        # Add attention sub-layer
        attention = ModelStruct(name=f"layers.{i}.attention", type="DummyAttention")
        attention.parameters = 1179648  # 4 * 768 * 768 (Q, K, V, O projections)
        attention.numHeads = 12
        attention.attentionType = "self-attention"
        attention.moduleRole = "attention"
        attention.architectureStage = "decoder"
        attention.layerIndex = i
        attention.inputShape = [1, "seq_len", 768]
        attention.outputShape = [1, "seq_len", 768]
        transformer_layer.add_child(attention)

        # Add feed-forward sub-layer
        ffn = ModelStruct(name=f"layers.{i}.feed_forward", type="Sequential")
        ffn.parameters = 1179648  # 768 * 3072 + 3072 * 768
        ffn.inputShape = [1, "seq_len", 768]
        ffn.outputShape = [1, "seq_len", 768]
        ffn.moduleRole = "mlp"
        ffn.architectureStage = "decoder"
        ffn.layerIndex = i
        transformer_layer.add_child(ffn)

        # Add layer norms
        norm1 = ModelStruct(name=f"layers.{i}.norm1", type="LayerNorm")
        norm1.parameters = 1536  # 2 * 768 (weight + bias)
        norm1.inputShape = [1, "seq_len", 768]
        norm1.outputShape = [1, "seq_len", 768]
        transformer_layer.add_child(norm1)

        norm2 = ModelStruct(name=f"layers.{i}.norm2", type="LayerNorm")
        norm2.parameters = 1536
        norm2.inputShape = [1, "seq_len", 768]
        norm2.outputShape = [1, "seq_len", 768]
        transformer_layer.add_child(norm2)

        root.add_child(transformer_layer)

    # Add final layer norm
    final_norm = ModelStruct(name="norm", type="LayerNorm")
    final_norm.parameters = 1536
    final_norm.inputShape = [1, "seq_len", 768]
    final_norm.outputShape = [1, "seq_len", 768]
    root.add_child(final_norm)

    # Add language modeling head
    lm_head = ModelStruct(name="lm_head", type="Linear")
    lm_head.parameters = 24576000  # 768 * vocab_size
    lm_head.inputShape = [1, "seq_len", 768]
    lm_head.outputShape = [1, "seq_len", 32000]
    root.add_child(lm_head)

    # Add categorization information
    root.categorization = categorize_model_structure(root)

    return root


def create_dummy_model_and_tokenizer(
    vocab_size: int = 32000,
    hidden_size: int = 768,
    num_layers: int = 32,
    use_moe: bool = False,
    use_rms_norm: bool = False
) -> Tuple[DummyModel, DummyTokenizer]:
    """Create a dummy model and tokenizer for testing"""
    model = DummyModel(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        use_moe=use_moe,
        use_rms_norm=use_rms_norm
    )
    tokenizer = DummyTokenizer(vocab_size=vocab_size)
    return model, tokenizer


def create_dummy_ltts_model(
    callback: Callable[[Dict[str, Any]], None],
    vocab_size: int = 32000,
    hidden_size: int = 768,
    num_layers: int = 32,
    use_moe: bool = False,
    use_rms_norm: bool = False
) -> Tuple[LTTSModel, ModelStruct]:
    """Create a dummy LTTS model with the specified callback"""
    model, tokenizer = create_dummy_model_and_tokenizer(
        vocab_size, hidden_size, num_layers, use_moe, use_rms_norm
    )

    # Create LTTS wrapper with new unified architecture
    ltts_model = LTTSModel(model, message_bus=callback)
    # Keep the tokenizer on the wrapper just like the Hugging Face loading
    # path does.  Without this, the inference service treats ``debug/dummy``
    # as a Hub model id and asks AutoTokenizer to download it.
    ltts_model._tokenizer = tokenizer

    # Create model structure
    struct = create_dummy_model_struct()

    # Add categorization information
    struct.categorization = categorize_model_structure(struct)

    return ltts_model, struct
