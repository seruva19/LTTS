from typing import Dict, Any, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import re

logger = logging.getLogger(__name__)

# Cap on stored decode steps per layer: captured tensors are kept on CPU, and
# unbounded growth over long generations would leak memory.
MAX_DECODE_STEPS = 256


class AttentionCaptureManager:
    def __init__(self, model: nn.Module):
        self.model = model
        self.captured_attn: Dict[str, Dict[int, torch.Tensor]] = {}
        self.layer_attention_weights: Dict[str, torch.Tensor] = {}
        self.qkv_cache: Dict[str, Dict[str, torch.Tensor]] = {}
        self.qkv_meta: Dict[str, Dict[str, Any]] = {}
        self.capture_meta: Dict[str, Dict[int, Dict[str, Any]]] = {}
        self.decode_step: int = 0
        self.attention_hooks: List[Any] = []
        self.current_attention_mask: Optional[torch.Tensor] = None
        self.enabled: bool = False

        self._setup_attention_capture_hooks()

    def set_enabled(self, enabled: bool) -> None:
        """Enable expensive QKV capture only for runs that request attention."""
        self.enabled = bool(enabled)
        if not self.enabled:
            self.qkv_cache.clear()

    def _setup_attention_capture_hooks(self) -> None:
        """Setup forward hooks to manually compute attention from Q, K, V."""
        logger.info("Setting up manual attention computation hooks")

        def create_qkv_hook(layer_path: str, proj_type: str):
            """Hook to capture Q, K, or V projections."""

            def hook_fn(module, input, output):
                if not self.enabled:
                    return
                try:
                    # Extract identifiers from path
                    match = re.search(r"layers\.(\d+)\.", layer_path)
                    layer_idx = int(match.group(1)) if match else None
                    base_layer_path = layer_path.rsplit(".", 1)[0]

                    cache_key = base_layer_path
                    if layer_idx is not None:
                        cache_key = f"{layer_idx}:{base_layer_path}"

                    if cache_key not in self.qkv_cache:
                        self.qkv_cache[cache_key] = {}
                    self.qkv_meta[cache_key] = {
                        "layer_idx": layer_idx,
                        "layer_path": base_layer_path,
                        "hook_path": layer_path,
                    }

                    # Store the projection output
                    self.qkv_cache[cache_key][proj_type] = output.detach()

                    # If we have all three (Q, K, V), compute attention
                    if all(k in self.qkv_cache[cache_key] for k in ["q", "k", "v"]):
                        self._compute_and_store_attention(cache_key)

                except Exception as e:
                    logger.debug("Error in QKV hook: %s", e)

            return hook_fn

        hooks_registered = 0
        registered_modules: set[str] = set()

        for name, module in self.model.named_modules():
            module_type = type(module).__name__.lower()

            # Register based on module attributes (covers SiglipAttention and similar)
            if (
                hasattr(module, "q_proj")
                and hasattr(module, "k_proj")
                and hasattr(module, "v_proj")
            ):
                base_path = name
                if isinstance(module.q_proj, nn.Module):
                    path = f"{base_path}.q_proj" if base_path else "q_proj"
                    if path not in registered_modules:
                        handle = module.q_proj.register_forward_hook(
                            create_qkv_hook(path, "q")
                        )
                        self.attention_hooks.append(handle)
                        registered_modules.add(path)
                        hooks_registered += 1
                        logger.debug("Registered Q hook for %s", path)
                if isinstance(module.k_proj, nn.Module):
                    path = f"{base_path}.k_proj" if base_path else "k_proj"
                    if path not in registered_modules:
                        handle = module.k_proj.register_forward_hook(
                            create_qkv_hook(path, "k")
                        )
                        self.attention_hooks.append(handle)
                        registered_modules.add(path)
                        hooks_registered += 1
                        logger.debug("Registered K hook for %s", path)
                if isinstance(module.v_proj, nn.Module):
                    path = f"{base_path}.v_proj" if base_path else "v_proj"
                    if path not in registered_modules:
                        handle = module.v_proj.register_forward_hook(
                            create_qkv_hook(path, "v")
                        )
                        self.attention_hooks.append(handle)
                        registered_modules.add(path)
                        hooks_registered += 1
                        logger.debug("Registered V hook for %s", path)
                continue

            # Fallback string matching for other attention implementations
            if (
                "self_attn.q_proj" in name
                or ".attn.q_proj" in name
                or ".attention.q_proj" in name
            ) and name not in registered_modules:
                handle = module.register_forward_hook(create_qkv_hook(name, "q"))
                self.attention_hooks.append(handle)
                registered_modules.add(name)
                hooks_registered += 1
                logger.debug("Registered fallback Q hook for %s", name)
            elif (
                "self_attn.k_proj" in name
                or ".attn.k_proj" in name
                or ".attention.k_proj" in name
            ) and name not in registered_modules:
                handle = module.register_forward_hook(create_qkv_hook(name, "k"))
                self.attention_hooks.append(handle)
                registered_modules.add(name)
                hooks_registered += 1
                logger.debug("Registered fallback K hook for %s", name)
            elif (
                "self_attn.v_proj" in name
                or ".attn.v_proj" in name
                or ".attention.v_proj" in name
            ) and name not in registered_modules:
                handle = module.register_forward_hook(create_qkv_hook(name, "v"))
                self.attention_hooks.append(handle)
                registered_modules.add(name)
                hooks_registered += 1
                logger.debug("Registered fallback V hook for %s", name)

        logger.info("✅ Registered %s QKV projection hooks", hooks_registered)

    def _get_attention_heads_info(
        self,
        layer_idx: Optional[int],
        layer_path: str,
        q_hidden: int,
        k_hidden: int,
    ) -> tuple[int, int]:
        num_heads = None
        num_kv_heads = None
        base_layer_path = layer_path or ""

        config = None
        if hasattr(self.model, "config"):
            config = self.model.config
        elif hasattr(self.model, "model") and hasattr(self.model.model, "config"):
            config = self.model.model.config

        config_head_dim = None
        if config is not None:
            num_heads = getattr(config, "num_attention_heads", None)
            num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
            config_head_dim = getattr(config, "head_dim", None)
            if num_heads is not None or config_head_dim is not None:
                logger.info(
                    "Config fields used for heads: num_attention_heads=%s, "
                    "num_key_value_heads=%s, head_dim=%s",
                    num_heads,
                    num_kv_heads,
                    config_head_dim,
                )

        target_module = None
        if base_layer_path:
            try:
                for name, module in self.model.named_modules():
                    if name == base_layer_path:
                        target_module = module
                        break
            except Exception as e:
                logger.debug("Could not locate module for %s: %s", base_layer_path, e)

        if target_module is None and layer_idx is not None:
            try:
                for name, module in self.model.named_modules():
                    if (
                        f"layers.{layer_idx}.self_attn" in name
                        or f"layers[{layer_idx}].self_attn" in name
                    ):
                        target_module = module
                        break
            except Exception as e:
                logger.debug("Could not extract heads from module: %s", e)

        if target_module is None and base_layer_path:
            try:
                for name, module in self.model.named_modules():
                    if name.endswith(base_layer_path):
                        target_module = module
                        break
            except Exception:
                pass

        if target_module is not None:
            if hasattr(target_module, "num_heads"):
                num_heads = getattr(target_module, "num_heads")
                if hasattr(target_module, "num_key_value_heads"):
                    num_kv_heads = getattr(target_module, "num_key_value_heads")
                elif hasattr(target_module, "num_key_value_groups"):
                    groups = getattr(target_module, "num_key_value_groups", 1)
                    num_kv_heads = max(1, num_heads // groups)
                else:
                    num_kv_heads = num_heads
                logger.info("Extracted num_heads=%s from module attribute", num_heads)
            elif hasattr(target_module, "num_attention_heads"):
                num_heads = getattr(target_module, "num_attention_heads")
                if hasattr(target_module, "num_key_value_heads"):
                    num_kv_heads = getattr(target_module, "num_key_value_heads")
                elif hasattr(target_module, "num_key_value_groups"):
                    groups = getattr(target_module, "num_key_value_groups", 1)
                    num_kv_heads = max(1, num_heads // groups)
                else:
                    num_kv_heads = num_heads
                logger.info("Extracted num_heads=%s from module attribute", num_heads)

        if num_heads is None and config_head_dim:
            # Config lacks num_attention_heads but provides head_dim
            # (e.g. some architectures): derive heads from hidden sizes.
            if q_hidden % config_head_dim == 0:
                num_heads = q_hidden // config_head_dim
                if k_hidden % config_head_dim == 0:
                    num_kv_heads = k_hidden // config_head_dim
                logger.info(
                    "Derived num_heads=%s from config head_dim=%s",
                    num_heads,
                    config_head_dim,
                )

        if num_heads is None:
            logger.warning(
                "No reliable head counts in model config (num_attention_heads/"
                "num_key_value_heads/head_dim) for layer %s; falling back to "
                "head_dim guess-and-check heuristic",
                layer_idx,
            )
            common_head_dims = [64, 128, 96, 80]
            for head_dim in common_head_dims:
                if q_hidden % head_dim == 0:
                    num_heads = q_hidden // head_dim
                    num_kv_heads = k_hidden // head_dim
                    break

        if num_heads is None:
            raise ValueError(
                f"Could not determine attention heads for layer {layer_idx}. "
                f"Q hidden={q_hidden}, K hidden={k_hidden}"
            )

        if num_kv_heads is None:
            num_kv_heads = num_heads

        if num_heads and num_kv_heads:
            q_divisible = q_hidden % num_heads == 0
            if q_divisible:
                inferred_q_head_dim = q_hidden // num_heads
                if inferred_q_head_dim > 0 and k_hidden % inferred_q_head_dim == 0:
                    inferred_kv_heads = k_hidden // inferred_q_head_dim
                    if inferred_kv_heads != 0 and inferred_kv_heads != num_kv_heads:
                        logger.debug(
                            "Adjusting num_kv_heads from %s to %s based on hidden sizes",
                            num_kv_heads,
                            inferred_kv_heads,
                        )
                        num_kv_heads = inferred_kv_heads

        return num_heads, num_kv_heads

    def _compute_and_store_attention(self, cache_key: str) -> None:
        try:
            entry = self.qkv_cache[cache_key]
            metadata = self.qkv_meta.get(cache_key, {})
            layer_idx = metadata.get("layer_idx")
            layer_path = metadata.get("layer_path", "")

            q = entry["q"]
            k = entry["k"]
            v = entry["v"]

            # Handle different tensor shapes
            if q.ndim == 2:
                q = q.unsqueeze(1)
            if k.ndim == 2:
                k = k.unsqueeze(1)
            if v.ndim == 2:
                v = v.unsqueeze(1)

            batch_size = q.shape[0]
            q_seq_len = q.shape[1]
            q_hidden_size = q.shape[2]

            k_seq_len = k.shape[1]
            k_hidden_size = k.shape[2]

            num_heads, num_kv_heads = self._get_attention_heads_info(
                layer_idx, layer_path, q_hidden_size, k_hidden_size
            )

            q_head_dim = q_hidden_size // num_heads

            if k_hidden_size % max(1, num_kv_heads) != 0:
                logger.debug(
                    "Adjusting num_kv_heads=%s to fit k_hidden_size=%s (layer=%s)",
                    num_kv_heads,
                    k_hidden_size,
                    layer_path,
                )
                num_kv_heads = max(1, k_hidden_size // max(1, q_head_dim))

            kv_head_dim = k_hidden_size // max(1, num_kv_heads)

            logger.debug(
                "Layer %s: num_heads=%s, num_kv_heads=%s, q_hidden=%s, k_hidden=%s, q_head_dim=%s, kv_head_dim=%s",
                layer_path or layer_idx,
                num_heads,
                num_kv_heads,
                q_hidden_size,
                k_hidden_size,
                q_head_dim,
                kv_head_dim,
            )

            # Verify head dimensions match
            if q_head_dim != kv_head_dim:
                if q_head_dim != 0 and k_hidden_size % q_head_dim == 0:
                    inferred_kv_heads = k_hidden_size // q_head_dim
                    if inferred_kv_heads != max(1, num_kv_heads):
                        logger.debug(
                            "Adjusting num_kv_heads from %s to %s to align head dimensions (layer=%s)",
                            num_kv_heads,
                            inferred_kv_heads,
                            layer_path,
                        )
                        num_kv_heads = inferred_kv_heads
                    kv_head_dim = q_head_dim
                else:
                    logger.warning(
                        "Head dimension mismatch: q_head_dim=%s, kv_head_dim=%s. Proceeding with kv_head_dim.",
                        q_head_dim,
                        kv_head_dim,
                    )

            q = q.view(batch_size, q_seq_len, num_heads, q_head_dim).transpose(1, 2)
            k = k.view(batch_size, k_seq_len, num_kv_heads, kv_head_dim).transpose(1, 2)
            v = v.view(batch_size, k_seq_len, num_kv_heads, kv_head_dim).transpose(1, 2)

            # Handle Grouped Query Attention (GQA)
            # After transpose, shape is [batch, num_heads, seq_len, head_dim]
            if num_kv_heads < num_heads:
                repeat_factor = num_heads // num_kv_heads
                # Repeat along the head dimension (dim=1)
                k = k.repeat_interleave(repeat_factor, dim=1)
                v = v.repeat_interleave(repeat_factor, dim=1)

            attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (q_head_dim**0.5)
            attn_scores = self._apply_attention_mask(attn_scores, self.current_attention_mask)
            # Decoder self-attention is causal: a query at position i must not
            # attend to keys at j > i. The padding mask above does not enforce
            # this, so the QKV-recompute fallback would otherwise produce
            # bidirectional (wrong) attention for a causal LM. Apply only to the
            # square self-attention case (prefill); decode steps have q_seq==1.
            q_len, k_len = attn_scores.shape[-2], attn_scores.shape[-1]
            if q_len == k_len and q_len > 1:
                causal = torch.tril(
                    torch.ones(q_len, k_len, device=attn_scores.device, dtype=torch.bool)
                )
                attn_scores = attn_scores.masked_fill(
                    ~causal, torch.finfo(attn_scores.dtype).min
                )
            attn_probs = F.softmax(attn_scores, dim=-1)
            # Fully masked rows produce NaN after softmax; replace with 0.
            attn_probs = torch.nan_to_num(attn_probs, nan=0.0)

            # Prefer the real module path; the constant key is only a fallback.
            base_layer_key = layer_path or metadata.get("hook_path", "")
            canonical_key = base_layer_key
            if layer_idx is not None:
                canonical_key = f"model.layers.{layer_idx}.self_attn"
                for name, _ in self.model.named_modules():
                    if name == base_layer_key:
                        canonical_key = base_layer_key
                        break

            if canonical_key:
                if canonical_key not in self.captured_attn:
                    self.captured_attn[canonical_key] = {}
                self._prune_decode_steps(canonical_key)
                self.captured_attn[canonical_key][self.decode_step] = (
                    attn_probs.detach().cpu()
                )
                self.layer_attention_weights[canonical_key] = (
                    attn_probs.detach().cpu()
                )
                if canonical_key not in self.capture_meta:
                    self.capture_meta[canonical_key] = {}
                self.capture_meta[canonical_key][self.decode_step] = {
                    "source": "qkv_recompute",
                    "approximate": True,
                }

            logger.debug(
                "Stored attention for layer='%s' (cache=%s) step=%s shape=%s",
                layer_path or layer_idx,
                cache_key,
                self.decode_step,
                tuple(attn_probs.shape),
            )

            del self.qkv_cache[cache_key]
            if cache_key in self.qkv_meta:
                del self.qkv_meta[cache_key]

        except Exception as e:
            logger.error(
                "Error computing attention for layer %s: %s",
                self.qkv_meta.get(cache_key, {}).get("layer_path", cache_key),
                e,
            )
            logger.error(
                "Q shape: %s, K shape: %s, V shape: %s",
                self.qkv_cache[cache_key].get("q").shape
                if "q" in self.qkv_cache[cache_key]
                else "missing",
                self.qkv_cache[cache_key].get("k").shape
                if "k" in self.qkv_cache[cache_key]
                else "missing",
                self.qkv_cache[cache_key].get("v").shape
                if "v" in self.qkv_cache[cache_key]
                else "missing",
            )
            import traceback

            logger.error(traceback.format_exc())

    def extract_from_outputs(self, outputs: Any) -> None:
        try:
            from miners.utils import extract_attention_weights

            model_attention_weights = extract_attention_weights(outputs)

            if model_attention_weights is not None:
                if hasattr(outputs, "attentions") and outputs.attentions is not None:
                    self.layer_attention_weights = {}
                    for i, layer_attention in enumerate(outputs.attentions):
                        if isinstance(layer_attention, torch.Tensor):
                            layer_key = self._canonical_layer_key(i)
                            self.layer_attention_weights[layer_key] = (
                                layer_attention.detach().cpu()
                            )
                            if layer_key not in self.captured_attn:
                                self.captured_attn[layer_key] = {}
                            self._prune_decode_steps(layer_key)
                            self.captured_attn[layer_key][self.decode_step] = (
                                layer_attention.detach().cpu()
                            )
                            if layer_key not in self.capture_meta:
                                self.capture_meta[layer_key] = {}
                            self.capture_meta[layer_key][self.decode_step] = {
                                "source": "outputs_attentions",
                                "approximate": False,
                            }
            else:
                logger.info(
                    "No attention weights found in model outputs (normal with hook-based capture)"
                )

        except Exception as e:
            logger.warning(
                "Error extracting attention weights from model outputs: %s", e
            )

    def _prune_decode_steps(self, layer_key: str) -> None:
        """Drop oldest decode steps beyond MAX_DECODE_STEPS for a layer."""
        steps = self.captured_attn.get(layer_key)
        if not steps:
            return
        excess = len(steps) - MAX_DECODE_STEPS
        if excess <= 0:
            return
        for step in sorted(steps)[:excess]:
            del steps[step]
            meta = self.capture_meta.get(layer_key)
            if meta and step in meta:
                del meta[step]

    def _canonical_layer_key(self, layer_index: int) -> str:
        """Map a flat attentions tuple index back to a real module path."""
        for name, module in self.model.named_modules():
            match = re.search(r"layers\.(\d+)\.", name)
            if match and int(match.group(1)) == layer_index:
                parent = name.rsplit(".", 1)[0]
                if any(
                    part in parent
                    for part in ("self_attn", "attn", "attention")
                ):
                    return parent
        return f"model.layers.{layer_index}.self_attn"

    def increment_step(self) -> None:
        self.decode_step += 1

    def set_attention_mask(self, attention_mask: Optional[torch.Tensor]) -> None:
        self.current_attention_mask = attention_mask

    def reset_run(self) -> None:
        self.captured_attn.clear()
        self.layer_attention_weights.clear()
        self.qkv_cache.clear()
        self.qkv_meta.clear()
        self.capture_meta.clear()
        self.current_attention_mask = None
        self.decode_step = 0

    def _apply_attention_mask(
        self,
        attn_scores: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if attention_mask is None:
            return attn_scores

        try:
            mask = attention_mask
            if not isinstance(mask, torch.Tensor):
                return attn_scores
            mask = mask.to(device=attn_scores.device)
            if mask.dim() == 2:
                # [batch, seq] -> [batch, 1, 1, seq]
                mask = mask[:, None, None, :]
            elif mask.dim() == 3:
                # [batch, seq, seq] or [batch, 1, seq]
                mask = mask[:, None, :, :]
            elif mask.dim() == 4:
                pass
            else:
                return attn_scores

            if mask.dtype != torch.bool:
                mask = mask > 0

            neg_inf = torch.finfo(attn_scores.dtype).min
            return attn_scores.masked_fill(~mask, neg_inf)
        except Exception as e:
            logger.debug("Error applying attention mask: %s", e)
            return attn_scores

    def cleanup(self) -> None:
        for handle in self.attention_hooks:
            try:
                handle.remove()
            except Exception as e:
                logger.debug("Error removing attention hook: %s", e)
        self.attention_hooks.clear()
