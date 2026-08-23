"""
LTTSModel Implementation

Event-dispatch wrapper for transformer models. Attention capture, module
registration, and event delivery are handled by dedicated components.
"""

from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn
import logging
from .ltts_layer import LTTSLayer
from .ltts_event import LTTSContext
from .miners_registry import get_global_registry
from .attention_capture import AttentionCaptureManager
from .module_registry import ModuleRegistry
from .event_dispatcher import EventDispatcher
from scripts.models.architecture_adapter import (
    describe_standalone_module,
    get_architecture_adapter,
)

# Provide Int4WeightOnlyConfig when absent in supported torchao versions.
try:
    import torchao.quantification

    if not hasattr(torchao.quantification, "Int4WeightOnlyConfig"):

        class Int4WeightOnlyConfig:
            def __init__(self, *args, **kwargs):
                pass

        torchao.quantification.Int4WeightOnlyConfig = Int4WeightOnlyConfig
except ImportError:
    pass

logger = logging.getLogger(__name__)


class LTTSModel(nn.Module):
    """
    Pure event dispatcher wrapper for transformer models.

    Dispatches lifecycle events to attached modules.
    """

    def __init__(self, model: nn.Module, message_bus=None):
        super().__init__()
        self.original_model = model
        self.message_bus = message_bus
        self.architecture_adapter = get_architecture_adapter(self.original_model)
        self.architecture = self.architecture_adapter.descriptor.to_dict()
        self.module_descriptors: Dict[str, Dict[str, Any]] = {}
        self._suspend_analysis_events = False
        # Optional no-arg callable set by the inference service; when it
        # returns True the current run has been cancelled by the client.
        self._cancellation_check = None
        self._generation_active = False
        self._generation_step = None
        self._generation_hook = self.original_model.register_forward_pre_hook(
            self._track_generation_forward
        )

        # Provide a default message bus if none is supplied or invalid
        if self.message_bus is None or not hasattr(self.message_bus, "publish"):
            try:
                from scripts.messages.message_bus import MessageBus

                self.message_bus = MessageBus()
                logger.info("LTTS_MODEL: Using an isolated default message bus")
            except Exception as e:
                logger.warning("LTTS_MODEL: Failed to load default message bus: %s", e)

        # Initialize layer wrappers
        self.layer_wrappers: Dict[str, LTTSLayer] = {}
        self._setup_layer_wrappers()

        # Initialize helper components
        self.attention_manager = AttentionCaptureManager(self.original_model)
        self.module_registry = ModuleRegistry(self.layer_wrappers)
        self.event_dispatcher = EventDispatcher(self.message_bus, get_global_registry())

    def _setup_layer_wrappers(self):
        """Create LTTSLayer wrappers for all transformer layers."""
        for name, module in self.original_model.named_modules():
            descriptor = self.architecture_adapter.describe_module(name, module)
            if self.architecture_adapter.should_wrap_module(name, module):
                layer_wrapper = LTTSLayer(
                    module,
                    name,
                    event_dispatcher=self._dispatch_event,
                    module_descriptor=descriptor.to_dict(),
                )
                self.layer_wrappers[name] = layer_wrapper
                self.module_descriptors[name] = descriptor.to_dict()

                # Layer binding works via forward hooks registered in LTTSLayer.__init__.
                # The hooks are attached to the original module (which stays in the tree),
                # so no module replacement is needed here.

    def _is_transformer_layer(self, module: nn.Module) -> bool:
        """
        Check if module is an actual transformer sub-layer component.

        Only wraps meaningful internal layers (attention, MLP, norm, embedding,
        decoder layers), not top-level model containers or generic wrappers.

        Args:
            module: Module to check

        Returns:
            True if module is a transformer layer component
        """
        descriptor = describe_standalone_module(module)
        return descriptor.attachable

    def forward(self, *args, **kwargs):
        """Forward pass with conditional event dispatching."""
        # Increment decode step for attention tracking
        self.attention_manager.increment_step()
        logger.debug(
            "LTTSModel forward pass started, decode_step: %s",
            self.attention_manager.decode_step,
        )

        # Dispatch model_before event
        if self.module_registry.has_modules_attached("model_before"):
            self._dispatch_model_before_event(args, kwargs)

        # Only request attention outputs when a module actually needs them.
        # Forcing output_attentions=True on every call materializes
        # [batch, heads, seq, seq] tensors for every layer, which can OOM on
        # large models / long prompts even for plain generation. Set it
        # explicitly (True/False) each call so a stale config flag from a prior
        # run can't silently re-enable it.
        attn_needed = bool(getattr(self, "_force_extract_attention", False))
        self.attention_manager.set_enabled(attn_needed)
        kwargs["output_attentions"] = attn_needed

        try:
            attention_mask = kwargs.get("attention_mask")
            self.attention_manager.set_attention_mask(attention_mask)
        except Exception as e:
            logger.debug("Failed to set attention mask: %s", e)

        # Execute original model
        outputs = self.original_model(*args, **kwargs)
        logger.debug("LTTSModel forward completed, outputs type: %s", type(outputs))

        # Prefer HF's native causal attentions: extract them whenever they were
        # requested, not only for model_after modules. Attention visualizers
        # attach to layer_after, and extract runs after the QKV hooks, so the
        # native weights take precedence over the QKV recompute fallback.
        if attn_needed:
            try:
                self.attention_manager.extract_from_outputs(outputs)
            except Exception as e:
                logger.debug("extract_from_outputs failed: %s", e)

        # Dispatch model_after event
        if self.module_registry.has_modules_attached("model_after"):
            self._dispatch_model_after_event(args, kwargs, outputs)

        logger.debug("LTTSModel forward pass completed")
        return outputs

    def generate(self, *args, **kwargs):
        """Generate method that delegates to the original model's generate method."""
        try:
            if not hasattr(self.original_model, "generate"):
                logger.warning(
                    "Original model %s does not have a generate method",
                    type(self.original_model).__name__,
                )
                raise AttributeError(
                    f"'{type(self.original_model).__name__}' object has no attribute 'generate'"
                )

            # Mirror forward(): decode-step tracking, attention setup and
            # model_before/model_after dispatch around the original call.
            self.attention_manager.increment_step()

            attn_needed = bool(getattr(self, "_force_extract_attention", False))
            self.attention_manager.set_enabled(attn_needed)
            kwargs.setdefault("output_attentions", attn_needed)

            attention_mask = kwargs.get("attention_mask")
            if attention_mask is None:
                # generate() builds masks from inputs; fall back to inputs kwarg.
                inputs = kwargs.get("inputs")
                if isinstance(inputs, dict):
                    attention_mask = inputs.get("attention_mask")
            if attention_mask is not None:
                try:
                    self.attention_manager.set_attention_mask(attention_mask)
                except Exception as e:
                    logger.debug("Failed to set attention mask: %s", e)

            self._generation_active = True
            self._generation_step = -1
            try:
                if self.module_registry.has_modules_attached("model_before"):
                    self._dispatch_model_before_event(args, kwargs)

                generate_method = getattr(self.original_model, "generate")
                outputs = generate_method(*args, **kwargs)

                if attn_needed:
                    try:
                        self.attention_manager.extract_from_outputs(outputs)
                    except Exception as e:
                        logger.debug("extract_from_outputs failed: %s", e)

                if self.module_registry.has_modules_attached("model_after"):
                    self._dispatch_model_after_event(args, kwargs, outputs)

                return outputs
            finally:
                self._generation_active = False
        except Exception as e:
            logger.error("Error in LTTSModel generate: %s", e)
            raise

    def _dispatch_model_before_event(self, inputs, kwargs):
        """Dispatch model_before event with complete context."""
        context = LTTSContext(
            inputs=inputs,
            module_path="model",
            module_class=type(self.original_model).__name__,
            layer_type="model",
            module_role="model",
            architecture_stage="model",
            architecture=self.architecture,
            message_bus=self.message_bus,
        )

        self._dispatch_event("model_before", context)

    def _dispatch_model_after_event(self, inputs, kwargs, outputs):
        """Dispatch model_after event with complete context."""
        context = LTTSContext(
            inputs=inputs,
            outputs=outputs,
            module_path="model",
            module_class=type(self.original_model).__name__,
            layer_type="model",
            module_role="model",
            architecture_stage="model",
            architecture=self.architecture,
            message_bus=self.message_bus,
        )

        self._dispatch_event("model_after", context)

    def _dispatch_event(self, event_type: str, context: LTTSContext):
        """
        Dispatch event to attached modules.

        Args:
            event_type: Type of event to dispatch
            context: Event context
        """
        if self._suspend_analysis_events:
            return

        if self._cancellation_check is not None and self._cancellation_check():
            from scripts.services.inference.cancellation import (
                InferenceCancelledError,
            )

            raise InferenceCancelledError(
                "Inference cancelled by client between layers " f"(event={event_type})"
            )

        # Add reference to LTTSModel for accessing model-level attention weights
        context.ltts_model = self
        context.generation_active = self._generation_active
        context.generation_step = (
            self._generation_step if self._generation_active else None
        )
        if not context.architecture:
            context.architecture = dict(self.architecture)

        # Use event dispatcher to handle module calls
        self.event_dispatcher.dispatch(
            event_type, context, self.module_registry.attached_modules
        )

    def attach_module(
        self,
        event_type: str,
        module: Any,
        module_config: Optional[Dict[str, Any]] = None,
        target_layer: Optional[str] = None,
        requested_layer: Optional[str] = None,
        schedule: Optional[Dict[str, Any]] = None,
    ):
        """
        Attach module to specific event type.

        Args:
            event_type: Event type to attach to
            module: Module to attach
            module_config: Optional module configuration
            target_layer: Resolved target layer path
            requested_layer: Originally requested layer path
        """
        self.module_registry.attach_module(
            event_type,
            module,
            module_config,
            target_layer,
            requested_layer,
            schedule,
        )

    def attach_model_level_module(self, module: Any):
        """
        Attach module to model-level events (model_after).

        Args:
            module: Module to attach
        """
        self.attach_module("model_after", module)
        logger.info("Attached model-level module %s to model_after event", module)

    def attach_module_to_layer(
        self,
        layer_name: str,
        module: Any,
        module_config: Optional[Dict[str, Any]] = None,
        schedule: Optional[Dict[str, Any]] = None,
    ):
        """
        Attach module to specific layer's events with automatic layer path correction.

        Args:
            layer_name: Layer name to attach to
            module: Module to attach
            module_config: Optional module configuration

        Returns:
            Resolved layer path or None if attachment failed
        """
        return self.module_registry.attach_to_layer(
            layer_name, module, module_config, schedule
        )

    def _track_generation_forward(self, _module, _inputs) -> None:
        if self._generation_active:
            self._generation_step = int(self._generation_step or 0) + 1

    def detach_module(self, event_type: str, module: Any):
        """
        Detach module from event type.

        Args:
            event_type: Event type to detach from
            module: Module to detach
        """
        self.module_registry.detach_module(event_type, module)

    def has_modules_attached(self, event_type: str) -> bool:
        """
        Check if any modules are attached to event type.

        Args:
            event_type: Event type to check

        Returns:
            True if modules are attached
        """
        return self.module_registry.has_modules_attached(event_type)

    # Properties to maintain backward compatibility
    @property
    def cached_attention_weights(self):
        """Compatibility property for cached attention weights."""
        return None  # Could aggregate from attention_manager if needed

    @property
    def layer_attention_weights(self):
        """Property for layer-specific attention weights."""
        return self.attention_manager.layer_attention_weights

    @property
    def captured_attn(self):
        """Property for captured attention weights."""
        return self.attention_manager.captured_attn

    @property
    def decode_step(self):
        """Property for current decode step."""
        return self.attention_manager.decode_step

    def close(self):
        """Release hooks and cached instrumentation owned by this wrapper."""

        self.module_registry.detach_all_modules()
        self.attention_manager.cleanup()
        for wrapper in self.layer_wrappers.values():
            wrapper.close()
        if self._generation_hook is not None:
            self._generation_hook.remove()
            self._generation_hook = None

    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.close()
        except Exception:
            pass
