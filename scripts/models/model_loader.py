import logging
import os
import yaml
from pathlib import Path
import importlib.util
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, Callable
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoModelForQuestionAnswering,
    AutoModelForSeq2SeqLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
)
from scripts.models.model_instance import ModelInstance
from scripts.core.ltts_model import LTTSModel
from scripts.models.architecture_adapter import ENCODER_MODEL_TYPES
from scripts.models.model_struct import ModelStruct, traverse_model
from scripts.utils.hf_auth import setup_hf_auth, get_hf_token
from scripts.configs import user_files

logger = logging.getLogger(__name__)

from scripts.utils.event_stream import EventStream


def _is_invalid_repo_error(exc: Exception) -> bool:
    """Type-based fallback for invalid-repo errors whose message differs."""
    try:
        from huggingface_hub.utils import HFValidationError

        return isinstance(exc, HFValidationError)
    except ImportError:
        return False


def _load_dummy_model_helpers():
    """
    Load dummy model helpers without relying on `tests` package resolution.

    Some environments already have a site-packages module named `tests`, which can
    shadow this repository's local `tests/` directory.
    """
    dummy_path = Path(__file__).resolve().parents[2] / "tests" / "dummy_model.py"
    if not dummy_path.exists():
        raise FileNotFoundError(
            f"Missing debug dummy model helper at {dummy_path}. "
            "Restore tests/dummy_model.py or remove debug/dummy from models config."
        )

    spec = importlib.util.spec_from_file_location("ltts_local_dummy_model", dummy_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to create import spec for {dummy_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    create_dummy_ltts_model = getattr(module, "create_dummy_ltts_model", None)
    create_dummy_model_struct = getattr(module, "create_dummy_model_struct", None)
    if not callable(create_dummy_ltts_model) or not callable(create_dummy_model_struct):
        raise ImportError(
            "tests/dummy_model.py is missing required callables: "
            "create_dummy_ltts_model and create_dummy_model_struct"
        )

    return create_dummy_ltts_model, create_dummy_model_struct


class ModelConfigItem(BaseModel):
    id: str
    data: Optional[Dict[str, Any]] = Field(default_factory=dict)


class ModelsRegistry:
    config_list: dict[str, ModelConfigItem]
    ltts_models: Dict[str, ModelInstance]

    def __init__(self, user_config=None):
        create_dummy_ltts_model, create_dummy_model_struct = _load_dummy_model_helpers()

        self.config_list = {}
        self.ltts_models = {
            "debug/dummy": ModelInstance(
                model=None, loaded=False, struct=create_dummy_model_struct()
            )
        }
        self._create_dummy_ltts_model = create_dummy_ltts_model

        self.event_stream = EventStream()
        self.hf_authenticated = setup_hf_auth()
        self.user_config = user_config

    def load_models(self, file_path: str = "./user/models.user.yaml"):
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"Missing required models file: {file_path}. "
                "Create it before starting the app."
            )
        with open(file_path, "r") as file:
            config = yaml.safe_load(file)

        models_dict = config.get("models", {})
        self.config_list = {}
        for model_id, model_data in models_dict.items():
            if isinstance(model_data, dict):
                self.config_list[model_id] = ModelConfigItem(
                    id=model_data.get("id", model_id), data=model_data.get("data", {})
                )
            else:
                self.config_list[model_id] = ModelConfigItem(id=model_id, data={})

    # Task inferred from the config's architecture when the caller does not
    # name one. Keys are the model's own `architectures` suffixes, which is the
    # only signal available before any weights are downloaded.
    _TASK_BY_ARCHITECTURE_SUFFIX = (
        ("ForCausalLM", "causal_lm"),
        ("LMHeadModel", "causal_lm"),
        ("ForMaskedLM", "masked_lm"),
        ("ForConditionalGeneration", "seq2seq_lm"),
        ("ForQuestionAnswering", "question_answering"),
        ("ForSequenceClassification", "sequence_classification"),
        ("ForTokenClassification", "token_classification"),
    )

    def describe_remote_model(self, model_id: str) -> Dict[str, Any]:
        """Fetch a model's config (not its weights) and infer how to load it.

        Downloads config.json only - a few kilobytes - so a mistyped repository
        fails here instead of part-way through a multi-gigabyte download.
        """
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(model_id)
        architectures = list(getattr(config, "architectures", None) or [])
        task = None
        for architecture in architectures:
            for suffix, candidate in self._TASK_BY_ARCHITECTURE_SUFFIX:
                if architecture.endswith(suffix):
                    task = candidate
                    break
            if task:
                break
        if task is None:
            task = "seq2seq_lm" if getattr(config, "is_encoder_decoder", False) else "base_model"

        return {
            "model_type": getattr(config, "model_type", None),
            "architectures": architectures,
            "task": task,
        }

    def add_model(
        self,
        model_id: str,
        *,
        task: Optional[str] = None,
        validate: bool = True,
        file_path: str = "./user/models.user.yaml",
    ) -> Dict[str, Any]:
        """Register a Hugging Face model id in the personal model registry."""
        model_id = (model_id or "").strip().strip("/")
        if not model_id:
            raise ValueError("Model id is empty")
        if model_id in self.config_list:
            raise ValueError(f"Model '{model_id}' is already registered")
        if task is not None and task not in self._AUTO_MODEL_TASKS:
            supported = ", ".join(sorted(self._AUTO_MODEL_TASKS))
            raise ValueError(f"Unsupported task '{task}'. Supported: {supported}")

        described = {}
        if validate:
            described = self.describe_remote_model(model_id)
            task = task or described.get("task")

        entry = {"id": model_id, "data": {}}
        if task:
            entry["data"]["task"] = task

        registry = user_files.read_yaml(file_path)
        models = registry.get("models")
        if not isinstance(models, dict):
            models = {}
        models[model_id] = entry
        registry["models"] = models
        user_files.write_yaml(file_path, registry)

        self.load_models(file_path)
        logger.info("Registered model '%s' (task=%s)", model_id, task or "auto")
        return {"model_id": model_id, "task": task, **described}

    def remove_model(
        self, model_id: str, *, file_path: str = "./user/models.user.yaml"
    ) -> None:
        """Drop a model from the personal registry (its cache is left alone)."""
        registry = user_files.read_yaml(file_path)
        models = registry.get("models")
        if not isinstance(models, dict) or model_id not in models:
            raise KeyError(f"model '{model_id}' is not registered")

        self.unload_model(model_id)
        del models[model_id]
        registry["models"] = models
        user_files.write_yaml(file_path, registry)
        # A shipped model would otherwise be put back by the next start, when
        # the personal registry is reconciled with the defaults.
        user_files.forget_default_model(model_id, user_path=file_path)
        self.load_models(file_path)
        logger.info("Removed model '%s' from the registry", model_id)

    def get_model(self, key) -> ModelInstance:
        if key in self.ltts_models:
            return self.ltts_models[key]
        else:
            return None

    # Maximum number of loaded models (including debug/dummy).
    # Oldest non-requested models are evicted when the limit is reached.
    _MAX_LOADED_MODELS = 3
    _AUTO_MODEL_TASKS = {
        "base_model": AutoModel,
        "causal_lm": AutoModelForCausalLM,
        "masked_lm": AutoModelForMaskedLM,
        "question_answering": AutoModelForQuestionAnswering,
        "seq2seq_lm": AutoModelForSeq2SeqLM,
        "sequence_classification": AutoModelForSequenceClassification,
        "token_classification": AutoModelForTokenClassification,
    }

    def load_model(self, model_id, engine, **params):
        if model_id != "debug/dummy" and model_id not in self.config_list:
            raise KeyError(f"model '{model_id}' not found in registry")

        if model_id in self.ltts_models:
            self.unload_model(model_id)

        if model_id not in self.ltts_models:
            # Evict least-recently-loaded models to stay within limit
            while len(self.ltts_models) >= self._MAX_LOADED_MODELS:
                candidates = {
                    mid: inst for mid, inst in self.ltts_models.items()
                    if mid != model_id
                }
                if not candidates:
                    break
                oldest_id = min(
                    candidates,
                    key=lambda mid: candidates[mid].loaded_at,
                )
                logger.info(
                    "Evicting model %s (LRU) to stay within %d-model limit",
                    oldest_id,
                    self._MAX_LOADED_MODELS,
                )
                self.unload_model(oldest_id)

            cb = self._make_callback(model_id)
            ltts_model, struct = self.load_ltts_model(model_id, engine, cb)
            self.ltts_models[model_id] = ModelInstance(
                model=ltts_model, loaded=True, struct=struct
            )
        instance = self.ltts_models[model_id]
        return instance.model, instance.struct

    def unload_model(self, model_id):
        if model_id in self.ltts_models:
            instance = self.ltts_models[model_id]
            assert instance is not None
            instance.flush()
            del self.ltts_models[model_id]

        else:
            logger.warning("Unload requested for unknown model_id: %s", model_id)

    def load_ltts_model(
        self, model_id: str, engine: str, callback
    ) -> tuple[LTTSModel, ModelStruct]:
        if model_id == "debug/dummy":
            return self._create_dummy_ltts_model(callback)

        source_id = self._resolve_model_source(model_id)
        hf_token = get_hf_token()
        model_kwargs = self._resolve_model_load_kwargs(model_id)
        shared_kwargs = {
            key: model_kwargs[key]
            for key in ("revision", "trust_remote_code", "subfolder")
            if key in model_kwargs
        }

        try:
            config = AutoConfig.from_pretrained(
                source_id, cache_dir="models", token=hf_token, **shared_kwargs
            )
            model_task, auto_model_class = self._resolve_model_task(model_id, config)
            tokenizer = AutoTokenizer.from_pretrained(
                source_id, cache_dir="models", token=hf_token, **shared_kwargs
            )
            model = auto_model_class.from_pretrained(
                source_id, cache_dir="models", token=hf_token, **model_kwargs
            )
        except Exception as e:
            if "not a valid model identifier" in str(e) or _is_invalid_repo_error(e):
                raise ValueError(
                    f"Model '{source_id}' not found. Check model name or ensure HF token is valid for private repos."
                ) from e
            raise

        device = self._resolve_device()
        # When device_map was used, accelerate already placed the weights on the
        # GPU (and .to() would raise), so only move explicitly otherwise.
        if device != "cpu" and "device_map" not in model_kwargs:
            try:
                model = model.to(device)
                logger.info("Moved model '%s' to %s", model_id, device)
            except Exception as move_error:
                logger.warning(
                    "Could not move model '%s' to %s (%s); staying on CPU",
                    model_id,
                    device,
                    move_error,
                )

        ltts_model = LTTSModel(model, message_bus=callback)
        # Keep the exact tokenizer and revision used at load time. Inference,
        # compatibility checks, and run exports can then avoid a second Hub
        # lookup and stay tied to the same model snapshot.
        ltts_model._tokenizer = tokenizer
        ltts_model._model_revision = getattr(config, "_commit_hash", None)
        ltts_model._model_task = model_task
        struct = traverse_model(ltts_model, "model", max_traversal=True)
        return ltts_model, struct

    def _resolve_model_source(self, model_id: str) -> str:
        """Resolve a registry alias to the actual Hub/local model identifier."""

        config_item = self.config_list.get(model_id)
        if config_item and config_item.id:
            return str(config_item.id)
        return str(model_id)

    def _resolve_model_task(self, model_id: str, config: Any):
        """Select the Hugging Face auto-model loader for the configured task."""

        config_item = self.config_list.get(model_id)
        configured_task = None
        if config_item and isinstance(config_item.data, dict):
            configured_task = config_item.data.get("task")

        if configured_task is not None:
            task = str(configured_task).strip().lower()
            if task not in self._AUTO_MODEL_TASKS:
                supported = ", ".join(sorted(self._AUTO_MODEL_TASKS))
                raise ValueError(
                    f"Unsupported model task '{configured_task}'. Supported tasks: {supported}"
                )
            return task, self._AUTO_MODEL_TASKS[task]

        architectures = " ".join(getattr(config, "architectures", None) or []).lower()
        model_type = str(getattr(config, "model_type", "") or "").lower()

        if bool(getattr(config, "is_encoder_decoder", False)):
            task = "seq2seq_lm"
        elif "questionanswering" in architectures:
            task = "question_answering"
        elif "tokenclassification" in architectures:
            task = "token_classification"
        elif "sequenceclassification" in architectures:
            task = "sequence_classification"
        elif "maskedlm" in architectures or model_type in ENCODER_MODEL_TYPES:
            task = "masked_lm"
        elif "causallm" in architectures or "lmheadmodel" in architectures:
            task = "causal_lm"
        else:
            task = "base_model"

        return task, self._AUTO_MODEL_TASKS[task]

    def _resolve_device(self) -> str:
        """Resolve the device to place models on, honoring performance.enable_gpu.

        The loader historically left every model on CPU regardless of available
        hardware; this honors the config so an available CUDA GPU is used.
        """
        try:
            import torch

            perf = getattr(self.user_config, "performance", None)
            enable_gpu = getattr(perf, "enable_gpu", True) if perf is not None else True
            if enable_gpu and torch.cuda.is_available():
                return "cuda"
        except Exception as e:
            logger.debug("Device resolution failed (%s); defaulting to CPU", e)
        return "cpu"

    def _resolve_model_load_kwargs(self, model_id: str) -> Dict[str, Any]:
        """
        Resolve Hugging Face `from_pretrained` keyword arguments, honoring
        default settings and per-model overrides.
        """
        kwargs: Dict[str, Any] = {"low_cpu_mem_usage": True}

        model_loading = getattr(self.user_config, "model_loading", None)
        if model_loading is not None:
            low_cpu = getattr(model_loading, "low_cpu_mem_usage", None)
            if low_cpu is not None:
                kwargs["low_cpu_mem_usage"] = low_cpu

        config_item = self.config_list.get(model_id)
        if config_item and isinstance(config_item.data, dict):
            load_kwargs = config_item.data.get("load_kwargs")
            if isinstance(load_kwargs, dict):
                kwargs.update(load_kwargs)
            low_cpu_override = config_item.data.get("low_cpu_mem_usage")
            if isinstance(low_cpu_override, bool):
                kwargs["low_cpu_mem_usage"] = low_cpu_override

        # On GPU, stream weights straight onto the device (device_map) to avoid a
        # full CPU staging copy + separate .to(device). Deliberately keep the
        # default float32 dtype: interpretability modules convert activations to
        # NumPy, which has no bfloat16, so loading in bf16 breaks them.
        if self._resolve_device() != "cpu":
            kwargs.setdefault("device_map", "cuda")

        return {key: value for key, value in kwargs.items() if value is not None}

    def _make_callback(self, model_id: str) -> Callable[[Dict[str, Any]], None]:
        def cb(event: Dict[str, Any]) -> None:
            self.event_stream.push(model_id, event)

        return cb
