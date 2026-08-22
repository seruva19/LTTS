import os
import logging
from datetime import datetime
from typing import Any, Dict
from fastapi import APIRouter, Body, HTTPException
import yaml
from scripts.configs import user_files
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SettingsData(BaseModel):
    ui: Dict[str, Any] = Field(default_factory=dict)
    performance: Dict[str, Any] = Field(default_factory=dict)
    visualization: Dict[str, Any] = Field(default_factory=dict)
    api: Dict[str, Any] = Field(default_factory=dict)
    modules: Dict[str, Any] = Field(default_factory=dict)
    display: Dict[str, Any] = Field(default_factory=dict)
    controls: Dict[str, Any] = Field(default_factory=dict)


def load_default_config():
    """Load default configuration from YAML file."""
    try:
        default_config_path = "user/default_config.yaml"
        if os.path.exists(default_config_path):
            with open(default_config_path, "r") as f:
                return yaml.safe_load(f)
        else:
            # Fallback to hardcoded defaults if file doesn't exist
            return {
                "ui": {
                    "theme": "auto",
                    "compact_mode": False,
                    "show_layer_details": True,
                    "auto_expand_layers": False,
                },
                "performance": {
                    "max_concurrent_models": 3,
                    "enable_gpu": True,
                    "memory_limit": 8192,
                    "cache_size": 1024,
                },
                "visualization": {
                    "default_attention_view": "heatmap",
                    "show_token_ids": False,
                    "highlight_threshold": 0.1,
                    "color_scheme": "viridis",
                },
                "api": {
                    "server_url": "http://localhost:8000",
                    "timeout": 30000,
                    "retry_attempts": 3,
                },
                "modules": {
                    "auto_load_modules": True,
                    "enable_websocket_updates": True,
                    "data_retention_minutes": 60,
                    "registry_url": "https://ltts-registry.example.com/modules.json",
                },
                "blueprint": {
                    "autoload_blueprint": False,
                    "load_last_blueprint_on_launch": False,
                    "auto_save_on_changes": True,
                    "default_blueprint_name": "Untitled Blueprint",
                    "save_location": "localStorage",
                    "blueprints_folder": "blueprints/",
                },
            }
    except Exception as e:
        logger.error("Failed to load default config: %s", e)
        return {}


def merge_configs(default_config, user_config):
    """Merge user config with default config, preserving user overrides."""
    merged = default_config.copy()

    def deep_merge(base, override):
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                deep_merge(base[key], value)
            else:
                base[key] = value

    deep_merge(merged, user_config)
    return merged


def create_settings_router(context):
    """Create and configure the settings router with context dependency injection."""
    router = APIRouter()

    @router.get("/api/settings/defaults")
    async def get_default_settings():
        """Load default settings from default_config.yaml."""
        try:
            default_config = load_default_config()

            return {"status": "success", "settings": default_config}

        except Exception as e:
            logger.error("Failed to load default settings: %s", e)
            raise HTTPException(
                status_code=500, detail=f"Failed to load default settings: {str(e)}"
            )

    @router.get("/api/settings")
    async def get_settings():
        """Load user settings from YAML config files."""
        try:
            # Load default configuration
            default_config = load_default_config()

            # Load user configuration
            user_config_path = user_files.USER_CONFIG
            user_config = {}
            if os.path.exists(user_config_path):
                with open(user_config_path, "r") as f:
                    user_config = yaml.safe_load(f) or {}

            # Merge configurations
            settings = merge_configs(default_config, user_config)

            return {"status": "success", "settings": settings}

        except Exception as e:
            logger.error("Failed to load settings: %s", e)
            raise HTTPException(
                status_code=500, detail=f"Failed to load settings: {str(e)}"
            )

    @router.post("/api/settings")
    async def save_settings(request: Dict[str, Any] = Body(...)):
        """Save user settings to YAML config file or localStorage based on preference."""
        try:
            settings = request.get("settings", {})
            storage_location = request.get("storage_location", "localStorage")

            # Ensure user directory exists
            os.makedirs("user", exist_ok=True)

            # Always save to disk as backup, regardless of storage preference
            user_config_path = user_files.USER_CONFIG
            existing_config = {}
            if os.path.exists(user_config_path):
                with open(user_config_path, "r") as f:
                    existing_config = yaml.safe_load(f) or {}

            # Update with new settings
            settings_dict = {**settings, "updated_at": datetime.now().isoformat()}

            # Merge with existing config to preserve other settings
            merged_config = merge_configs(existing_config, settings_dict)

            # Save to YAML file
            with open(user_config_path, "w") as f:
                yaml.dump(merged_config, f, default_flow_style=False, indent=2)

            return {
                "status": "success",
                "message": f"Settings saved successfully to {storage_location}",
                "storage_location": storage_location,
            }

        except Exception as e:
            logger.error("Failed to save settings: %s", e)
            raise HTTPException(
                status_code=500, detail=f"Failed to save settings: {str(e)}"
            )

    @router.post("/api/settings/reset")
    async def reset_settings():
        """Reset settings to defaults by removing user overrides."""
        try:
            user_config_path = user_files.USER_CONFIG

            # Load existing config to preserve non-settings data
            existing_config = {}
            if os.path.exists(user_config_path):
                with open(user_config_path, "r") as f:
                    existing_config = yaml.safe_load(f) or {}

            # Remove settings categories but preserve other config
            settings_keys = [
                "ui",
                "performance",
                "visualization",
                "api",
                "modules",
                "display",
                "controls",
                "storage",
                "blueprints",
                "load_last_blueprint_on_launch",
            ]
            for key in settings_keys:
                existing_config.pop(key, None)

            # Save the cleaned config
            with open(user_config_path, "w") as f:
                yaml.dump(existing_config, f, default_flow_style=False, indent=2)

            # Return default settings
            default_config = load_default_config()

            return {
                "status": "success",
                "message": "Settings reset to defaults",
                "settings": default_config,
            }

        except Exception as e:
            logger.error("Failed to reset settings: %s", e)
            raise HTTPException(
                status_code=500, detail=f"Failed to reset settings: {str(e)}"
            )

    return router
