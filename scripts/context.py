from fastapi import APIRouter
from scripts.blueprints.blueprint_loader import BlueprintsConfig
from scripts.configs.user_config import UserConfig
from scripts.messages.message_bus import MessageBus
from scripts.models.model_loader import ModelsRegistry
from scripts.modules.modules_store import ModulesStore
from scripts.utils.hf_auth import setup_hf_auth
import logging
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from scripts.core.module_loader import ModuleLoader

logger = logging.getLogger(__name__)


class LTTSContext:
    router: APIRouter
    message_bus: MessageBus
    modules_store: ModulesStore
    user_config: UserConfig
    models: ModelsRegistry
    blueprints: BlueprintsConfig
    module_loader: Optional["ModuleLoader"] = None
    ltts_integration: Optional[object] = None

    def __init__(self):
        auth_success = setup_hf_auth()
        if auth_success:
            logger.info("HuggingFace authentication configured successfully")
        else:
            logger.warning(
                "No HuggingFace token found - private models will not be accessible"
            )
