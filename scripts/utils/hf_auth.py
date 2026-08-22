"""
HuggingFace Authentication Utilities

Automatically loads HF token from keys/hf.key file for model access.
"""

import os
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def load_hf_token() -> Optional[str]:
    try:
        project_root = Path(__file__).parent.parent.parent
        token_file = project_root / "keys" / "hf.key"

        logger.debug("Looking for HF token at: %s", token_file.absolute())

        if token_file.exists():
            with open(token_file, "r", encoding="utf-8") as f:
                token = f.read().strip()

            if token:
                logger.info("HuggingFace token loaded from %s", token_file)
                return token
            else:
                logger.warning("Token file %s exists but is empty", token_file)
        else:
            logger.info("No HuggingFace token file found at %s", token_file)
            logger.info(
                "To use private HF models, create keys/hf.key with your HuggingFace token"
            )

    except Exception as e:
        logger.error("Error loading HuggingFace token: %s", e)

    return None


def setup_hf_auth() -> bool:
    token = load_hf_token()

    if token:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = token
        os.environ["HF_TOKEN"] = token

        try:
            from huggingface_hub import login

            login(token=token, add_to_git_credential=False, write_permission=False)
            logger.info(
                "Successfully authenticated with HuggingFace Hub (session only)"
            )
            return True
        except ImportError:
            logger.warning(
                "huggingface_hub not available, using environment variables only"
            )
            return True
        except Exception as e:
            logger.warning(
                "HuggingFace Hub login failed, using environment variables: %s", e
            )
            return True

    return False


def get_hf_token() -> Optional[str]:
    token = os.environ.get("HUGGINGFACE_HUB_TOKEN") or os.environ.get("HF_TOKEN")

    if not token:
        token = load_hf_token()
        if token:
            os.environ["HUGGINGFACE_HUB_TOKEN"] = token
            os.environ["HF_TOKEN"] = token

    return token
