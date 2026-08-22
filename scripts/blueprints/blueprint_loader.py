import uuid
import json
import yaml
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
import os


class Blueprint(BaseModel):
    id: str
    data: str


class BlueprintsConfig(BaseModel):
    blueprints: List[Blueprint]


def load_blueprints(folder_path: str = "blueprints") -> BlueprintsConfig:
    blueprints = []
    os.makedirs(folder_path, exist_ok=True)
    for filename in os.listdir(folder_path):
        if filename.startswith("."):
            continue

        filepath = os.path.join(folder_path, filename)

        if os.path.isfile(filepath):
            try:
                with open(filepath, "r", encoding="utf-8") as file:
                    blueprint = Blueprint(id=filename, data=file.read())
                    blueprints.append(blueprint)
            except Exception as e:
                pass

    return BlueprintsConfig(blueprints=blueprints)


def save_blueprint(
    blueprint_data: str,
    blueprint_id: Optional[str] = None,
    folder_path: str = "blueprints",
) -> str:
    """Save raw blueprint payload and return saved file path."""
    os.makedirs(folder_path, exist_ok=True)
    blueprint_id = blueprint_id or str(uuid.uuid4())
    filename = blueprint_id if blueprint_id.endswith(".json") else f"{blueprint_id}.json"
    filepath = os.path.join(folder_path, filename)
    with open(filepath, "w", encoding="utf-8") as file:
        file.write(blueprint_data)
    return filepath


def remove_blueprint(blueprint_id: str, folder_path: str = "blueprints") -> bool:
    """Remove a blueprint by id/filename."""
    if not blueprint_id:
        return False
    candidates = [blueprint_id]
    if not blueprint_id.endswith(".json"):
        candidates.append(f"{blueprint_id}.json")

    for candidate in candidates:
        filepath = os.path.join(folder_path, candidate)
        if os.path.exists(filepath):
            os.remove(filepath)
            return True
    return False


def load_blueprint_from_model(
    model_state: Dict[str, Any], blueprint_id: Optional[str] = None
) -> Blueprint:
    """
    Create a blueprint payload from model state data.

    This helper keeps compatibility with older callers expecting a serialized
    `Blueprint` wrapper around model-derived content.
    """
    blueprint_id = blueprint_id or str(uuid.uuid4())
    serialized = json.dumps(model_state, ensure_ascii=False, indent=2)
    return Blueprint(id=blueprint_id, data=serialized)
