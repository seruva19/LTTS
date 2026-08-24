"""Versioned, integrity-checked storage for headless research results."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


def _normalize_numbers(value: Any) -> Any:
    """Make checksum semantics stable across JSON clients using 1 or 1.0."""

    if isinstance(value, Mapping):
        return {str(key): _normalize_numbers(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_normalize_numbers(item) for item in value]
    if isinstance(value, float) and value.is_integer():
        return int(value)
    return value


def _canonical(payload: Mapping[str, Any]) -> bytes:
    normalized_payload = dict(payload)
    created_at = normalized_payload.get("created_at")
    if isinstance(created_at, str):
        try:
            normalized_payload["created_at"] = datetime.fromisoformat(
                created_at.replace("Z", "+00:00")
            ).astimezone(timezone.utc).isoformat()
        except ValueError:
            # Invalid timestamps are preserved and validated by the caller's
            # schema; checksum calculation must remain deterministic.
            pass
    try:
        encoded = json.dumps(
            _normalize_numbers(normalized_payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Research bundle data must be finite and JSON-serializable") from exc
    return encoded.encode("utf-8")


def _checksum(payload: Mapping[str, Any]) -> str:
    return "sha256:" + hashlib.sha256(_canonical(payload)).hexdigest()


def _legacy_checksum(payload: Mapping[str, Any]) -> str:
    """Checksum emitted before numeric canonicalization; accepted for imports."""

    try:
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Research bundle data must be finite and JSON-serializable") from exc
    return "sha256:" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ResearchRunBundle:
    kind: str
    payload: Mapping[str, Any]
    metadata: Mapping[str, Any] = field(default_factory=dict)
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not self.kind.strip():
            raise ValueError("Research bundle kind cannot be empty")
        _canonical(self._content())

    def _content(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "created_at": self.created_at,
            "metadata": dict(self.metadata),
            "payload": dict(self.payload),
        }

    def to_dict(self) -> Dict[str, Any]:
        content = self._content()
        return {**content, "checksum": _checksum(content)}


def bundle_result(
    kind: str,
    result: Any,
    *,
    metadata: Optional[Mapping[str, Any]] = None,
) -> ResearchRunBundle:
    """Create a bundle from a result dataclass or an ordinary mapping."""

    converter = getattr(result, "to_dict", None)
    payload = converter() if callable(converter) else result
    if not isinstance(payload, Mapping):
        raise TypeError("Bundled result must be a mapping or expose to_dict()")
    return ResearchRunBundle(
        kind=str(kind), payload=dict(payload), metadata=dict(metadata or {})
    )


def save_research_bundle(bundle: ResearchRunBundle, path: "str | Path") -> Path:
    """Atomically save a bundle without leaving a partial destination file."""

    if not isinstance(bundle, ResearchRunBundle):
        raise TypeError("bundle must be ResearchRunBundle")
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(
        bundle.to_dict(), indent=2, ensure_ascii=False, allow_nan=False
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=str(destination.parent)
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(serialized)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary.exists():
            temporary.unlink()
    return destination


def load_research_bundle(path: "str | Path") -> ResearchRunBundle:
    source = Path(path).expanduser().resolve()
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read research bundle: {source}") from exc
    return research_bundle_from_dict(document)


def research_bundle_from_dict(document: Mapping[str, Any]) -> ResearchRunBundle:
    """Validate an in-memory bundle document without requiring filesystem I/O."""

    if not isinstance(document, Mapping):
        raise ValueError("Research bundle root must be an object")
    document = dict(document)
    checksum = document.pop("checksum", None)
    accepted_checksums = {_checksum(document), _legacy_checksum(document)}
    if not isinstance(checksum, str) or checksum not in accepted_checksums:
        raise ValueError("Research bundle checksum mismatch")
    if document.get("schema_version") != 1:
        raise ValueError(
            f"Unsupported research bundle schema: {document.get('schema_version')}"
        )
    payload = document.get("payload")
    metadata = document.get("metadata")
    if not isinstance(payload, dict) or not isinstance(metadata, dict):
        raise ValueError("Research bundle payload and metadata must be objects")
    return ResearchRunBundle(
        kind=str(document.get("kind") or ""),
        payload=payload,
        metadata=metadata,
        created_at=str(document.get("created_at") or ""),
        schema_version=1,
    )
