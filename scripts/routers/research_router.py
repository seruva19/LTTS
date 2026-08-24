from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from ltts_engine import (
    ResearchStudyArchive,
    StudyArtifactCollection,
    bundle_study_archive,
    compare_study_artifacts,
    research_bundle_from_dict,
    study_archive_from_bundle,
)
from scripts.services.research_study import ResearchStudyService
from scripts.services.causal_scan import CausalScanService


class ResearchExampleInput(BaseModel):
    id: str = Field(..., min_length=1, max_length=256)
    prompt: str = Field(..., min_length=1, max_length=100_000)


class ResearchManifestInput(BaseModel):
    name: str = Field(..., min_length=1, max_length=256)
    split: str = Field("validation", pattern="^(train|validation|test)$")
    revision: Optional[str] = Field(None, max_length=256)


class ResearchStudyInput(BaseModel):
    name: str = Field("Research Study", min_length=1, max_length=256)
    model_id: str = Field(..., min_length=1, max_length=256)
    manifest: ResearchManifestInput
    examples: List[ResearchExampleInput] = Field(..., min_length=1, max_length=32)
    seeds: List[int] = Field(default_factory=lambda: [0], min_length=1, max_length=16)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    client_token: Optional[str] = Field(None, max_length=128)
    modules: Optional[Dict[str, List[Union[str, Dict[str, Any]]]]] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    @field_validator("seeds")
    @classmethod
    def validate_seeds(cls, seeds: List[int]) -> List[int]:
        if len(seeds) != len(set(seeds)):
            raise ValueError("Research study seeds must be unique")
        for seed in seeds:
            if seed < 0 or seed > 2**32 - 1:
                raise ValueError("Research study seeds must be uint32 values")
        return seeds

    @field_validator("examples")
    @classmethod
    def validate_examples(cls, examples: List[ResearchExampleInput]):
        ids = [example.id for example in examples]
        if len(ids) != len(set(ids)):
            raise ValueError("Research study example ids must be unique")
        return examples


class ResearchArchiveInput(BaseModel):
    archive: Dict[str, Any]


class ResearchBundleInput(BaseModel):
    bundle: Dict[str, Any]


class ResearchComparisonInput(BaseModel):
    baseline: Dict[str, Any]
    candidate: Dict[str, Any]
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CausalScanInput(BaseModel):
    model_id: str = Field(..., min_length=1, max_length=256)
    prompt: str = Field(..., min_length=1, max_length=100_000)
    layers: List[str] = Field(..., min_length=1, max_length=256)
    seeds: List[int] = Field(default_factory=lambda: [0], min_length=1, max_length=16)
    parameters: Dict[str, Any] = Field(default_factory=dict)
    modules: Optional[Dict[str, List[Union[str, Dict[str, Any]]]]] = None
    intervention: Dict[str, Any]
    measurement_module: str = Field(..., min_length=1, max_length=256)
    metric: str = Field("value", min_length=1, max_length=128)
    client_token: Optional[str] = Field(None, max_length=128)

    @field_validator("layers")
    @classmethod
    def validate_layers(cls, layers: List[str]) -> List[str]:
        normalized = [layer.strip() for layer in layers]
        if any(not layer for layer in normalized) or len(normalized) != len(set(normalized)):
            raise ValueError("Causal scan layers must be unique non-empty paths")
        return normalized

    @field_validator("seeds")
    @classmethod
    def validate_causal_seeds(cls, seeds: List[int]) -> List[int]:
        if len(seeds) != len(set(seeds)) or any(seed < 0 or seed > 2**32 - 1 for seed in seeds):
            raise ValueError("Causal scan seeds must be unique uint32 values")
        return seeds


def _unprocessable(operation):
    try:
        return operation()
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def create_research_router(context):
    router = APIRouter()
    service = ResearchStudyService(context)
    causal_scan = CausalScanService(context)

    @router.post("/api/research/studies/run")
    async def run_research_study(input: ResearchStudyInput):
        return await service.run(
            name=input.name,
            model_id=input.model_id,
            manifest=input.manifest.model_dump(exclude_none=True),
            examples=[example.model_dump() for example in input.examples],
            seeds=input.seeds,
            parameters=input.parameters,
            modules=input.modules,
            client_token=input.client_token or "default",
            metadata=input.metadata,
        )

    @router.post("/api/research/studies/archive")
    async def archive_research_study(input: ResearchArchiveInput):
        archive = _unprocessable(lambda: ResearchStudyArchive.from_dict(input.archive))
        return bundle_study_archive(archive).to_dict()

    @router.post("/api/research/studies/restore")
    async def restore_research_study(input: ResearchBundleInput):
        return _unprocessable(
            lambda: study_archive_from_bundle(
                research_bundle_from_dict(input.bundle)
            ).to_dict()
        )

    @router.post("/api/research/studies/compare")
    async def compare_research_studies(input: ResearchComparisonInput):
        return _unprocessable(
            lambda: compare_study_artifacts(
                StudyArtifactCollection.from_dict(input.baseline),
                StudyArtifactCollection.from_dict(input.candidate),
                metadata=input.metadata,
            ).to_dict()
        )

    @router.post("/api/research/causal-scan")
    async def run_causal_scan(input: CausalScanInput):
        return await causal_scan.run(
            model_id=input.model_id,
            prompt=input.prompt,
            layers=input.layers,
            seeds=input.seeds,
            parameters=input.parameters,
            modules=input.modules,
            intervention=input.intervention,
            measurement_module=input.measurement_module,
            metric=input.metric,
            client_token=input.client_token or "default",
        )

    return router
