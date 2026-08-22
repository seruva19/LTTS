from typing import Optional

from pydantic import BaseModel, Field, ConfigDict


class ModelLoadingConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    low_cpu_mem_usage: bool = True


class PerformanceConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    max_concurrent_models: int = 3
    enable_gpu: bool = True
    memory_limit: int = 8192
    cache_size: int = 1024
    data_retention_minutes: int = 60


class UserConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    llm_engine: str = "transformers"
    autoload_blueprint: bool = False
    model_loading: ModelLoadingConfig = Field(default_factory=ModelLoadingConfig)
    performance: Optional[PerformanceConfig] = Field(default_factory=PerformanceConfig)
