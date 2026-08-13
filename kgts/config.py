"""Pipeline configuration: YAML -> typed pydantic settings (design appendix A)."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field


class SeedSpec(BaseModel):
    label: str
    description: str = ""
    children: list[str] = Field(default_factory=list)  # human-given first layer


class RunConfig(BaseModel):
    name: str = "default"
    workdir: str = ".kgts"
    seed: int = 42


class LLMConfig(BaseModel):
    model: str = "gpt-4o-mini"
    cheap_model: str | None = None
    api_base: str | None = None
    rpm: int | None = 60
    cache: bool = True


class BudgetConfig(BaseModel):
    max_llm_calls: int | None = 2000
    max_cost_usd: float | None = 10.0
    max_nodes: int = 500
    max_explore_rounds: int = 4


class AlignConfig(BaseModel):
    recall_top_k: int = 8
    embed_threshold: float = 0.75


class AtomicityConfig(BaseModel):
    min_materials: int = 5
    min_synth_success: float = 0.7
    synth_trials: int = 3
    max_child_material_overlap: float = 0.5


class BuildConfig(BaseModel):
    align: AlignConfig = Field(default_factory=AlignConfig)
    atomicity: AtomicityConfig = Field(default_factory=AtomicityConfig)
    max_parents: int = 3


class SampleQuotas(BaseModel):
    min_node_coverage: int = 1
    max_per_node: int = 20
    long_tail_alpha: float = 1.0


class JointConfig(BaseModel):
    k_hop: int = 2
    max_group_size: int = 4


class SampleConfig(BaseModel):
    mixture: dict[str, float] = Field(
        default_factory=lambda: {"breadth": 0.2, "depth": 0.6, "joint": 0.2}
    )
    n_samples: int = 1000
    quotas: SampleQuotas = Field(default_factory=SampleQuotas)
    joint: JointConfig = Field(default_factory=JointConfig)
    prioritizer: str = "inverse_frequency"


class WebSourceConfig(BaseModel):
    provider: str = "tavily"
    api_key_env: str = "TAVILY_API_KEY"


class LocalSourceConfig(BaseModel):
    paths: list[str] = Field(default_factory=lambda: ["./corpus"])
    chunk_size: int = 800
    chunk_overlap: int = 100


class PostprocessConfig(BaseModel):
    dedup: str = "simhash"  # simhash | url
    rerank: str = "none"  # cross_encoder | llm | none
    license_mode: str = "whitelist"  # web materials default to whitelist


class RetrieveConfig(BaseModel):
    sources: list[str] = Field(default_factory=lambda: ["local"])
    per_node_materials: int = 8
    web: WebSourceConfig = Field(default_factory=WebSourceConfig)
    local: LocalSourceConfig = Field(default_factory=LocalSourceConfig)
    postprocess: PostprocessConfig = Field(default_factory=PostprocessConfig)


class SynthesizeConfig(BaseModel):
    task_types: list[str] = Field(
        default_factory=lambda: [
            "atomic_qa",
            "aggregated_qa",
            "multihop_qa",
            "grounded_summary",
            "comparative_analysis",
        ]
    )
    type_weights: dict[str, float] = Field(default_factory=dict)
    style: dict[str, Any] = Field(
        default_factory=lambda: {"language": "zh", "length": "medium", "difficulty": "mixed"}
    )
    insufficient_material: str = "reject_sample"  # reject_sample | drop


class RubricJudgeConfig(BaseModel):
    model: str | None = None  # None -> reuse llm.model
    pass_score: float = 0.7


class VerifyConfig(BaseModel):
    fallback: str = "rubric_judge"
    rubric_judge: RubricJudgeConfig = Field(default_factory=RubricJudgeConfig)


class ExportConfig(BaseModel):
    formats: list[str] = Field(default_factory=lambda: ["sft", "rl"])
    out_dir: str = "./output"


class Config(BaseModel):
    run: RunConfig = Field(default_factory=RunConfig)
    seeds: list[SeedSpec] = Field(default_factory=list)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    budget: BudgetConfig = Field(default_factory=BudgetConfig)
    build: BuildConfig = Field(default_factory=BuildConfig)
    sample: SampleConfig = Field(default_factory=SampleConfig)
    retrieve: RetrieveConfig = Field(default_factory=RetrieveConfig)
    synthesize: SynthesizeConfig = Field(default_factory=SynthesizeConfig)
    verify: VerifyConfig = Field(default_factory=VerifyConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)

    def config_hash(self) -> str:
        return hashlib.sha256(self.model_dump_json().encode()).hexdigest()[:16]


def load_config(path: str | Path) -> Config:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    return Config.model_validate(raw)
