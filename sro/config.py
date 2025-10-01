# sro/config.py
from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any
import yaml
from pydantic import BaseModel, Field, PositiveInt, field_validator, model_validator, ConfigDict

class SROProverCfg(BaseModel):
    M: PositiveInt = Field(8, description="frontier size (first-hop keep)")
    L: PositiveInt = Field(24, description="second-hop pool size")
    B: PositiveInt = Field(64, description="pair budget (max (i,j) evals)")

    tau1: float = 0.75  # one-hop accept threshold
    tau2: float = 0.80  # two-hop accept threshold
    delta: float = 0.10 # safety margin (proof score − Cmax)
    kappa: float = 0.05 # optimism cushion in UB
    epsilon: float = 0.02 # alternation slack

    max_sentences_per_claim: PositiveInt = 64

    model_config = ConfigDict(extra="ignore")

    @field_validator("tau1", "tau2", "delta", "kappa", "epsilon")
    @classmethod
    def _prob_in_unit(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("value must be in [0,1]")
        return float(v)

    @model_validator(mode="after")
    def _cross_invariants(self) -> "SROProverCfg":
        if self.tau2 < self.tau1:
            raise ValueError(f"tau2 ({self.tau2}) must be >= tau1 ({self.tau1})")
        if self.M > self.max_sentences_per_claim:
            raise ValueError("M must be <= max_sentences_per_claim")
        if self.L > self.max_sentences_per_claim:
            raise ValueError("L must be <= max_sentences_per_claim")
        # Optional: enforce disjoint frontier/second pool budget
        # if self.M + self.L > self.max_sentences_per_claim:
        #     raise ValueError("M + L must be <= max_sentences_per_claim")
        return self

class PathsCfg(BaseModel):
    artifacts_dir: Path = Path("artifacts")
    proofs_dir: Path = Path("artifacts/proofs")
    logs_dir: Path = Path("artifacts/logs")

    model_config = ConfigDict(extra="ignore")

    def ensure(self) -> None:
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.proofs_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

class Config(BaseModel):
    sro_prover: SROProverCfg = SROProverCfg()
    paths: PathsCfg = PathsCfg()
    model_config = ConfigDict(extra="ignore")

# --- Back-compat for flat YAML ---
_FLAT_KEYS = {
    "M","L","B","tau1","tau2","delta","kappa","epsilon","max_sentences_per_claim"
}

def _normalize_data(data: Dict[str, Any]) -> Dict[str, Any]:
    if not data:
        return {}
    data = dict(data)
    if "sro_prover" in data and isinstance(data["sro_prover"], dict):
        return data
    flat = {k: data.pop(k) for k in list(data.keys()) if k in _FLAT_KEYS}
    if flat:
        data["sro_prover"] = flat
    return data

def load_config(path: Optional[str] = "configs/default.yaml") -> Config:
    data: Dict[str, Any] = {}
    if path:
        p = Path(path)
        if p.exists():
            try:
                with p.open("r", encoding="utf-8") as f:
                    data = yaml.safe_load(f) or {}
            except yaml.YAMLError as e:
                raise RuntimeError(f"Failed to parse YAML at {path}: {e}") from e
    data = _normalize_data(data)
    print('data',data)
    cfg = Config(**data)
    print('cfg',cfg)
    cfg.paths.ensure()
    return cfg
