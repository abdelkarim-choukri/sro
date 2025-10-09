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
        # raise ValueError("M + L must be <= max_sentences_per_claim")
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

class RetrievalCfg(BaseModel):
    # corpus_jsonl: path to corpus JSONL file (one JSON object per line with fields: source_id, text).
    corpus_jsonl: Path = Path("data/corpus/corpus.jsonl")
    k_bm25: PositiveInt = 200
    k_dense: PositiveInt = 200
    k_fused: PositiveInt = 24          # how many sentences we keep after RRF+MMR
    mmr_lambda: float = 0.7            # MMR tradeoff λ (relevance vs redundancy)
    rrf_c: PositiveInt = 60            # RRF constant
    use_cross_encoder: bool = True     # try CE rerank; fallback if it fails
    rerank_top: PositiveInt = 64       # only top-K go to CE

    model_config = ConfigDict(extra="ignore")

    @field_validator("mmr_lambda")
    @classmethod
    def _lambda01(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("mmr_lambda must be in [0,1]")
        return float(v)

class ClaimsCfg(BaseModel):
    # K: max number of claims we attempt to prove
    K: PositiveInt = 3
    # ψ (psi): min cosine similarity between question and sentence to consider it a claim
    min_question_cosine: float = 0.30
    # H: regex words that indicate hedging / speculation → drop
    hedge_terms: list[str] = [
        r"\brumor(s|ed)?\b",
        r"\breportedly\b",
        r"\bmay\b",
        r"\bmight\b",
        r"\bpossibly\b",
        r"\bsuggest(ed|s|ing)?\b",
        r"\baccording to (sources|rumors)\b",
    ]
    # w_src: per-source head weight (prefix before ':'); defaults to 1.0 if not present
    reliability_weights: Dict[str, float] = {
        "news": 1.00,
        "press": 0.95,
        "blog": 0.60,
        "seed": 0.80,
        "alt": 0.75,
    }

    model_config = ConfigDict(extra="ignore")

    @field_validator("min_question_cosine")
    @classmethod
    def _psi01(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("min_question_cosine must be in [0,1]")
        return float(v)

class Config(BaseModel):
    sro_prover: SROProverCfg = SROProverCfg()
    paths: PathsCfg = PathsCfg()
    retrieval: RetrievalCfg = RetrievalCfg()
    claims: ClaimsCfg = ClaimsCfg()          # <-- NEW
    model_config = ConfigDict(extra="ignore")

# --- Back-compat for flat YAML ---
_FLAT_KEYS = {
    "M","L","B","tau1","tau2","delta","kappa","epsilon","max_sentences_per_claim"
}

def _normalize_data(data: Dict[str, Any]) -> Dict[str, Any]:
    if not data:
        return {}
    data = dict(data)
    # existing flat → sro_prover migration
    flat = {k: data.pop(k) for k in list(data.keys()) if k in _FLAT_KEYS}
    if flat:
        data.setdefault("sro_prover", {}).update(flat)
    # allow users to omit retrieval; defaults will fill
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
    cfg = Config(**data)
    
    cfg.paths.ensure()
    return cfg

