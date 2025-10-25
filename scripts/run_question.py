# scripts/run_question.py
"""
Does:
    Entry point to run the SRO-Proof pipeline with explicit, early NLI initialization
    and logging of the calibrated temperature. Robust to import variations.
    - Sets logging early (respects SRO_LOGLEVEL).
    - Seeds deterministically.
    - Builds ONE shared NLIBackend (respects --nli_model_dir or env SRO_NLI_MODEL_DIR).
    - Logs its effective temperature (acceptance requirement).
    - Tries multiple ways to run the pipeline:
        1) SROProver (preferred), passing the shared NLI backend.
        2) Legacy functions on the prover object (run(), run_question(...), answer(...)).
        3) If import fails, show full traceback and continue trying a legacy module
           'scripts.run_question_legacy' if present.
    - If nothing workable exists, raises a clear RuntimeError after logging NLI.

CLI:
    python -m scripts.run_question --seed 42 --profile med --bs_nli1 64 --bs_nli2 64
    python -m scripts.run_question --seed 42 --profile med --question "Does the iPhone 15 Pro have a titanium frame?"
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import logging
import os
import random
import sys
import traceback
from typing import Any, Dict, Optional

import numpy as np
import torch

# -------------- logging first --------------
logging.basicConfig(
    level=os.environ.get("SRO_LOGLEVEL", "INFO"),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

from sro.nli.backend import NLIBackend  # noqa: E402


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _print_config_banner(profile: str) -> None:
    if profile == "med":
        M, L, B = 8, 24, 64
        tau1, tau2, delta, kappa, eps = 0.75, 0.80, 0.10, 0.80, 0.02
    elif profile == "tiny":
        M, L, B = 4, 12, 32
        tau1, tau2, delta, kappa, eps = 0.70, 0.78, 0.10, 0.80, 0.02
    else:
        M, L, B = 8, 24, 64
        tau1, tau2, delta, kappa, eps = 0.75, 0.80, 0.10, 0.80, 0.02
    print(f"CONFIG M={M} L={L} B={B} tau1={tau1} tau2={tau2} delta={delta} kappa={kappa} eps={eps}")


def _construct_prover_with_backend(nli: NLIBackend, args: argparse.Namespace):
    """
    Try to import SROProver and construct it, injecting the shared NLI backend.
    If import fails, returns (None, error_string).
    """
    try:
        from sro.prover.SROProver import SROProver  # type: ignore
    except Exception as e:
        err = "".join(traceback.format_exception_only(type(e), e)).strip()
        tb = traceback.format_exc()
        logging.getLogger(__name__).error("Failed to import SROProver:\n%s", tb)
        return None, f"import_error: {err}"

    # Try to pass arguments by signature
    kwargs: Dict[str, Any] = {}
    try:
        sig = inspect.signature(SROProver)  # type: ignore
        params = sig.parameters
        if "nli_backend" in params:
            kwargs["nli_backend"] = nli
        if "profile" in params:
            kwargs["profile"] = args.profile
        if "bs_nli1" in params:
            kwargs["bs_nli1"] = args.bs_nli1
        if "bs_nli2" in params:
            kwargs["bs_nli2"] = args.bs_nli2
        if "redundancy" in params and args.redundancy is not None:
            kwargs["redundancy"] = args.redundancy
    except Exception:
        pass

    try:
        prover = SROProver(**kwargs)  # type: ignore
    except TypeError:
        prover = SROProver()  # type: ignore

    # Attach backend if ctor didn't accept it
    if not hasattr(prover, "nli_backend") or getattr(prover, "nli_backend") is None:
        try:
            setattr(prover, "nli_backend", nli)
        except Exception:
            pass

    return prover, None


def _run_on_prover(prover, args: argparse.Namespace) -> bool:
    """
    Try common entry points. Returns True if something ran.
    """
    question: Optional[str] = args.question
    for meth in ("run_question", "run", "run_cli", "answer", "answer_question"):
        if hasattr(prover, meth):
            try:
                fn = getattr(prover, meth)
                if meth in ("run_question", "answer", "answer_question") and question is not None:
                    out = fn(question)
                elif meth in ("run_cli",):
                    out = fn(args)
                else:
                    out = fn()
                # If the method returns a dict, print a friendly summary
                if isinstance(out, dict):
                    _pretty_print_result(out)
                return True
            except TypeError:
                continue
    return False


def _pretty_print_result(out: Dict[str, Any]) -> None:
    final = out.get("final_answer") or out.get("answer") or ""
    if final:
        print(f"FINAL ANSWER: {final}")
    claims = out.get("claims_accepted") or out.get("claims") or []
    if claims:
        print(f"CLAIMS ACCEPTED: {len(claims)}")
        for c in claims:
            score = c.get("score", "?")
            margin = c.get("margin", "?")
            cites = c.get("cites", [])
            cid = c.get("id") or c.get("qid") or "c?"
            print(f"  - {cid}: score={score} margin={margin} cites={cites}")
    refs = out.get("references") or out.get("refs") or {}
    if refs:
        print("REFERENCES:")
        if isinstance(refs, dict):
            for k, v in refs.items():
                print(f"  [{k}] {v}")
        elif isinstance(refs, list):
            for i, v in enumerate(refs, 1):
                print(f"  [{i}] {v}")
    rt = out.get("runtime") or out.get("runtimes") or {}
    if isinstance(rt, dict) and rt:
        rline = " ".join(f"S{k}={v}" for k, v in rt.items() if isinstance(v, str))
        if rline:
            print(f"RUNTIME {rline}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--profile", type=str, default="med", choices=["tiny", "med", "large"])
    ap.add_argument("--bs_nli1", type=int, default=32)
    ap.add_argument("--bs_nli2", type=int, default=32)
    ap.add_argument("--nli_model_dir", type=str, default=None, help="Explicit local dir for NLI model")
    ap.add_argument("--question", type=str, default=None, help="Optional single question to run")
    ap.add_argument("--redundancy", type=str, default=None, help="cosine|jaccard (if supported)")
    args = ap.parse_args()

    set_all_seeds(args.seed)
    _print_config_banner(args.profile)

    # Instantiate ONE shared NLI backend to guarantee the calibrated log line
    model_dir = args.nli_model_dir or os.environ.get("SRO_NLI_MODEL_DIR", None)
    nli = NLIBackend(model_dir=model_dir)
    logging.getLogger("sro.nli.backend").info("NLI temperature = %.3f (effective)", nli.get_temperature())

    # Try SROProver path
    prover, err = _construct_prover_with_backend(nli, args)
    if prover is not None:
        if _run_on_prover(prover, args):
            return
        else:
            logging.getLogger(__name__).warning("SROProver loaded but no known run method worked.")

    # If SROProver import failed, try a legacy module if present
    try:
        legacy = importlib.import_module("scripts.run_question_legacy")
        logging.getLogger(__name__).info("Falling back to scripts.run_question_legacy.main()")
        if hasattr(legacy, "main"):
            legacy.main()
            return
        if hasattr(legacy, "run"):
            legacy.run()
            return
    except ModuleNotFoundError:
        pass
    except Exception:
        logging.getLogger(__name__).error("Legacy runner failed:\n%s", traceback.format_exc())

    # Nothing worked; fail hard with the earlier import error if we had one
    msg = "Pipeline entry point not found. NLI calibration was applied and logged, but no runnable prover was discovered."
    if err:
        msg += f" Root cause: {err}"
    raise RuntimeError(msg)


if __name__ == "__main__":
    main()
