#!/usr/bin/env python3
"""
Unified Training Pipeline Orchestrator
=======================================
Coordinates Vision → GNN → LLM training with data dependency handling.

Usage:
    python scripts/train_unified_pipeline.py                 # all stages
    python scripts/train_unified_pipeline.py --stage vision  # single stage
    python scripts/train_unified_pipeline.py --stage gnn llm # multiple
    python scripts/train_unified_pipeline.py --dry-run       # validate only
"""
import argparse, json, logging, os, subprocess, sys, time, shutil, yaml
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("unified_pipeline")

@dataclass
class StageResult:
    name: str
    success: bool
    duration_s: float
    message: str = ""

def _run_script(script: str, env_extra: Dict = None) -> int:
    cmd = [sys.executable, str(PROJECT_ROOT / script)]
    env = {**os.environ, **(env_extra or {}), "PYTORCH_ENABLE_MPS_FALLBACK": "1"}
    logger.info(f"  ▸ Running: {' '.join(cmd)}")
    return subprocess.run(cmd, cwd=str(PROJECT_ROOT), env=env).returncode

def _check_files(files: List[str]) -> bool:
    missing = [f for f in files if not (PROJECT_ROOT / f).exists()]
    if missing:
        logger.error(f"  Missing: {missing}")
        return False
    return True

# ── Inter-stage dependency validation ─────────────────────────────────
def _validate_stage_dependencies(stage: str) -> Optional[str]:
    """
    Validate that upstream stage artifacts exist before starting a stage.
    Returns an error message if validation fails, None if OK.

    Data dependency chain:
      Vision → produces checkpoint → GNN can use vision embeddings
      GNN    → produces graph checkpoint → LLM SFT data may reference graph
      LLM    → requires SFT data + HF auth
    """
    if stage == "gnn":
        # GNN benefits from vision checkpoint but doesn't strictly require it
        vision_cfg = PROJECT_ROOT / "configs" / "vision_cls.yaml"
        if vision_cfg.exists():
            import yaml
            with open(vision_cfg) as f:
                vcfg = yaml.safe_load(f)
            vision_out = PROJECT_ROOT / vcfg.get("training", {}).get("output_dir", "models/vision/classifier")
            if not vision_out.exists():
                logger.warning("  ⚠️  Vision output dir not found — GNN will train without vision features")
    elif stage == "llm":
        # LLM strictly requires SFT data
        required = [
            "data/processed/llm_sft/generated_sft_train.jsonl",
            "data/processed/llm_sft/generated_sft_val.jsonl",
        ]
        missing = [f for f in required if not (PROJECT_ROOT / f).exists()]
        if missing:
            return f"LLM SFT data missing: {missing}. Run generate_sft_dataset.py first."
        # Verify HuggingFace authentication
        try:
            from huggingface_hub import HfApi
            HfApi().whoami()
        except Exception:
            return "HuggingFace not authenticated — run: huggingface-cli login"
    return None


# ── Stage 1: Vision ──────────────────────────────────────────────────
def run_vision_stage(dry_run: bool = False) -> StageResult:
    logger.info("\n" + "=" * 70)
    logger.info("STAGE 1: VISION CLASSIFIER TRAINING")
    logger.info("  Upgrades: SE blocks, LLM projection (4096d), temperature calibration")
    logger.info("=" * 70)
    if not (PROJECT_ROOT / "configs" / "vision_cls.yaml").exists():
        return StageResult("vision", False, 0, "configs/vision_cls.yaml not found")
    if dry_run:
        logger.info("  [DRY-RUN] Would run: training/vision/train_classifier.py")
        logger.info("  [DRY-RUN] Multi-head: SE + LLM projection enabled via config")
        return StageResult("vision", True, 0, "dry-run")
    t0 = time.time()
    rc = _run_script("training/vision/train_classifier.py")
    return StageResult("vision", rc == 0, time.time() - t0, f"exit={rc}")

# ── Stage 2: GNN ─────────────────────────────────────────────────────
def run_gnn_stage(dry_run: bool = False) -> StageResult:
    logger.info("\n" + "=" * 70)
    logger.info("STAGE 2: GNN (GATv2) TRAINING")
    logger.info("  Upgrades: GATv2 dynamic attention, edge attrs, L2-norm embeddings")
    logger.info("=" * 70)
    if not (PROJECT_ROOT / "configs" / "gnn.yaml").exists():
        return StageResult("gnn", False, 0, "configs/gnn.yaml not found")
    dep_err = _validate_stage_dependencies("gnn")
    if dep_err:
        return StageResult("gnn", False, 0, dep_err)
    if dry_run:
        logger.info("  [DRY-RUN] Would run: training/gnn/train_gnn.py")
        logger.info("  [DRY-RUN] GATv2Model with edge_dim support")
        return StageResult("gnn", True, 0, "dry-run")
    t0 = time.time()
    rc = _run_script("training/gnn/train_gnn.py", env_extra={"PYTORCH_ENABLE_MPS_FALLBACK": "1"})
    return StageResult("gnn", rc == 0, time.time() - t0, f"exit={rc}")

# ── Stage 3: LLM LoRA SFT ────────────────────────────────────────────
def run_llm_stage(dry_run: bool = False) -> StageResult:
    logger.info("\n" + "=" * 70)
    logger.info("STAGE 3: LLM LoRA SFT (Llama-3-8B)")
    logger.info("  Upgrades: LoRA r=64, dtype-aware loading, MPS memory cleanup")
    logger.info("=" * 70)
    if not (PROJECT_ROOT / "configs" / "llm_sft.yaml").exists():
        return StageResult("llm", False, 0, "configs/llm_sft.yaml not found")
    dep_err = _validate_stage_dependencies("llm")
    if dep_err:
        return StageResult("llm", False, 0, dep_err)
    if dry_run:
        logger.info("  [DRY-RUN] Would run: training/llm/train_sft.py")
        return StageResult("llm", True, 0, "dry-run")
    t0 = time.time()
    rc = _run_script("training/llm/train_sft.py")
    return StageResult("llm", rc == 0, time.time() - t0, f"exit={rc}")

# ── Pre-flight ────────────────────────────────────────────────────────
def preflight_check() -> bool:
    import torch
    logger.info("\n" + "=" * 70 + "\nPRE-FLIGHT CHECK\n" + "=" * 70)
    ok = True
    dev = "CUDA" if torch.cuda.is_available() else ("MPS" if torch.backends.mps.is_available() else "CPU")
    logger.info(f"  Device: {dev}")
    try:
        import psutil
        mem = psutil.virtual_memory()
        logger.info(f"  RAM: {mem.total/1e9:.1f}GB total, {mem.available/1e9:.1f}GB free")
        if mem.available < 16e9:
            logger.warning("  ⚠️  <16GB free — LLM may OOM")
    except ImportError:
        pass
    disk = shutil.disk_usage(str(PROJECT_ROOT))
    logger.info(f"  Disk free: {disk.free/1e9:.0f}GB")
    if disk.free < 20e9:
        logger.warning("  ⚠️  <20GB disk free")
    for pkg in ["torch", "transformers", "peft", "datasets", "torch_geometric", "timm"]:
        try:
            mod = __import__(pkg); logger.info(f"  {pkg:20s} {getattr(mod, '__version__', '?')}")
        except ImportError:
            logger.error(f"  ❌ {pkg} NOT INSTALLED"); ok = False
    return ok

STAGE_MAP = {"vision": run_vision_stage, "gnn": run_gnn_stage, "llm": run_llm_stage}

def main():
    parser = argparse.ArgumentParser(description="ReLEAF AI — Unified Training Pipeline")
    parser.add_argument("--stage", nargs="*", choices=list(STAGE_MAP.keys()))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-preflight", action="store_true")
    args = parser.parse_args()
    stages = args.stage or list(STAGE_MAP.keys())

    logger.info("╔══════════════════════════════════════════════════════════════╗")
    logger.info("║     ReLEAF AI — UNIFIED TRAINING PIPELINE                  ║")
    logger.info(f"║     Stages: {', '.join(stages):46s} ║")
    logger.info(f"║     Mode:   {'DRY-RUN' if args.dry_run else 'TRAINING':46s} ║")
    logger.info("╚══════════════════════════════════════════════════════════════╝")

    if not args.skip_preflight and not preflight_check():
        logger.error("Pre-flight FAILED"); sys.exit(1)

    results, t0 = [], time.time()
    for name in stages:
        r = STAGE_MAP[name](dry_run=args.dry_run); results.append(r)
        if not r.success:
            logger.error(f"\n❌ '{name}' FAILED: {r.message}"); break

    dur = time.time() - t0
    logger.info("\n" + "=" * 70 + "\nPIPELINE SUMMARY\n" + "=" * 70)
    for r in results:
        d = f"{r.duration_s:.0f}s" if r.duration_s else "—"
        logger.info(f"  {'✅' if r.success else '❌'} {r.name:10s} {d:>8s}  {r.message}")
    ok = all(r.success for r in results)
    logger.info(f"\n{'✅ ALL STAGES PASSED' if ok else '❌ PIPELINE FAILED'} ({dur:.0f}s)")

    log_dir = PROJECT_ROOT / "logs"; log_dir.mkdir(exist_ok=True)
    log_path = log_dir / f"unified_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(log_path, "w") as f:
        json.dump({"ts": datetime.now().isoformat(), "stages": stages, "dry_run": args.dry_run,
                    "duration_s": dur, "results": [{"name": r.name, "ok": r.success,
                    "duration_s": r.duration_s, "msg": r.message} for r in results]}, f, indent=2)
    logger.info(f"Log: {log_path}")
    sys.exit(0 if ok else 1)

if __name__ == "__main__":
    main()
