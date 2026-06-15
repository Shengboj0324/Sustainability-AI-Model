# Production Multimodal AI Upgrade Report

## Executive Summary

This upgrade pass moved ReleAF AI materially closer to an industrial deployable multimodal sustainability intelligence platform. The repository now has stable shared API contracts, corrected service routing, safer secret handling, deterministic multimodal orchestration for CI/staging, provenance-aware degraded RAG retrieval, and a strict data-pipeline validation/repair gate that verifies the training handoff for Vision, LLM SFT, and GNN workloads.

The most important new data-pipeline improvement is that the active Vision training configuration no longer points at the original duplicate/leaky dataset. A strict scan found 1,143 exact image hashes leaking across train/val/test in the original `data/processed/vision_cls` tree, plus corrupt images. The repaired pipeline now points `configs/vision_cls.yaml` at `data/processed/vision_cls_clean`, a non-destructive clean hardlinked view with 22,702 decode-valid images, zero exact hash leakage across splits, and a manifest for reproducibility.

The GNN training path was also hardened. The active GNN config now consumes processed parquet files by default, and `training/gnn/train_gnn.py` fails loudly if those files are invalid unless explicit rule fallback is enabled. The repaired processed graph contains 43 nodes, 118 edges, and 128-dimensional node features.

The latest full repository test run passed:

```text
120 passed, 26 warnings
```

Deployment readiness verdict: STAGING READY

This is not marked PRODUCTION READY because Docker validation found real remaining production gaps: the service image is still monolithic and large, full vector RAG startup still depends on an external embedding checkpoint/cache, the LLM service needs a real local or OpenAI-compatible backend, and full compose startup with Postgres/Redis plus all application services was not completed because host Postgres/Redis ports were already occupied.

## Current Architecture Before Changes

- API Gateway, Orchestrator, Vision, LLM, RAG, KG, Org Search, and monitoring modules existed.
- Service schemas were fragmented across gateway and service files.
- The orchestrator instantiated its workflow executor at import time before configuration was loaded.
- Orchestrator workflow action names did not match actual downstream routes.
- Compose referenced service Dockerfiles that do not exist in this repository.
- Compose launched Vision/LLM modules as `server:app` even though the real files are `server_v2.py`.
- RAG returned `503` whenever sentence-transformers or embeddings were unavailable.
- RAG imports broke when integration tests imported `services/rag_service/server.py` as a top-level module.
- Prometheus metric globals crashed under duplicate module import aliases.
- `kaggle.json` was tracked and contained a populated credential shape.
- Neo4j/Postgres service defaults included reusable development passwords.
- Vision training data pointed at an original processed dataset with duplicate/leaky exact hashes across train/val/test.
- GNN processed parquet files did not match the configured training contract.
- The local dependency set had `transformers 4.57.1` installed with incompatible `huggingface-hub 1.8.0`, breaking LLM training imports.

## Architecture After Changes

- `services/shared/schemas.py` provides import-safe Pydantic contracts for multimodal requests, final answers, citations, confidence, service metadata, warnings, errors, retrieved documents, Vision results, KG results, and LLM results.
- `services/orchestrator/engine.py` provides deterministic multimodal orchestration for CI/staging and graceful fallback testing.
- `/orchestrate` supports explicit `ORCHESTRATOR_EXECUTION_MODE=deterministic_test`, returning confidence, citations, warnings, metadata, response ID, and explicit non-production mode disclosure.
- Networked orchestrator adapters now route configured workflow actions to actual endpoints:
  - Vision: `/analyze`
  - RAG: `/retrieve`
  - KG: `/upcycling/paths`, `/relationships`, `/material/properties`
  - Org Search: `/search`
- Orchestrator service URLs now honor deployment-time environment variables such as `VISION_SERVICE_URL` and `RAG_SERVICE_URL`.
- RAG supports both legacy flat `qdrant` config and current nested `vector_store.qdrant` config.
- RAG now has real local lexical fallback retrieval with provenance and trust metadata.
- RAG Qdrant startup now uses `httpx.Limits`, bounded startup probing, and clears partial clients on failed startup so degraded lexical retrieval is explicit and testable.
- Docker Compose now points to the actual root `Dockerfile` and the actual Vision/LLM `server_v2` modules.
- Compose config requires external Postgres and Neo4j passwords.
- Prometheus metrics are idempotent for circuit breaker, Redis, and RAG modules.
- Kaggle credentials were moved out of the repo to `~/.kaggle/kaggle.json`; repo now has `kaggle.json.template` and ignores real `kaggle.json`.
- `scripts/data_pipeline_stress_validate.py` provides a strict data-pipeline audit and opt-in repair harness for dependency contracts, training imports, Vision ImageFolder splits, image decode integrity, LLM SFT JSONL files, and GNN parquet graph contracts.
- `configs/vision_cls.yaml` now points to `data/processed/vision_cls_clean`, while preserving `source_data_dir: data/processed/vision_cls` for provenance.
- `configs/gnn.yaml` now sets `use_processed_files: true` and `allow_rule_fallback: false`, making processed graph validity part of the training launch contract.
- `training/gnn/train_gnn.py` now loads processed graph and node-feature parquet files by default and exposes rule fallback only as an explicit degraded mode.

## Files Modified

- `.dockerignore`
- `.gitignore`
- `Dockerfile`
- `configs/gnn.yaml`
- `configs/vision_cls.yaml`
- `docker-compose.yml`
- `pyproject.toml`
- `requirements.txt`
- `requirements-arm.txt`
- `scripts/init_databases.py`
- `scripts/train_unified_pipeline.py`
- `services/common/circuit_breaker.py`
- `services/common/health_checks.py`
- `services/common/redis_cache.py`
- `services/kg_service/server.py`
- `services/llm_service/server_v2.py`
- `services/orchestrator/main.py`
- `services/orchestrator/engine.py`
- `services/org_search_service/server.py`
- `services/rag_service/server.py`
- `tests/integration/test_rag_production.py`
- `tests/test_reasoning_system.py`
- `tests/unit/test_training_notebook_contract.py`
- `tests/unit/test_llm_service_backend_modes.py`
- `tests/unit/test_multimodal_orchestrator_engine.py`
- `tests/unit/test_rag_service.py`
- `training/gnn/train_gnn.py`
- `Sustainability_AI_Model_Training.ipynb`

## New Files Added

- `services/shared/schemas.py`
- `services/orchestrator/engine.py`
- `scripts/industrial_e2e_validate.py`
- `scripts/stress_new_multimodal_paths.py`
- `scripts/data_pipeline_stress_validate.py`
- `scripts/validate_training_notebook.py`
- `tests/unit/test_multimodal_orchestrator_engine.py`
- `tests/unit/test_data_pipeline_stress_validator.py`
- `tests/unit/test_training_notebook_contract.py`
- `kaggle.json.template`
- `.dockerignore`
- `PRODUCTION_MULTIMODAL_AI_UPGRADE_REPORT.md`
- `outputs/stress/new_multimodal_paths_stress_report.json`
- `outputs/stress/new_multimodal_paths_stress_stdout.txt`
- `outputs/data_pipeline/data_pipeline_stress_report.json`
- `data/processed/vision_cls_clean/manifest.json`

## Generated Or Repaired Data Artifacts

- `data/processed/vision_cls_clean`: clean hardlinked Vision ImageFolder view used by active training config.
- `data/processed/gnn/graph.parquet`: regenerated processed graph contract with 118 edges.
- `data/processed/gnn/node_features.parquet`: regenerated node table with 43 nodes and 128 feature columns.

The original `data/processed/vision_cls` tree was not deleted. It is retained as the source/provenance dataset, but it is no longer the active training input because it contains exact duplicate leakage across splits and corrupt images.

## Major Bugs Or Gaps Discovered

- Compose deployment was broken by missing per-service Dockerfiles.
- Compose service commands referenced missing Vision/LLM modules.
- Orchestrator service URLs were not overridden from container environment variables.
- RAG could not serve degraded retrieval despite having local corpora available.
- Full pytest collection initially failed due RAG import style and duplicate Prometheus collectors.
- A tracked `kaggle.json` contained credential fields with a populated key.
- SFT validation rejected valid chat data with a system prompt.
- First stress run found that bounded long RAG queries were rejected by Pydantic with `422` before the handler could truncate and audit them.
- First stress run also exposed a validation-harness false positive: parsing any occurrence of the word `failed` misclassified `0 failed` validator summaries as failed.
- Lexical fallback scoring recomputed document-frequency statistics per query, which was correct but unnecessarily expensive under stress.
- Compose rendered successfully, but emitted an obsolete top-level `version` warning.
- Local `transformers 4.57.1` was incompatible with `huggingface-hub 1.8.0`, breaking `training.llm.train_sft` imports.
- Original Vision processed data had 1,143 exact hashes leaking across train/val/test, 3,165 duplicate hashes, and 6,880 duplicate files.
- First repaired Vision split exposed 3 corrupt image files during full decode; the repair harness was tightened to skip undecodable source files and rebuild.
- Original GNN parquet files had only 20 nodes, 12 edges, and 2 feature columns, missing the configured `node_type`/`name`/128-feature contract.
- RAG Qdrant startup used a raw dict for `limits`, causing `'dict' object has no attribute 'max_connections'` after dependency repair allowed deeper startup execution.
- Docker build initially sent an enormous context because no `.dockerignore` excluded `data/`, `deployment_package/`, checkpoints, `.git`, and generated artifacts.
- Docker packaging failed because `pyproject.toml` listed namespace directories as regular packages.
- Docker dependency resolution initially selected future/large incompatible dependency sets, including CUDA-oriented Torch packages on Linux/arm64 risk paths.
- Docker container imports initially failed on OpenCV because the runtime image lacked `libGL.so.1`.
- Docker container imports initially failed for Vision because model source was accidentally excluded along with model checkpoints/artifacts.
- Qdrant Compose healthcheck was invalid for `qdrant/qdrant:v1.8.0`: `/health` returns 404 and the image does not include `curl`.
- Deterministic orchestrator overclaimed medium confidence and citations for meaningless input such as `???`.
- LLM service startup tried local model loading before explicit API/degraded operation and could trigger heavy/gated model behavior during container startup.
- LLM and KG detailed health endpoints serialized dataclass health results as Python repr strings instead of JSON.
- RAG degraded startup did not mark startup complete, leaving startup probes in `starting` even when fallback retrieval was being served.
- Shared health aggregation could report healthy for a service that was explicitly not ready and had no dependency checks.
- The root training notebook had become a divergent training system: embedded package uninstall/install cells, Kaggle credential-writing logic, hard-coded legacy data ingestion paths, copied Vision/GNN training loops, April historical outputs, and stale readiness claims.
- `scripts/train_unified_pipeline.py --dry-run` still required Hugging Face authentication for LLM, which made local data/config validation depend on external credentials.

## Fixes Implemented

- Added stable shared schemas for API/frontend service contracts.
- Added `.dockerignore` to reduce Docker build context from gigabytes to megabytes by excluding datasets, generated deployment packages, checkpoints, `.git`, and local secrets while preserving importable model source.
- Fixed Dockerfile packaging by copying importable source before editable install and adding required OpenCV runtime libraries.
- Converted setuptools config to namespace package discovery for `services*` and `training*`.
- Bounded major dependency versions in `pyproject.toml` and added `pycocotools` for Linux data/vision validation.
- Fixed Qdrant Compose healthcheck to use a real `/collections` probe via Bash TCP instead of missing `curl` and nonexistent `/health`.
- Added deterministic multimodal orchestration for:
  - Text-only QA.
  - Image-only item/bin classification.
  - Multimodal image + text bin decisions.
  - Safety checks.
  - Upcycling recommendations.
  - Organization search.
  - Corrupted image handling.
- Added explicit `deterministic_test` metadata so fallback execution never pretends to be production model inference.
- Tightened deterministic orchestrator low-evidence handling so nonsensical text-only input returns very low confidence, no citations, and a `LOW_TEXT_QUALITY` warning.
- Added explicit LLM backend controls: `LLM_BACKEND=api/openai/openai_compatible` and `LLM_DISABLE_LOCAL=true`.
- Fixed LLM readiness so a service with no local/API backend is alive but not ready, and generation returns `503`.
- Added JSON-safe serialization helpers to the shared `HealthCheckResult` dataclass.
- Fixed shared health aggregation so not-ready/no-check services cannot report healthy while dependency-specific failures still surface.
- Fixed RAG degraded startup so startup probes reach a completed state while readiness remains not-ready.
- Implemented RAG local lexical fallback over checked-in corpora, including:
  - JSONL and nested JSON corpus loading.
  - BM25-like scoring.
  - matched term metadata.
  - lineage metadata.
  - trust indicators.
  - embedding metadata indicating `local-lexical-bm25-fallback`.
  - degraded-mode endpoint metadata.
- Fixed RAG endpoint so degraded retrieval returns structured results instead of `503`.
- Fixed RAG Qdrant client construction and failed-startup cleanup.
- Fixed Docker Compose build/module wiring.
- Added deployment URL overrides to the orchestrator.
- Fixed RAG config compatibility and embedding timeout behavior.
- Made Prometheus collectors resilient to duplicate imports.
- Removed tracked Kaggle credential and replaced it with a template.
- Required database passwords via environment variables.
- Added `scripts/industrial_e2e_validate.py`, a dependency-light industrial validation gate runnable in a plain Linux Python container.
- Added `scripts/stress_new_multimodal_paths.py`, a deterministic stress harness for the newly hardened RAG/orchestrator/deployment paths.
- Added `scripts/data_pipeline_stress_validate.py`, a strict data pipeline validator and repair tool.
- Increased the RAG request model's bounded query length to 4,096 so large but acceptable requests reach the endpoint's truncation and audit logic instead of failing pre-handler with `422`.
- Precomputed lexical fallback document frequencies at corpus-load time to keep repeated degraded retrieval stress runs stable.
- Fixed the stress harness validator parsing so `SUMMARY: ... 0 failed` is accepted only when the explicit zero-failure marker is present.
- Removed the obsolete Docker Compose top-level `version` field.
- Pinned `huggingface-hub>=0.34.0,<1.0` in Python dependency files and repaired the local environment to `huggingface-hub 0.36.2`.
- Rebuilt Vision training splits into a clean hardlinked view with zero exact split leakage and zero decode failures under full scan.
- Updated Vision config to use the clean data view and preserve source dataset provenance.
- Regenerated GNN processed graph and node feature parquet files to match the configured training contract.
- Updated GNN training data loading so processed files are the default path and rule fallback is explicit.
- Rebuilt `Sustainability_AI_Model_Training.ipynb` into a thin 12-cell industrial runbook over maintained repository scripts.
- Removed notebook-side package mutation, Kaggle credential writes, copied training loops, stale outputs, and historical accuracy/readiness output.
- Added notebook cells for active config inspection, strict data-pipeline validation, repository dataset validators, GNN handoff, LLM SFT dummy-tokenizer handoff, unified-pipeline dry-run, and explicit launch controls.
- Updated `scripts/train_unified_pipeline.py` so strict data-pipeline validation runs by default before any training stage.
- Added `--skip-data-validation` and `--full-data-scan` controls to the unified training launcher.
- Changed LLM dry-run behavior so external Hugging Face auth is required only for real LLM training launches, not for local dry-run validation.
- Added a notebook contract validator that fails on stale/unsafe snippets, uncleared outputs, execution counts, missing current validators, or missing clean data references.

## Tests Added

- `test_request_and_task_classification_text_only`
- `test_request_and_task_classification_multimodal_safety`
- `test_text_only_flow_preserves_rag_citations_and_schema`
- `test_image_only_flow_returns_structured_vision_metadata`
- `test_multimodal_safety_flow_avoids_unsafe_disposal_claim`
- `test_upcycling_flow_uses_kg_relationships`
- `test_org_search_discloses_missing_location`
- `test_low_quality_text_only_request_does_not_invent_evidence`
- `test_corrupted_image_is_honest_low_confidence_warning`
- `test_fastapi_orchestrate_endpoint_uses_deterministic_contract`
- `test_orchestrator_applies_service_url_env_overrides`
- `test_lexical_retrieval_fallback_returns_provenance`
- `test_retrieve_endpoint_uses_lexical_fallback_when_models_unloaded`
- `test_retrieve_endpoint_truncates_long_queries`
- `test_failed_qdrant_connection_clears_client`
- `test_gnn_table_generator_matches_training_contract`
- `test_gnn_training_loader_reads_processed_parquet`
- `test_image_decode_rejects_corrupt_file`
- `test_training_notebook_contract_is_current`
- `test_llm_disable_local_starts_degraded_without_backend`
- `test_llm_explicit_api_backend_skips_local_model`
- `test_health_check_result_serializes_json_safe_enums`
- `test_not_ready_health_checker_does_not_report_healthy_without_checks`

## Commands Run

- `rg --files`
- `find . -maxdepth 3 ...`
- `rg -n "TODO|FIXME|placeholder|dummy|mock|degraded|hardcoded|..."`
- `python -m py_compile ...`
- `python -m compileall -q services scripts tests`
- `python -m compileall -q services scripts training tests`
- `pytest tests/unit/test_rag_service.py -q --no-cov`
- `pytest tests/unit/test_multimodal_orchestrator_engine.py -q --no-cov`
- `pytest tests/test_reasoning_system.py tests/unit/test_rag_service.py tests/unit/test_multimodal_orchestrator_engine.py -q --no-cov`
- `pytest tests/unit/test_data_pipeline_stress_validator.py -q --no-cov`
- `pytest tests/integration/test_rag_production.py -q --no-cov`
- `pytest -q --no-cov`
- `python scripts/industrial_e2e_validate.py`
- `python scripts/industrial_e2e_validate.py --runtime`
- `python scripts/stress_new_multimodal_paths.py`
- `python scripts/data_pipeline_stress_validate.py --sample-size 200`
- `python scripts/data_pipeline_stress_validate.py --vision-source-dir data/processed/vision_cls --repair-vision-splits --update-vision-config --repair-gnn --full-image-scan`
- `python scripts/data_pipeline_stress_validate.py --full-image-scan`
- `python scripts/validate_training_notebook.py`
- `python scripts/validate_vision_data.py`
- `python scripts/validate_all_datasets.py`
- `python scripts/train_unified_pipeline.py --dry-run --skip-preflight`
- Targeted GNN training handoff check for `load_graph_data` and `create_train_val_test_split`.
- Targeted Vision dataloader handoff check for classifier dataset creation and one batch.
- Targeted LLM dummy-tokenizer handoff check for configured SFT JSONL files.
- `python -m pip install 'huggingface-hub>=0.34.0,<1.0'`
- `POSTGRES_PASSWORD=change-me-for-validation NEO4J_PASSWORD=change-me-for-validation docker compose config --quiet`
- `POSTGRES_PASSWORD=stress_postgres_password NEO4J_PASSWORD=stress_neo4j_password docker compose config --quiet`
- `POSTGRES_PASSWORD=data_pipeline_postgres NEO4J_PASSWORD=data_pipeline_neo4j docker compose config --quiet`
- `python -m json.tool Sustainability_AI_Model_Training.ipynb`
- `docker context show`
- `docker info`
- `docker compose build`
- `docker compose build orchestrator`
- `docker compose build llm-service orchestrator`
- `docker compose up -d qdrant neo4j`
- `docker run --rm sustainability-ai-model-orchestrator:latest python -m compileall -q services scripts training models`
- `docker run --rm -i sustainability-ai-model-orchestrator:latest python - <<'PY' ... core import smoke ... PY`
- `docker run --rm -v "$PWD":/app -w /app python:3.11-slim python scripts/industrial_e2e_validate.py`
- Dockerized `scripts/data_pipeline_stress_validate.py --full-image-scan`
- Dockerized deterministic orchestrator `/orchestrate` text/image/multimodal/low-evidence runtime checks.
- Dockerized deterministic orchestrator stress run: 140 requests, concurrency 16.
- Dockerized LLM degraded-mode health/readiness/generate checks with updated source bind-mounted.
- Dockerized KG service readiness and material endpoint check against Docker Neo4j.
- Dockerized RAG offline degraded startup and local lexical fallback retrieval check with updated source bind-mounted.
- Secret/default scan with `rg` for cloud keys, private keys, API keys, and reusable password defaults.
- Checkpoint metadata inspection with `torch.load(..., map_location="cpu")`.

## Test Results

- Full suite: 120 passed, 26 warnings.
- Training notebook contract: passed.
  - Notebook JSON is valid.
  - Notebook has 12 cells: 9 code and 3 markdown.
  - Notebook has 0 outputs and 0 execution counts.
  - Notebook references current validation/launch scripts and `data/processed/vision_cls_clean`.
  - Notebook no longer contains Kaggle credential-writing snippets, copied training loops, historical checkpoint output, or stale readiness claims.
- Unified training launcher dry-run: passed for Vision, GNN, and LLM after running data-pipeline validation.
- Data pipeline stress validator: 7 passed, 0 failed, 0 warnings.
  - Dependencies: `transformers 4.57.1`, `huggingface-hub 0.36.2`, `datasets 4.1.1`, `torch 2.2.0`, `torchvision 0.17.0`, `pillow 11.3.0`, `pandas 2.0.3`, `pyarrow 21.0.0`.
  - Training imports: all configured Vision, LLM, and GNN training modules import cleanly.
  - Vision active clean split: train 18,147; val 2,259; test 2,296.
  - Vision exact leakage: 0 cross-split hashes, 0 duplicate hashes, 0 duplicate files in the clean view.
  - Vision decode full scan: 22,702 scanned, 0 failures.
  - LLM SFT configured JSONL: 6,848 examples across generated train, expert train, and generated validation files.
  - GNN parquet contract: 43 nodes, 118 edges, 128 feature columns.
- Vision data validator: passed all 8 checks.
  - Dataloader initialized.
  - Forward pass succeeded.
  - 3 loss/backward steps succeeded.
  - 4,096-dimensional LLM embedding check succeeded.
  - Random 500 image integrity scan found 0 errors.
- GNN training handoff check: loaded `(43, 128)` node features and `(2, 118)` edge index; split masks produced 94 train, 11 validation, and 13 test edges.
- Vision classifier handoff check: 18,147 train images, 2,259 validation images, 30 classes, batch shape `(2, 3, 448, 448)`, 0 skipped pre-validation samples.
- LLM SFT handoff check: dummy tokenizer processed 6,164 training and 684 validation examples with `input_ids`, `attention_mask`, and `labels`.
- RAG integration suite: 11 passed, 4 warnings.
- RAG unit suite: 10 passed, 4 warnings.
- LLM backend/health unit suite: 4 passed, 4 warnings.
- New stress harness: 5 passed, 0 failed, 394 total checks.
  - `rag_lexical_direct_concurrency`: 90 checks, 450 documents validated.
  - `orchestrator_engine_concurrency_and_malformed_inputs`: 161 responses validated.
  - `rag_endpoint_degraded_envelope`: 80 endpoint requests validated.
  - `orchestrator_fastapi_envelope_stress`: 60 endpoint responses validated.
  - `deployment_validation_commands`: compose config plus static/runtime industrial validators passed.
- Industrial static validator: 8 passed, 0 failed.
- Industrial runtime validator: 9 passed, 0 failed.
- Linux container static validator: 8 passed, 0 failed in `python:3.11-slim`.
- Compose config render: passed with required external password env vars.
- Docker build: passed for all service images after `.dockerignore`, Dockerfile, packaging, dependency, OpenCV, and model-source fixes.
- Docker image size: each service image is still about 3.74GB, which is functional but not production-optimized.
- Docker import smoke: passed for orchestrator, RAG, KG, org search, Vision, LLM, Vision training, LLM SFT training, and GNN training modules.
- Docker compile-all: passed for `services`, `scripts`, `training`, and `models`.
- Docker infrastructure:
  - Qdrant and Neo4j started under Compose.
  - Qdrant healthcheck fixed and Docker reported `healthy`.
  - Neo4j healthcheck reported `healthy`.
- Docker orchestrator runtime:
  - `/health/ready` returned ready.
  - Text-only, image-only, multimodal, and low-evidence `/orchestrate` cases returned stable schemas.
  - Low-evidence `???` case returned `confidence_score: 0.18`, `confidence_level: very_low`, 0 sources, 0 citations, and `LOW_TEXT_QUALITY`.
  - Stress: 140 requests at concurrency 16, 140 successes, 0 failures, p50 4.46ms, p95 29.34ms, max 32.01ms in deterministic mode.
- Docker LLM degraded runtime:
  - `LLM_DISABLE_LOCAL=true` skipped local model loading.
  - `/health/startup` reached started.
  - `/health/ready` returned not-ready.
  - `/health` returned 503 JSON with `No LLM backend available`.
  - `/generate` returned 503 with explicit backend configuration guidance.
- Docker KG runtime:
  - Connected to Docker Neo4j.
  - `/health/ready` returned ready.
  - `/health` returned JSON healthy with Neo4j dependency details after shared health serialization fix.
  - `/material/properties` accepted `material_name` and returned a stable `KGResponse`.
- Docker RAG degraded runtime:
  - Offline embedding model startup failed honestly.
  - `/health/startup` now reaches started in degraded mode.
  - `/health/ready` remains not-ready.
  - `/health` returns 503 JSON with `startup_complete: true`.
  - `/retrieve` returns structured local lexical fallback documents with `doc_id`, `source`, provenance, lineage, trust indicators, and degraded metadata.
- Py compile check: passed for touched modules.
- Compile-all check: passed for `services`, `scripts`, `training`, and `tests`.
- Model artifact gate: passed.
  - Vision checkpoint present.
  - Physics-informed vision checkpoint present.
  - GNN checkpoint present.
  - deployment metadata reports 30 item classes and 93.20% validation accuracy.
- RAG degraded-mode runtime evidence:
  - Loaded 6,839 local lexical RAG documents.
  - `/retrieve` returned structured provenance-aware degraded results instead of `503`.

Warnings remaining:

- FastAPI `on_event` deprecation warnings.
- Pytest collection warnings for helper classes named `Test*` with constructors.
- Local environment warnings: Python 3.9, x86/Rosetta on ARM Mac.
- RAG reranker configuration is still incomplete, so reranker loading is skipped.
- RAG startup can still reload the embedding model on repeated startup attempts, which is inefficient under repeated degraded-mode tests.

## Docker And Linux Validation

Docker Desktop was available on the final pass:

```text
docker context: desktop-linux
Docker Compose: v2.39.2-desktop.1
Docker Engine: 28.3.3
Daemon platform: linux/arm64
```

Docker validation found and fixed concrete container defects:

- Build context reduced from gigabyte-scale to megabyte-scale with `.dockerignore`.
- Root Dockerfile now builds installable package source.
- Runtime image now has OpenCV system libraries.
- Python dependency versions are bounded enough to avoid future CUDA-heavy resolver paths on Linux/arm64.
- Qdrant healthcheck is now valid for the actual Qdrant image.
- Core imports and compile checks pass inside Linux containers.
- Data pipeline full image scan passes inside Docker: 22,702 images scanned, 0 decode failures, 0 exact split leakage.

Important caveat: final LLM/RAG/shared-health fixes were verified in Docker using read-only bind mounts over the rebuilt images. The monolithic image rebuild passed immediately before those last source-level health fixes, but the final health serialization/RAG degraded-startup changes still need one more full image bake in CI or a Linux runner before release tagging.

## Remaining Blockers

- Full live docker-compose startup with Postgres, Redis, and all application services was not completed because host Postgres/Redis ports were already occupied.
- Production RAG vector retrieval has not been validated against a live embedding model plus populated Qdrant collection in this environment.
- Neo4j/KG connectivity was validated against Docker Neo4j, but the graph was not seeded with production material/upcycling data during this pass.
- Vision checkpoint inference has not been benchmarked in a Linux container during this pass.
- LLM production backend has not been exercised with a live local or OpenAI-compatible model; degraded/no-backend behavior was validated.
- Final health/RAG source fixes need to be baked into Docker images after the bind-mounted verification.
- Service images are still monolithic and about 3.74GB each.
- Dockerfile layer layout still invalidates dependency installation on service-code edits, causing slow rebuilds.
- Full real Vision, LLM, and GNN training runs were not launched; the validated scope is strict data handoff, imports, schema, decoding, loader, tokenization, and split construction.
- Kubernetes manifests have not been applied to a cluster.
- FastAPI lifespan migration remains outstanding.
- The clean Vision dataset is a local generated hardlinked view. It must either be persisted as an artifact or regenerated with `scripts/data_pipeline_stress_validate.py` in any fresh environment before training.

## Deployment Readiness Verdict

STAGING READY

The repository now has enough validated structure and capability for industrial staging: contracts are stable, the full suite passes, Docker images build, core Linux container import/compile/runtime checks pass, secrets are externalized, model artifacts are present, RAG has a real provenance-aware degraded retrieval path, and the active training data contracts for Vision, LLM SFT, and GNN pass strict validation.

It is not PRODUCTION READY until final source fixes are baked into images, full compose can run without host port conflicts, real RAG vector retrieval is validated with a populated Qdrant collection, LLM is exercised against a real backend, Vision checkpoint inference is benchmarked in Linux, and production KG data is seeded and validated.

## Next Recommended Engineering Milestones

1. Rebuild all Docker images after the final health/RAG source fixes and rerun the Docker import/compile/runtime smoke suite.
2. Resolve host Postgres/Redis port conflicts or run Compose on an isolated Linux runner, then execute `POSTGRES_PASSWORD=... NEO4J_PASSWORD=... docker compose up --build` and validate `/health`, `/ready`, `/metrics`, `/orchestrate`, and `/retrieve`.
3. Validate Qdrant vector retrieval by ingesting a small corpus and asserting citation provenance.
4. Validate Neo4j/KG routes against seeded material/upcycling/hazard graph data.
5. Run real Vision checkpoint inference and latency tests in Linux.
6. Launch short bounded training smoke runs for Vision, LLM SFT, and GNN using the validated active data contracts.
7. Exercise LLM service with a live backend and verify prompt-grounded citation behavior.
8. Migrate FastAPI `on_event` hooks to lifespan handlers.
9. Promote the data-pipeline and multimodal stress harnesses into CI and run them against live containers once Docker/Linux execution is available.
