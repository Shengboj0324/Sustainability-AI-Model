#!/usr/bin/env python3
"""R2 mirroring utilities for vetted 3D vision datasets.

This script is intentionally conservative:

- it never uploads unless --execute is provided;
- it requires R2 credentials from environment variables;
- it writes manifest records for every planned/uploaded object;
- it does not bypass dataset license/access gates.

Supported immediate operations:

- preflight: verify config and credentials;
- objectron-plan: build a URL-to-R2 plan from public Objectron index files;
- upload-url-list: stream a JSONL URL plan into R2;
- upload-local: upload a previously downloaded local dataset tree into R2.

ARKitScenes should be downloaded with Apple's official downloader first, then
uploaded with upload-local after license/terms are confirmed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import quote

import requests
import yaml
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "vision_3d_datasets.yaml"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "r2_3d_ingestion"
OBJECTRON_PUBLIC_BASE = "https://storage.googleapis.com/objectron"


@dataclass
class MirrorObject:
    dataset: str
    source_url: str
    r2_key: str
    asset_type: str
    license_summary: str
    status: str = "planned"
    bytes: int | None = None
    sha256: str | None = None
    error: str | None = None


class HashingStream:
    def __init__(self, raw: Any) -> None:
        self.raw = raw
        self.digest = hashlib.sha256()
        self.bytes_read = 0

    def read(self, size: int = -1) -> bytes:
        chunk = self.raw.read(size)
        if chunk:
            self.digest.update(chunk)
            self.bytes_read += len(chunk)
        return chunk


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    for key in ("r2", "datasets"):
        if key not in config:
            raise ValueError(f"Missing required config section: {key}")
    return config


def r2_env(config: dict[str, Any]) -> dict[str, str]:
    required = config["r2"].get("required_env", ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY"])
    missing = [name for name in required if not os.getenv(name)]
    if missing:
        raise RuntimeError(
            "Missing R2 credentials: "
            + ", ".join(missing)
            + ". Bucket URL/account ID alone cannot authorize uploads."
        )
    return {
        "aws_access_key_id": os.environ["AWS_ACCESS_KEY_ID"],
        "aws_secret_access_key": os.environ["AWS_SECRET_ACCESS_KEY"],
        "aws_session_token": os.getenv("AWS_SESSION_TOKEN") or "",
        "region_name": os.getenv("AWS_DEFAULT_REGION") or config["r2"].get("default_region", "auto"),
    }


def make_s3_client(config: dict[str, Any]):
    try:
        import boto3
        from botocore.config import Config
    except Exception as exc:  # pragma: no cover - dependency gate
        raise RuntimeError("boto3 is required for R2 upload operations") from exc

    env = r2_env(config)
    session_kwargs = {
        "aws_access_key_id": env["aws_access_key_id"],
        "aws_secret_access_key": env["aws_secret_access_key"],
        "region_name": env["region_name"],
    }
    if env["aws_session_token"]:
        session_kwargs["aws_session_token"] = env["aws_session_token"]

    return boto3.client(
        "s3",
        endpoint_url=config["r2"]["endpoint_url"],
        config=Config(signature_version="s3v4", retries={"max_attempts": 5, "mode": "standard"}),
        **session_kwargs,
    )


def write_jsonl(path: Path, rows: Iterable[MirrorObject]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), sort_keys=True) + "\n")


def read_jsonl(path: Path) -> list[MirrorObject]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            try:
                rows.append(MirrorObject(**payload))
            except TypeError as exc:
                raise ValueError(f"Invalid mirror row at {path}:{line_no}: {exc}") from exc
    return rows


def http_text(url: str, timeout: float) -> str:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.text


def objectron_index_url(class_name: str, split: str) -> str:
    suffix = "annotations_train" if split == "train" else "annotations_test"
    return f"{OBJECTRON_PUBLIC_BASE}/v1/index/{quote(class_name)}_{suffix}"


def build_objectron_plan(
    config: dict[str, Any],
    classes: list[str],
    splits: list[str],
    include_videos: bool,
    include_records: bool,
    max_items_per_class_split: int | None,
    timeout: float,
) -> list[MirrorObject]:
    dataset_cfg = config["datasets"]["objectron_relevant_classes"]
    prefix = dataset_cfg["r2_prefix"].strip("/")
    license_summary = dataset_cfg["license_summary"]
    rows: list[MirrorObject] = []

    for class_name in classes:
        if class_name not in dataset_cfg["selected_classes"]:
            raise ValueError(f"Objectron class {class_name!r} is not in selected_classes")
        for split in splits:
            index_url = objectron_index_url(class_name, split)
            index_key = f"{prefix}/v1/index/{class_name}_{split}.txt"
            rows.append(
                MirrorObject(
                    dataset="objectron_relevant_classes",
                    source_url=index_url,
                    r2_key=index_key,
                    asset_type="index",
                    license_summary=license_summary,
                )
            )
            sample_ids = [line.strip() for line in http_text(index_url, timeout).splitlines() if line.strip()]
            if max_items_per_class_split is not None:
                sample_ids = sample_ids[:max_items_per_class_split]

            for sample_id in sample_ids:
                rows.append(
                    MirrorObject(
                        dataset="objectron_relevant_classes",
                        source_url=f"{OBJECTRON_PUBLIC_BASE}/annotations/{sample_id}.pbdata",
                        r2_key=f"{prefix}/annotations/{sample_id}.pbdata",
                        asset_type="annotation",
                        license_summary=license_summary,
                    )
                )
                rows.append(
                    MirrorObject(
                        dataset="objectron_relevant_classes",
                        source_url=f"{OBJECTRON_PUBLIC_BASE}/videos/{sample_id}/geometry.pbdata",
                        r2_key=f"{prefix}/videos/{sample_id}/geometry.pbdata",
                        asset_type="ar_metadata",
                        license_summary=license_summary,
                    )
                )
                if include_videos:
                    rows.append(
                        MirrorObject(
                            dataset="objectron_relevant_classes",
                            source_url=f"{OBJECTRON_PUBLIC_BASE}/videos/{sample_id}/video.MOV",
                            r2_key=f"{prefix}/videos/{sample_id}/video.MOV",
                            asset_type="video",
                            license_summary=license_summary,
                        )
                    )
            if include_records:
                rows.append(
                    MirrorObject(
                        dataset="objectron_relevant_classes",
                        source_url=f"{OBJECTRON_PUBLIC_BASE}/v1/records_shuffled/{class_name}",
                        r2_key=f"{prefix}/v1/records_shuffled/{class_name}/",
                        asset_type="records_prefix_reference",
                        license_summary=license_summary,
                    )
                )
    return rows


def upload_url(client: Any, bucket: str, row: MirrorObject, timeout: float) -> MirrorObject:
    response = requests.get(row.source_url, stream=True, timeout=timeout)
    response.raise_for_status()
    response.raw.decode_content = True
    stream = HashingStream(response.raw)

    try:
        client.upload_fileobj(
            Fileobj=stream,
            Bucket=bucket,
            Key=row.r2_key,
            ExtraArgs={
                "Metadata": {
                "source-url-sha256": hashlib.sha256(row.source_url.encode("utf-8")).hexdigest(),
                "dataset": row.dataset,
                "asset-type": row.asset_type,
                }
            },
        )
        row.status = "uploaded"
        row.bytes = stream.bytes_read
        row.sha256 = stream.digest.hexdigest()
    except Exception as exc:
        row.status = "failed"
        row.error = f"{type(exc).__name__}: {exc}"
    return row


def iter_local_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield path


def upload_local_tree(
    config: dict[str, Any],
    local_dir: Path,
    r2_prefix: str,
    execute: bool,
) -> list[MirrorObject]:
    if not local_dir.exists() or not local_dir.is_dir():
        raise ValueError(f"local_dir must be an existing directory: {local_dir}")

    rows: list[MirrorObject] = []
    for path in iter_local_files(local_dir):
        rel = path.relative_to(local_dir).as_posix()
        rows.append(
            MirrorObject(
                dataset="local_prepared_3d_dataset",
                source_url=str(path),
                r2_key=f"{r2_prefix.strip('/')}/{rel}",
                asset_type="local_file",
                license_summary="Caller is responsible for confirming local dataset terms before upload.",
                bytes=path.stat().st_size,
            )
        )

    if not execute:
        return rows

    client = make_s3_client(config)
    bucket = config["r2"]["bucket"]
    for row in tqdm(rows, desc="Uploading local files"):
        try:
            client.upload_file(str(local_dir / Path(row.source_url).relative_to(local_dir)), bucket, row.r2_key)
            row.status = "uploaded"
        except Exception as exc:
            row.status = "failed"
            row.error = f"{type(exc).__name__}: {exc}"
    return rows


def command_preflight(config: dict[str, Any], require_credentials: bool) -> dict[str, Any]:
    payload = {
        "bucket": config["r2"]["bucket"],
        "endpoint_url": config["r2"]["endpoint_url"],
        "datasets": sorted(config["datasets"].keys()),
        "credentials_present": False,
        "credential_error": None,
    }
    try:
        r2_env(config)
        payload["credentials_present"] = True
    except RuntimeError as exc:
        payload["credential_error"] = str(exc)
        if require_credentials:
            raise
    return payload


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--timeout", type=float, default=60.0)
    sub = parser.add_subparsers(dest="command", required=True)

    p_preflight = sub.add_parser("preflight")
    p_preflight.add_argument("--require-credentials", action="store_true")

    p_obj = sub.add_parser("objectron-plan")
    p_obj.add_argument("--classes", nargs="+", default=["bottle", "cup", "book", "cereal_box", "shoe", "laptop", "chair"])
    p_obj.add_argument("--splits", nargs="+", default=["train", "test"], choices=["train", "test"])
    p_obj.add_argument("--include-videos", action="store_true")
    p_obj.add_argument("--include-records", action="store_true")
    p_obj.add_argument("--max-items-per-class-split", type=int, default=None)

    p_upload_urls = sub.add_parser("upload-url-list")
    p_upload_urls.add_argument("--plan", required=True)
    p_upload_urls.add_argument("--execute", action="store_true")

    p_upload_local = sub.add_parser("upload-local")
    p_upload_local.add_argument("--local-dir", required=True)
    p_upload_local.add_argument("--r2-prefix", required=True)
    p_upload_local.add_argument("--execute", action="store_true")

    args = parser.parse_args()
    config = load_config(Path(args.config))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.command == "preflight":
        payload = command_preflight(config, args.require_credentials)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if payload["credentials_present"] or not args.require_credentials else 1

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if args.command == "objectron-plan":
        rows = build_objectron_plan(
            config=config,
            classes=args.classes,
            splits=args.splits,
            include_videos=args.include_videos,
            include_records=args.include_records,
            max_items_per_class_split=args.max_items_per_class_split,
            timeout=args.timeout,
        )
        out = output_dir / f"objectron_plan_{timestamp}.jsonl"
        write_jsonl(out, rows)
        print(json.dumps({"plan": str(out), "objects": len(rows), "execute": False}, indent=2))
        return 0

    if args.command == "upload-url-list":
        rows = read_jsonl(Path(args.plan))
        if not args.execute:
            print(json.dumps({"plan": args.plan, "objects": len(rows), "execute": False}, indent=2))
            return 0
        try:
            client = make_s3_client(config)
        except RuntimeError as exc:
            print(json.dumps({"execute": True, "error": str(exc)}, indent=2))
            return 2
        bucket = config["r2"]["bucket"]
        uploaded = [upload_url(client, bucket, row, args.timeout) for row in tqdm(rows, desc="Uploading URLs")]
        out = output_dir / f"url_upload_manifest_{timestamp}.jsonl"
        write_jsonl(out, uploaded)
        failures = [row for row in uploaded if row.status == "failed"]
        print(json.dumps({"manifest": str(out), "objects": len(uploaded), "failures": len(failures)}, indent=2))
        return 1 if failures else 0

    if args.command == "upload-local":
        try:
            rows = upload_local_tree(config, Path(args.local_dir), args.r2_prefix, args.execute)
        except RuntimeError as exc:
            print(json.dumps({"execute": args.execute, "error": str(exc)}, indent=2))
            return 2
        out = output_dir / f"local_upload_manifest_{timestamp}.jsonl"
        write_jsonl(out, rows)
        failures = [row for row in rows if row.status == "failed"]
        print(
            json.dumps(
                {"manifest": str(out), "objects": len(rows), "failures": len(failures), "execute": args.execute},
                indent=2,
            )
        )
        return 1 if failures else 0

    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
