"""Детерминированная упаковка и безопасная распаковка training snapshots."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any

from .dataset_distribution import validate_prepared_dataset

ARCHIVE_SUFFIX = ".tar.zst"
ARCHIVE_LEVEL = 19
ARCHIVED_METADATA = ("dataset.json", "split.json", "annot_COCO.json")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_files(root: Path) -> dict[str, str]:
    root = Path(root).resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset directory does not exist: {root}")
    hashes = {}
    paths = sorted(root.rglob("*"), key=lambda path: path.relative_to(root).as_posix())
    for path in paths:
        if path.is_symlink():
            raise ValueError(f"Prepared dataset must not contain symlinks: {path}")
        if not path.is_file() or path.name.endswith(":Zone.Identifier"):
            continue
        hashes[path.relative_to(root).as_posix()] = _sha256(path)
    return hashes


def _tree_sha256(hashes: dict[str, str]) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(hashes.items()):
        digest.update(name.encode("utf-8"))
        digest.update(bytes.fromhex(value))
    return digest.hexdigest()


def _archive_base(path: Path) -> str:
    name = Path(path).name
    if not name.endswith(ARCHIVE_SUFFIX):
        raise ValueError(f"Archive path must end with {ARCHIVE_SUFFIX}: {name}")
    base = name[: -len(ARCHIVE_SUFFIX)]
    if not base:
        raise ValueError("Archive filename must have a non-empty base name")
    return base


def _sidecars(path: Path) -> dict[str, Path]:
    base = _archive_base(path)
    parent = Path(path).parent
    return {
        "dataset.json": parent / f"{base}.dataset.json",
        "split.json": parent / f"{base}.split.json",
        "annot_COCO.json": parent / f"{base}.annot_COCO.json",
        "archive.json": parent / f"{base}.archive.json",
    }


def _write_tar_zst(root: Path, target: Path, hashes: dict[str, str]) -> None:
    try:
        import zstandard
    except ImportError as error:
        raise RuntimeError("Archive creation requires zstandard in the full environment") from error

    compressor = zstandard.ZstdCompressor(
        level=ARCHIVE_LEVEL, write_checksum=True, write_content_size=False, threads=0
    )
    with Path(target).open("wb") as raw:
        with compressor.stream_writer(raw, closefd=False) as compressed:
            with tarfile.open(fileobj=compressed, mode="w|", format=tarfile.PAX_FORMAT) as tar:
                for name in sorted(hashes):
                    source = Path(root) / name
                    info = tarfile.TarInfo(name=name)
                    info.size = source.stat().st_size
                    info.mode = 0o644
                    info.mtime = 0
                    info.uid = 0
                    info.gid = 0
                    info.uname = ""
                    info.gname = ""
                    with source.open("rb") as stream:
                        tar.addfile(info, stream)


def _extract_tar_zst(archive: Path, destination: Path) -> None:
    try:
        import zstandard
    except ImportError as error:
        raise RuntimeError("Archive extraction requires zstandard in the full environment") from error

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    resolved_root = destination.resolve()
    seen = set()
    with Path(archive).open("rb") as raw:
        with zstandard.ZstdDecompressor().stream_reader(raw) as decompressed:
            with tarfile.open(fileobj=decompressed, mode="r|") as tar:
                for member in tar:
                    relative = PurePosixPath(member.name)
                    if (
                        not relative.parts
                        or relative.is_absolute()
                        or "\\" in member.name
                        or any(part in {".", ".."} or ":" in part for part in relative.parts)
                    ):
                        raise ValueError(f"Unsafe path in training archive: {member.name}")
                    name = relative.as_posix()
                    if name in seen:
                        raise ValueError(f"Duplicate path in training archive: {name}")
                    seen.add(name)
                    target = destination.joinpath(*relative.parts)
                    if not target.resolve().is_relative_to(resolved_root):
                        raise ValueError(f"Archive path escapes output directory: {member.name}")
                    if member.isdir():
                        target.mkdir(parents=True, exist_ok=True)
                    elif member.isfile():
                        target.parent.mkdir(parents=True, exist_ok=True)
                        source = tar.extractfile(member)
                        if source is None:
                            raise ValueError(f"Could not read archive member: {member.name}")
                        with source, target.open("xb") as output:
                            shutil.copyfileobj(source, output, length=1024 * 1024)
                        if target.stat().st_size != member.size:
                            raise ValueError(f"Truncated archive member: {member.name}")
                    else:
                        raise ValueError(f"Unsupported archive member: {member.name}")


def _verify_sidecars(archive: Path, extracted: Path) -> None:
    sidecars = _sidecars(archive)
    for name in ARCHIVED_METADATA:
        sidecar = sidecars[name]
        if sidecar.exists():
            copy = Path(extracted) / name
            if not sidecar.is_file() or not copy.is_file() or _sha256(sidecar) != _sha256(copy):
                raise ValueError(f"Metadata sidecar differs from archive: {sidecar}")
    report_path = sidecars["archive.json"]
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(f"Could not read archive report: {report_path}") from error
        expected = report.get("archive_sha256") if isinstance(report, dict) else None
        if expected and _sha256(archive) != expected:
            raise ValueError(f"Archive SHA-256 differs from report: {archive}")


def pack_training_dataset(data_root: Path, archive_path: Path) -> dict[str, Any]:
    """Упаковать dataset, проверить точный round-trip и сохранить sidecar-отчёт."""
    root = Path(data_root).resolve()
    archive = Path(archive_path).resolve()
    _archive_base(archive)
    if root in archive.parents:
        raise ValueError("Archive output must be outside the prepared dataset")
    validate_prepared_dataset(root)
    for name in ARCHIVED_METADATA:
        if not (root / name).is_file():
            raise FileNotFoundError(f"Prepared dataset is missing required file: {name}")
    if archive.exists():
        raise FileExistsError(f"Refusing to replace existing archive: {archive}")

    archive.parent.mkdir(parents=True, exist_ok=True)
    sidecars = _sidecars(archive)
    if any(path.exists() for path in sidecars.values()):
        raise FileExistsError("Refusing to replace existing training archive sidecars")
    source_hashes = _hash_files(root)
    source_bytes = sum((root / name).stat().st_size for name in source_hashes)

    with tempfile.TemporaryDirectory(prefix=".colony-pack-", dir=archive.parent) as workspace:
        workspace = Path(workspace)
        staged_archive = workspace / archive.name
        _write_tar_zst(root, staged_archive, source_hashes)
        extracted = workspace / "roundtrip"
        _extract_tar_zst(staged_archive, extracted)
        if _hash_files(extracted) != source_hashes:
            raise RuntimeError("Training archive round-trip SHA-256 verification failed")

        archive_bytes = staged_archive.stat().st_size
        report = {
            "schema_version": 1,
            "archive": archive.name,
            "archive_sha256": _sha256(staged_archive),
            "compression": "zstandard",
            "compression_level": ARCHIVE_LEVEL,
            "tar_metadata": "normalized",
            "files": len(source_hashes),
            "uncompressed_bytes": source_bytes,
            "archive_bytes": archive_bytes,
            "compressed_to_uncompressed_ratio": archive_bytes / source_bytes,
            "uncompressed_to_archive_factor": source_bytes / archive_bytes,
            "dataset_tree_sha256": _tree_sha256(source_hashes),
            "roundtrip_sha256_verified": True,
            "metadata_sidecars": {},
        }
        staged_sidecars = workspace / "sidecars"
        staged_sidecars.mkdir()
        for name in ARCHIVED_METADATA:
            target = staged_sidecars / sidecars[name].name
            shutil.copyfile(root / name, target)
            report["metadata_sidecars"][name] = {
                "file": target.name,
                "sha256": _sha256(target),
            }
        staged_report = staged_sidecars / sidecars["archive.json"].name
        staged_report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

        os.replace(staged_archive, archive)
        for sidecar in sidecars.values():
            os.replace(staged_sidecars / sidecar.name, sidecar)
    return report


def unpack_training_archive(archive_path: Path, output_dir: Path) -> dict[str, Any]:
    """Безопасно распаковать архив в новый каталог и сверить доступные sidecar-файлы."""
    archive = Path(archive_path).resolve()
    _archive_base(archive)
    if not archive.is_file():
        raise FileNotFoundError(f"Training archive does not exist: {archive}")
    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to replace existing dataset directory: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=".colony-unpack-", dir=output.parent) as workspace:
        staged = Path(workspace) / "dataset"
        _extract_tar_zst(archive, staged)
        for name in ARCHIVED_METADATA:
            if not (staged / name).is_file():
                raise FileNotFoundError(f"Training archive is missing required file: {name}")
        manifest, split = validate_prepared_dataset(staged)
        _verify_sidecars(archive, staged)
        files = len(_hash_files(staged))
        sources = sum(record.kind == "source" for record in manifest.samples)
        variants = sum(record.kind == "augmented" for record in manifest.samples)
        os.replace(staged, output)

    return {
        "output": str(output),
        "files": files,
        "samples": len(manifest.samples),
        "sources": sources,
        "black_background_variants": variants,
        "split_counts": {subset: len(split["groups"][subset]) for subset in ("train", "val", "test")},
    }
