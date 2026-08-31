#!/usr/bin/env python3
"""Convert an ANN-Benchmarks HDF5 file into the C++ benchmark format."""

from __future__ import annotations

import argparse
import os
import shutil
import ssl
import struct
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


MAGIC = b"VDBANN01"
FORMAT_VERSION = 1
DISTANCE_CODES = {"angular": 1, "euclidean": 2}
HEADER = struct.Struct("<8sIIQQII")
CHUNK_ROWS = 16_384
DOWNLOAD_CHUNK_BYTES = 1 << 20

# Datasets from https://ann-benchmarks.com/ that this suite downloads and
# prepares automatically. See benchmarks/README.md for why each one is here.
KNOWN_DATASETS = {
    name: f"https://ann-benchmarks.com/{name}.hdf5"
    for name in (
        "glove-25-angular",
        "sift-128-euclidean",
        "glove-100-angular",
        "gist-960-euclidean",
        "nytimes-256-angular",
        "fashion-mnist-784-euclidean",
    )
}


class ConversionError(ValueError):
    """Raised when an ANN-Benchmarks dataset cannot be converted."""


def download_with_urllib(url: str, temporary_path: Path) -> None:
    request = urllib.request.Request(
        url, headers={"User-Agent": "vectordb-benchmarks/1.0"}
    )
    with urllib.request.urlopen(request) as response:
        total = response.length or 0
        written = 0
        with temporary_path.open("wb") as output:
            while True:
                chunk = response.read(DOWNLOAD_CHUNK_BYTES)
                if not chunk:
                    break
                output.write(chunk)
                written += len(chunk)
                if total:
                    print(
                        f"\r  {written / (1 << 20):.1f} / "
                        f"{total / (1 << 20):.1f} MiB",
                        end="",
                        flush=True,
                    )
    print()


def download_with_curl(url: str, temporary_path: Path) -> None:
    if shutil.which("curl") is None:
        raise ConversionError("curl is required as a TLS fallback but was not found on PATH")
    subprocess.run(
        [
            "curl",
            "--fail",
            "--location",
            "--show-error",
            "--user-agent",
            "vectordb-benchmarks/1.0",
            "--output",
            str(temporary_path),
            url,
        ],
        check=True,
    )


def download_dataset(name: str, destination: Path) -> None:
    url = KNOWN_DATASETS[name]
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_name(destination.name + ".part")
    print(f"Downloading {name} from {url} ...")
    try:
        try:
            download_with_urllib(url, temporary_path)
        except urllib.error.URLError as error:
            if not isinstance(error.reason, ssl.SSLCertVerificationError):
                raise
            # Some ANN-Benchmarks mirrors serve a legacy certificate chain
            # that this Python/OpenSSL combination rejects under strict
            # X.509 checks even though the chain is otherwise valid and
            # widely trusted (e.g. by curl/the OS trust store). Retry with
            # curl, which already validates the connection successfully.
            print("  TLS verification failed via urllib, retrying with curl ...")
            download_with_curl(url, temporary_path)
        os.replace(temporary_path, destination)
    except (OSError, urllib.error.URLError, subprocess.CalledProcessError) as error:
        temporary_path.unlink(missing_ok=True)
        raise ConversionError(f"failed to download {name} from {url}: {error}") from error


def ensure_downloaded(path: Path, dataset_name: str | None = None) -> None:
    if path.exists():
        return
    name = dataset_name or path.stem
    if name not in KNOWN_DATASETS:
        raise ConversionError(
            f"{path} does not exist and '{name}' is not a known dataset that "
            "can be downloaded automatically. Known datasets: "
            + ", ".join(sorted(KNOWN_DATASETS))
        )
    download_dataset(name, path)


def dense_shape(dataset: Any, name: str) -> tuple[int, int]:
    if len(dataset.shape) != 2:
        raise ConversionError(f"'{name}' must be a rank-2 dense array")
    rows, columns = (int(value) for value in dataset.shape)
    if rows <= 0 or columns <= 0:
        raise ConversionError(f"'{name}' must not be empty")
    return rows, columns


def write_array(output: Any, dataset: Any, dtype: str) -> None:
    import numpy as np

    for first_row in range(0, len(dataset), CHUNK_ROWS):
        values = np.asarray(
            dataset[first_row : first_row + CHUNK_ROWS],
            dtype=dtype,
            order="C",
        )
        output.write(values.tobytes(order="C"))


def convert(input_path: Path, output_path: Path) -> None:
    try:
        import h5py
    except ImportError as error:
        raise ConversionError(
            "h5py is required; install it with 'python3 -m pip install h5py'"
        ) from error

    if input_path.resolve() == output_path.resolve():
        raise ConversionError("input and output paths must be different")

    temporary_path = output_path.with_name(output_path.name + ".tmp")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with h5py.File(input_path, "r") as source:
            missing = {
                name
                for name in ("train", "test", "neighbors")
                if name not in source
            }
            if missing:
                raise ConversionError(
                    "dataset is missing arrays: " + ", ".join(sorted(missing))
                )

            distance = source.attrs.get("distance")
            if isinstance(distance, bytes):
                distance = distance.decode("utf-8")
            distance = str(distance).lower()
            if distance not in DISTANCE_CODES:
                raise ConversionError(
                    f"unsupported dataset distance '{distance}'; expected one "
                    "of: " + ", ".join(sorted(DISTANCE_CODES))
                )

            train_rows, train_dimension = dense_shape(
                source["train"], "train"
            )
            query_rows, query_dimension = dense_shape(source["test"], "test")
            neighbor_rows, neighbors_per_query = dense_shape(
                source["neighbors"], "neighbors"
            )

            if train_dimension != query_dimension:
                raise ConversionError(
                    "training and query dimensions do not match"
                )
            if query_rows != neighbor_rows:
                raise ConversionError(
                    "query and ground-truth row counts do not match"
                )
            if neighbors_per_query > train_rows:
                raise ConversionError(
                    "ground-truth width exceeds the training vector count"
                )

            with temporary_path.open("wb") as output:
                output.write(
                    HEADER.pack(
                        MAGIC,
                        FORMAT_VERSION,
                        train_dimension,
                        train_rows,
                        query_rows,
                        neighbors_per_query,
                        DISTANCE_CODES[distance],
                    )
                )
                write_array(output, source["train"], "<f4")
                write_array(output, source["test"], "<f4")
                write_array(output, source["neighbors"], "<u8")

        os.replace(temporary_path, output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def report_output(output_path: Path) -> None:
    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Wrote {output_path} ({size_mb:.1f} MiB)")


def prepare_named_dataset(name: str, data_dir: Path) -> None:
    input_path = data_dir / f"{name}.hdf5"
    output_path = data_dir / f"{name}.vdbann"
    ensure_downloaded(input_path, name)
    convert(input_path, output_path)
    report_output(output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert train, test, and neighbors from an ANN-Benchmarks HDF5 "
            "file into the dependency-free binary format used by the C++ "
            "benchmarks. Missing source .hdf5 files are downloaded "
            "automatically from https://ann-benchmarks.com/ when the "
            "filename (or --dataset) matches a known dataset."
        )
    )
    parser.add_argument(
        "input", type=Path, nargs="?", help="source .hdf5 file"
    )
    parser.add_argument(
        "output", type=Path, nargs="?", help="destination .vdbann file"
    )
    parser.add_argument(
        "--dataset",
        choices=sorted(KNOWN_DATASETS),
        help="prepare one known dataset into --data-dir by name, "
        "downloading its .hdf5 file first if needed",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="prepare every known dataset into --data-dir",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("benchmark-data"),
        help="directory for .hdf5/.vdbann files used by --dataset/--all "
        "(default: benchmark-data)",
    )
    args = parser.parse_args()

    if args.all and args.dataset:
        parser.error("--all and --dataset are mutually exclusive")
    if (args.all or args.dataset) and (args.input or args.output):
        parser.error(
            "--dataset/--all cannot be combined with explicit input/output paths"
        )
    if not args.all and not args.dataset and (args.input is None or args.output is None):
        parser.error("input and output are required unless --dataset or --all is given")
    return args


def main() -> int:
    args = parse_args()
    try:
        if args.all:
            for name in sorted(KNOWN_DATASETS):
                prepare_named_dataset(name, args.data_dir)
        elif args.dataset:
            prepare_named_dataset(args.dataset, args.data_dir)
        else:
            ensure_downloaded(args.input)
            convert(args.input, args.output)
            report_output(args.output)
    except (ConversionError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
