#!/usr/bin/env python3
"""Convert an ANN-Benchmarks HDF5 file into the C++ benchmark format."""

from __future__ import annotations

import argparse
import os
import struct
import sys
from pathlib import Path
from typing import Any


MAGIC = b"VDBANN01"
FORMAT_VERSION = 1
ANGULAR_DISTANCE = 1
HEADER = struct.Struct("<8sIIQQII")
CHUNK_ROWS = 16_384


class ConversionError(ValueError):
    """Raised when an ANN-Benchmarks dataset cannot be converted."""


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
            if str(distance).lower() != "angular":
                raise ConversionError(
                    "dataset distance must be 'angular' for random-projection "
                    "LSH"
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
                        ANGULAR_DISTANCE,
                    )
                )
                write_array(output, source["train"], "<f4")
                write_array(output, source["test"], "<f4")
                write_array(output, source["neighbors"], "<u8")

        os.replace(temporary_path, output_path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert train, test, and neighbors from an ANN-Benchmarks HDF5 "
            "file into the dependency-free binary format used by the C++ "
            "benchmarks."
        )
    )
    parser.add_argument("input", type=Path, help="source .hdf5 file")
    parser.add_argument("output", type=Path, help="destination .vdbann file")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        convert(args.input, args.output)
    except (ConversionError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1

    size_mb = args.output.stat().st_size / (1024 * 1024)
    print(f"Wrote {args.output} ({size_mb:.1f} MiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
