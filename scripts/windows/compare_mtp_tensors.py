"""Compare embedded MTP tensors between two immutable checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors import safe_open


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("left", type=Path)
    parser.add_argument("right", type=Path)
    return parser.parse_args()


def weight_map(root: Path) -> dict[str, str]:
    index = json.loads((root / "model.safetensors.index.json").read_text())
    return index["weight_map"]


def load_tensor(root: Path, mapping: dict[str, str], name: str) -> torch.Tensor:
    with safe_open(
        root / mapping[name], framework="pt", device="cpu"
    ) as checkpoint:
        return checkpoint.get_tensor(name)


def main() -> None:
    args = parse_args()
    mappings = (weight_map(args.left), weight_map(args.right))
    names = sorted(name for name in mappings[0] if name.startswith("mtp."))
    if names != sorted(name for name in mappings[1] if name.startswith("mtp.")):
        raise ValueError("MTP tensor names differ between checkpoints")

    for name in names:
        left = load_tensor(args.left, mappings[0], name)
        right = load_tensor(args.right, mappings[1], name)
        if left.shape != right.shape or left.dtype != right.dtype:
            print(
                f"{name} | left={tuple(left.shape)}/{left.dtype} | "
                f"right={tuple(right.shape)}/{right.dtype} | compatible=False"
            )
            continue
        equal = torch.equal(left, right)
        max_abs_delta = 0.0
        if not equal:
            max_abs_delta = float((left.float() - right.float()).abs().max())
        print(
            f"{name} | shape={tuple(left.shape)} | dtype={left.dtype} | "
            f"equal={equal} | max_abs_delta={max_abs_delta:.9g}"
        )


if __name__ == "__main__":
    main()
