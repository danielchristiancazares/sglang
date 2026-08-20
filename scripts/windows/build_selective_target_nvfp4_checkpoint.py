"""Build a RadixArk checkpoint with only target attention projections in NVFP4."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import struct
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sizes(path: Path) -> dict[str, int]:
    with path.open("rb") as handle:
        header_length = struct.unpack("<Q", handle.read(8))[0]
        header = json.loads(handle.read(header_length))
    return {
        name: value["data_offsets"][1] - value["data_offsets"][0]
        for name, value in header.items()
        if name != "__metadata__"
    }


def load_tensor(root: Path, index: dict, name: str):
    shard = root / index["weight_map"][name]
    with safe_open(shard, framework="pt", device="cpu") as handle:
        # Windows closes the underlying file mapping with the safe_open handle.
        # The output group outlives that handle, so give every tensor owned storage.
        return handle.get_tensor(name).clone()


def convert_runtime_quant_config(config: dict, selected_bases: list[str]) -> dict:
    quant_config = config["quantization_config"]
    quantized_layers = quant_config["quantized_layers"]
    selected_set = set(selected_bases)
    if not selected_set.issubset(quantized_layers):
        missing = sorted(selected_set - quantized_layers.keys())
        raise ValueError(f"config.json is missing selected layers: {missing}")

    for base_name in selected_bases:
        quantized_layers[base_name] = {"quant_algo": "NVFP4", "group_size": 16}

    config_groups = quant_config.get("config_groups", {})
    nvfp4_groups = [
        group
        for group in config_groups.values()
        if group.get("weights", {}).get("num_bits") == 4
    ]
    if len(nvfp4_groups) != 1:
        raise ValueError(
            f"Expected one 4-bit config group in config.json, got {len(nvfp4_groups)}"
        )
    for group in config_groups.values():
        if "targets" in group:
            group["targets"] = [
                target for target in group["targets"] if target not in selected_set
            ]
    nvfp4_groups[0]["targets"] = sorted(
        set(nvfp4_groups[0].get("targets", ())) | selected_set
    )
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("base", type=Path)
    parser.add_argument("donor", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume-metadata-only", action="store_true")
    parser.add_argument("--max-shard-size-gib", type=float, default=2.0)
    args = parser.parse_args()

    base = args.base.resolve()
    donor = args.donor.resolve()
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        if not args.resume_metadata_only:
            raise FileExistsError(f"Output must be absent or metadata-only: {output}")

    base_index = read_json(base / "model.safetensors.index.json")
    donor_index = read_json(donor / "model.safetensors.index.json")
    base_quant = read_json(base / "hf_quant_config.json")
    donor_quant = read_json(donor / "hf_quant_config.json")

    quantized_layers = base_quant["quantization"]["quantized_layers"]
    selected_bases = sorted(
        name
        for name, config in quantized_layers.items()
        if config.get("quant_algo") == "FP8"
    )
    if len(selected_bases) != 208:
        raise ValueError(f"Expected 208 target FP8 projection bases, got {len(selected_bases)}")
    if donor_quant["quantization"].get("quant_algo") != "NVFP4":
        raise ValueError("Donor must declare whole-checkpoint NVFP4")

    def selected(name: str) -> bool:
        return any(name.startswith(f"{base_name}.") for base_name in selected_bases)

    donor_keys_by_base = {}
    for base_name in selected_bases:
        keys = sorted(
            name
            for name in donor_index["weight_map"]
            if name.startswith(f"{base_name}.")
        )
        suffixes = {name.removeprefix(f"{base_name}.") for name in keys}
        expected = {"input_scale", "weight", "weight_scale", "weight_scale_2"}
        if suffixes != expected:
            raise ValueError(
                f"Donor schema mismatch for {base_name}: {sorted(suffixes)}"
            )
        donor_keys_by_base[base_name] = keys

    source_map = {
        name: (base, base_index)
        for name in base_index["weight_map"]
        if not selected(name)
    }
    for base_name, keys in donor_keys_by_base.items():
        for name in keys:
            source_map[name] = (donor, donor_index)

    size_cache = {}
    tensor_size_map = {}
    for name, (root, index) in source_map.items():
        shard_name = index["weight_map"][name]
        cache_key = (root, shard_name)
        if cache_key not in size_cache:
            size_cache[cache_key] = tensor_sizes(root / shard_name)
        tensor_size_map[name] = size_cache[cache_key][name]

    max_shard_bytes = int(args.max_shard_size_gib * 1024**3)
    groups = []
    current = []
    current_size = 0
    for name in sorted(source_map):
        size = tensor_size_map[name]
        if current and current_size + size > max_shard_bytes:
            groups.append(current)
            current = []
            current_size = 0
        current.append(name)
        current_size += size
    if current:
        groups.append(current)

    output_map = {}
    by_shard = {}
    shard_count = len(groups)
    for index, names in enumerate(groups, start=1):
        shard_name = f"model-{index:05d}-of-{shard_count:05d}.safetensors"
        by_shard[shard_name] = names
        output_map.update({name: shard_name for name in names})

    plan = {
        "base": str(base),
        "donor": str(donor),
        "output": str(output),
        "selected_projection_bases": len(selected_bases),
        "selected_tensor_count": sum(map(len, donor_keys_by_base.values())),
        "output_tensor_count": len(output_map),
        "max_shard_size_gib": args.max_shard_size_gib,
        "shards": {
            name: {
                "tensor_count": len(keys),
                "tensor_bytes": sum(tensor_size_map[key] for key in keys),
            }
            for name, keys in sorted(by_shard.items())
        },
    }
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return

    output.mkdir(parents=True, exist_ok=True)
    excluded_files = {
        "hf_quant_config.json",
        "model.safetensors.index.json",
        "selective-nvfp4-manifest.json",
    }
    if not args.resume_metadata_only:
        for source in base.iterdir():
            if (
                source.is_file()
                and source.name not in excluded_files
                and source.suffix != ".safetensors"
            ):
                shutil.copy2(source, output / source.name)

    total_size = 0
    shard_manifest = []
    for shard_name, names in sorted(by_shard.items()):
        destination = output / shard_name
        if args.resume_metadata_only:
            if not destination.is_file():
                raise FileNotFoundError(f"Missing output shard: {destination}")
            total_size += sum(tensor_size_map[name] for name in names)
            shard_manifest.append(
                {
                    "name": shard_name,
                    "size": destination.stat().st_size,
                    "sha256": sha256(destination),
                }
            )
            continue
        tensors = {}
        for name in sorted(names):
            source_root, source_index = source_map[name]
            tensor = load_tensor(source_root, source_index, name)
            tensors[name] = tensor
            total_size += tensor.numel() * tensor.element_size()
        save_file(tensors, destination, metadata={"format": "pt"})
        shard_manifest.append(
            {
                "name": shard_name,
                "size": destination.stat().st_size,
                "sha256": sha256(destination),
            }
        )
        del tensors

    for base_name in selected_bases:
        quantized_layers[base_name] = {"quant_algo": "NVFP4", "group_size": 16}
    (output / "hf_quant_config.json").write_text(
        json.dumps(base_quant, indent=2) + "\n", encoding="utf-8"
    )
    runtime_config = convert_runtime_quant_config(
        read_json(base / "config.json"), selected_bases
    )
    (output / "config.json").write_text(
        json.dumps(runtime_config, indent=2) + "\n", encoding="utf-8"
    )
    output_index = {
        "metadata": {
            **base_index.get("metadata", {}),
            "total_size": total_size,
        },
        "weight_map": dict(sorted(output_map.items())),
    }
    if args.resume_metadata_only:
        existing_index = read_json(output / "model.safetensors.index.json")
        if existing_index.get("weight_map") != output_index["weight_map"]:
            raise ValueError("Existing output index does not match the rebuilt tensor map")
    else:
        (output / "model.safetensors.index.json").write_text(
            json.dumps(output_index, indent=2) + "\n", encoding="utf-8"
        )

    manifest = {
        **plan,
        "selection": "all 208 base-checkpoint FP8 quantized_layers",
        "preserved": "all other tensors and metadata from base checkpoint",
        "source_checkpoints_mutated": False,
        "output_shards": shard_manifest,
        "hf_quant_config_sha256": sha256(output / "hf_quant_config.json"),
        "config_sha256": sha256(output / "config.json"),
        "index_sha256": sha256(output / "model.safetensors.index.json"),
    }
    (output / "selective-nvfp4-manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
