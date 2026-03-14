"""
Profile-aware data preparation and runtime utilities for autoresearch.

Usage:
    uv run prepare.py --profile climbmix_legacy
    uv run prepare.py --profile tinystories_8gb_search
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Iterator

import pyarrow.parquet as pq
import requests
import rustbpe
import tiktoken
import torch

from ar_runtime import (
    dataset_cache_dir,
    detect_device_type,
    ensure_dir,
    load_profile,
    read_json,
    tokenizer_cache_dir,
    write_json,
)


def download_file(url: str, filepath: Path, max_attempts: int = 5) -> bool:
    if filepath.exists():
        return True

    ensure_dir(filepath.parent)
    temp_path = filepath.with_suffix(filepath.suffix + ".tmp")
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            with temp_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
            os.replace(temp_path, filepath)
            print(f"  downloaded {filepath.name}")
            return True
        except (OSError, requests.RequestException) as exc:
            print(f"  attempt {attempt}/{max_attempts} failed for {filepath.name}: {exc}")
            temp_path.unlink(missing_ok=True)
            filepath.unlink(missing_ok=True)
            if attempt < max_attempts:
                time.sleep(2**attempt)
    return False


def resolve_hf_parquet_urls(dataset_cfg: dict[str, object]) -> list[str]:
    dataset_id = str(dataset_cfg["hf_dataset"])
    split_name = str(dataset_cfg.get("hf_split", "train"))
    url = f"https://huggingface.co/api/datasets/{dataset_id}/parquet"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    payload = response.json()
    entries = []

    if isinstance(payload, list):
        entries = payload
    elif isinstance(payload, dict):
        if "parquet_files" in payload:
            entries = payload["parquet_files"]
        else:
            config_name = dataset_cfg.get("hf_config")
            if config_name is None:
                if "default" in payload:
                    config_name = "default"
                elif len(payload) == 1:
                    config_name = next(iter(payload))
            selected = payload.get(config_name, {}) if config_name else {}
            if isinstance(selected, dict) and split_name in selected:
                entries = selected[split_name]
            elif isinstance(selected, list):
                entries = selected

    urls = []
    for entry in entries:
        if isinstance(entry, str):
            urls.append(entry)
            continue
        if not isinstance(entry, dict):
            continue
        if entry.get("split") and entry.get("split") != split_name:
            continue
        for key in ("url", "parquet_url", "download_url"):
            if entry.get(key):
                urls.append(entry[key])
                break
    if not urls:
        raise RuntimeError(f"Could not resolve parquet URLs for dataset {dataset_id}")
    return urls


def climbmix_file_paths(profile: dict[str, object], num_shards: int | None) -> list[tuple[str, Path]]:
    dataset_cfg = profile["dataset"]
    data_dir = ensure_dir(dataset_cache_dir(profile) / "data")
    base_url = str(dataset_cfg["base_url"])
    max_train_shard = int(dataset_cfg["max_train_shard"])
    val_shard = int(dataset_cfg["val_shard"])
    default_num_shards = int(dataset_cfg.get("default_num_shards", 10))
    requested = default_num_shards if num_shards is None else num_shards
    num_train = min(requested, max_train_shard)
    shard_ids = list(range(num_train))
    if val_shard not in shard_ids:
        shard_ids.append(val_shard)
    paths: list[tuple[str, Path]] = []
    for shard_id in shard_ids:
        filename = f"shard_{shard_id:05d}.parquet"
        paths.append((f"{base_url}/{filename}", data_dir / filename))
    return paths


def hf_parquet_file_paths(profile: dict[str, object]) -> list[tuple[str, Path]]:
    dataset_cfg = profile["dataset"]
    data_dir = ensure_dir(dataset_cache_dir(profile) / "data")
    urls = resolve_hf_parquet_urls(dataset_cfg)
    paths = []
    for index, url in enumerate(urls):
        filename = Path(url).name
        if not filename.endswith(".parquet"):
            filename = f"{dataset_cfg['hf_split']}-{index:05d}.parquet"
        paths.append((url, data_dir / filename))
    return paths


def download_dataset(
    profile: dict[str, object],
    num_shards: int | None = None,
    download_workers: int = 8,
) -> None:
    dataset_cfg = profile["dataset"]
    kind = dataset_cfg["kind"]
    if kind == "climbmix_shards":
        file_pairs = climbmix_file_paths(profile, num_shards)
    elif kind == "hf_parquet":
        file_pairs = hf_parquet_file_paths(profile)
    else:
        raise ValueError(f"Unsupported dataset kind: {kind}")

    existing = sum(1 for _, path in file_pairs if path.exists())
    if existing == len(file_pairs):
        print(f"Data: all {len(file_pairs)} files already downloaded")
        return

    print(f"Data: downloading {len(file_pairs) - existing} file(s) ({existing} already cached)")
    workers = max(1, min(download_workers, len(file_pairs)))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(download_file, url, path) for url, path in file_pairs]
        failures = 0
        for future in as_completed(futures):
            if not future.result():
                failures += 1
    if failures:
        raise RuntimeError(f"Failed to download {failures} dataset file(s)")

    metadata = {
        "dataset_id": profile["dataset_id"],
        "kind": kind,
        "downloaded_at": time.time(),
        "files": [str(path) for _, path in file_pairs],
    }
    write_json(dataset_cache_dir(profile) / "metadata.json", metadata)


def list_parquet_files(profile: dict[str, object]) -> list[Path]:
    data_dir = dataset_cache_dir(profile) / "data"
    return sorted(path for path in data_dir.glob("*.parquet"))


def iter_parquet_texts(path: Path, text_column: str) -> Iterator[str]:
    parquet_file = pq.ParquetFile(path)
    for row_group_idx in range(parquet_file.num_row_groups):
        row_group = parquet_file.read_row_group(row_group_idx, columns=[text_column])
        for text in row_group.column(text_column).to_pylist():
            yield text


def iter_ranged_parquet_texts(
    paths: list[Path],
    text_column: str,
    start: int,
    end: int | None,
) -> Iterator[str]:
    row_index = 0
    for path in paths:
        parquet_file = pq.ParquetFile(path)
        for row_group_idx in range(parquet_file.num_row_groups):
            row_group = parquet_file.read_row_group(row_group_idx, columns=[text_column])
            texts = row_group.column(text_column).to_pylist()
            next_row_index = row_index + len(texts)
            if next_row_index <= start:
                row_index = next_row_index
                continue
            if end is not None and row_index >= end:
                return
            batch_start = max(0, start - row_index)
            batch_end = len(texts) if end is None else min(len(texts), end - row_index)
            if batch_start < batch_end:
                for text in texts[batch_start:batch_end]:
                    yield text
            row_index = next_row_index


def iter_documents_once(profile: dict[str, object], split: str) -> Iterator[str]:
    dataset_cfg = profile["dataset"]
    text_column = str(dataset_cfg.get("text_column", "text"))
    kind = dataset_cfg["kind"]
    paths = list_parquet_files(profile)
    if not paths:
        raise RuntimeError("No parquet files found. Run prepare.py first.")

    if kind == "climbmix_shards":
        val_filename = f"shard_{int(dataset_cfg['val_shard']):05d}.parquet"
        if split == "train":
            target_paths = [path for path in paths if path.name != val_filename]
        else:
            target_paths = [path for path in paths if path.name == val_filename]
        if not target_paths:
            raise RuntimeError(f"No parquet files found for split={split}")
        for path in target_paths:
            yield from iter_parquet_texts(path, text_column)
        return

    if kind == "hf_parquet":
        start, end = dataset_cfg[f"{split}_range"]
        yield from iter_ranged_parquet_texts(paths, text_column, int(start), end)
        return

    raise ValueError(f"Unsupported dataset kind: {kind}")


def text_iterator(profile: dict[str, object]) -> Iterator[str]:
    tokenizer_cfg = profile["tokenizer"]
    max_chars = int(tokenizer_cfg.get("train_max_chars", 1_000_000_000))
    doc_cap = int(tokenizer_cfg.get("doc_cap", 10_000))
    total_chars = 0
    for text in iter_documents_once(profile, "train"):
        doc = text[:doc_cap] if len(text) > doc_cap else text
        total_chars += len(doc)
        yield doc
        if total_chars >= max_chars:
            return


def train_tokenizer(profile: dict[str, object]) -> None:
    tokenizer_cfg = profile["tokenizer"]
    target_dir = tokenizer_cache_dir(profile)
    tokenizer_path = target_dir / "tokenizer.pkl"
    token_bytes_path = target_dir / "token_bytes.pt"
    if tokenizer_path.exists() and token_bytes_path.exists():
        print(f"Tokenizer: already trained at {target_dir}")
        return

    ensure_dir(target_dir)
    files = list_parquet_files(profile)
    if len(files) < 1:
        raise RuntimeError("Need dataset files before training tokenizer")

    print("Tokenizer: training BPE tokenizer...")
    started = time.time()
    tokenizer = rustbpe.Tokenizer()
    special_tokens = list(tokenizer_cfg["special_tokens"])
    vocab_size = int(tokenizer_cfg["vocab_size"]) - len(special_tokens)
    tokenizer.train_from_iterator(
        text_iterator(profile),
        vocab_size,
        pattern=str(tokenizer_cfg["split_pattern"]),
    )

    mergeable_ranks = {bytes(key): value for key, value in tokenizer.get_mergeable_ranks()}
    token_offset = len(mergeable_ranks)
    special_token_map = {
        token: token_offset + index for index, token in enumerate(special_tokens)
    }
    encoding = tiktoken.Encoding(
        name=profile["tokenizer_id"],
        pat_str=tokenizer.get_pattern(),
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_token_map,
    )

    with tokenizer_path.open("wb") as handle:
        pickle.dump(encoding, handle)
    print(f"Tokenizer: trained in {time.time() - started:.1f}s")

    token_bytes = []
    special_set = set(special_tokens)
    for token_id in range(encoding.n_vocab):
        token_str = encoding.decode([token_id])
        if token_str in special_set:
            token_bytes.append(0)
        else:
            token_bytes.append(len(token_str.encode("utf-8")))
    torch.save(torch.tensor(token_bytes, dtype=torch.int32), token_bytes_path)
    print(f"Tokenizer: saved to {target_dir}")


class Tokenizer:
    def __init__(self, enc: tiktoken.Encoding, bos_token: str) -> None:
        self.enc = enc
        self.bos_token_id = enc.encode_single_token(bos_token)

    @classmethod
    def from_profile(cls, profile: dict[str, object]) -> "Tokenizer":
        target_dir = tokenizer_cache_dir(profile)
        with (target_dir / "tokenizer.pkl").open("rb") as handle:
            enc = pickle.load(handle)
        bos_token = str(profile["tokenizer"]["bos_token"])
        return cls(enc, bos_token)

    def get_vocab_size(self) -> int:
        return self.enc.n_vocab

    def get_bos_token_id(self) -> int:
        return self.bos_token_id

    def encode(self, text: str | list[str], prepend: int | None = None, num_threads: int = 8):
        if prepend is not None:
            prepend_id = prepend
        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend_id)
            return ids
        if isinstance(text, list):
            batches = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for row in batches:
                    row.insert(0, prepend_id)
            return batches
        raise ValueError(f"Unsupported input type: {type(text)}")

    def decode(self, ids: list[int]) -> str:
        return self.enc.decode(ids)


def get_token_bytes(profile: dict[str, object], device: str = "cpu") -> torch.Tensor:
    path = tokenizer_cache_dir(profile) / "token_bytes.pt"
    return torch.load(path, map_location=device)


def document_batches(
    profile: dict[str, object],
    split: str,
    tokenizer_batch_size: int = 128,
) -> Iterator[tuple[list[str], int]]:
    epoch = 1
    while True:
        batch: list[str] = []
        for text in iter_documents_once(profile, split):
            batch.append(text)
            if len(batch) >= tokenizer_batch_size:
                yield batch, epoch
                batch = []
        if batch:
            yield batch, epoch
        epoch += 1


def make_dataloader(
    profile: dict[str, object],
    tokenizer: Tokenizer,
    batch_size: int,
    sequence_len: int,
    split: str,
    buffer_size: int = 1000,
):
    assert split in {"train", "val"}
    row_capacity = sequence_len + 1
    batches = document_batches(profile, split)
    bos_token = tokenizer.get_bos_token_id()
    doc_buffer: list[list[int]] = []
    epoch = 1

    def refill_buffer() -> None:
        nonlocal epoch
        doc_batch, epoch = next(batches)
        doc_buffer.extend(tokenizer.encode(doc_batch, prepend=bos_token))

    device = detect_device_type()
    row_buffer = torch.empty((batch_size, row_capacity), dtype=torch.long)
    cpu_buffer = torch.empty(
        2 * batch_size * sequence_len,
        dtype=torch.long,
        pin_memory=(device == "cuda"),
    )
    device_buffer = torch.empty(2 * batch_size * sequence_len, dtype=torch.long, device=device)
    cpu_inputs = cpu_buffer[: batch_size * sequence_len].view(batch_size, sequence_len)
    cpu_targets = cpu_buffer[batch_size * sequence_len :].view(batch_size, sequence_len)
    inputs = device_buffer[: batch_size * sequence_len].view(batch_size, sequence_len)
    targets = device_buffer[batch_size * sequence_len :].view(batch_size, sequence_len)

    while True:
        for row_idx in range(batch_size):
            position = 0
            while position < row_capacity:
                while len(doc_buffer) < buffer_size:
                    refill_buffer()

                remaining = row_capacity - position
                best_idx = -1
                best_len = 0
                for index, doc in enumerate(doc_buffer):
                    doc_len = len(doc)
                    if doc_len <= remaining and doc_len > best_len:
                        best_idx = index
                        best_len = doc_len

                if best_idx >= 0:
                    doc = doc_buffer.pop(best_idx)
                    row_buffer[row_idx, position : position + len(doc)] = torch.tensor(
                        doc, dtype=torch.long
                    )
                    position += len(doc)
                else:
                    shortest_idx = min(range(len(doc_buffer)), key=lambda index: len(doc_buffer[index]))
                    doc = doc_buffer.pop(shortest_idx)
                    row_buffer[row_idx, position : position + remaining] = torch.tensor(
                        doc[:remaining], dtype=torch.long
                    )
                    position += remaining

        cpu_inputs.copy_(row_buffer[:, :-1])
        cpu_targets.copy_(row_buffer[:, 1:])
        device_buffer.copy_(cpu_buffer, non_blocking=(device == "cuda"))
        yield inputs, targets, epoch


@torch.no_grad()
def evaluate_bpb(
    profile: dict[str, object],
    model: torch.nn.Module,
    tokenizer: Tokenizer,
    batch_size: int,
) -> float:
    device = next(model.parameters()).device
    token_bytes = get_token_bytes(profile, device=str(device))
    loader = make_dataloader(profile, tokenizer, batch_size, int(profile["max_seq_len"]), "val")
    steps = max(1, int(profile["eval_tokens"]) // (batch_size * int(profile["max_seq_len"])))
    total_nats = 0.0
    total_bytes = 0
    for _ in range(steps):
        x, y, _ = next(loader)
        loss_flat = model(x, y, reduction="none").view(-1)
        y_flat = y.view(-1)
        nbytes = token_bytes[y_flat]
        mask = nbytes > 0
        total_nats += (loss_flat * mask).sum().item()
        total_bytes += int(nbytes.sum().item())
    return total_nats / (math.log(2) * total_bytes)


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare data and tokenizer for autoresearch")
    parser.add_argument("--profile", default="climbmix_legacy")
    parser.add_argument("--num-shards", type=int, default=None)
    parser.add_argument("--download-workers", type=int, default=8)
    args = parser.parse_args()

    profile = load_profile(args.profile)
    print(f"Preparing profile: {profile['profile_id']}")
    print(f"Dataset cache: {dataset_cache_dir(profile)}")
    print(f"Tokenizer cache: {tokenizer_cache_dir(profile)}")
    print()

    download_dataset(profile, num_shards=args.num_shards, download_workers=args.download_workers)
    print()
    train_tokenizer(profile)
    print()
    print("Done! Ready to train.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
