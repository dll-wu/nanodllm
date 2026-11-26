"""Reconstruct the 32 large-model generation prompts from the held-out local shard."""

import json
from bisect import bisect_right
from itertools import accumulate
from pathlib import Path

import pyarrow.parquet as pq

from .paper_eval_checkpoint import stream_sha256, tokenizer_identity
from .paper_eval_panels import reference_manifest
from .paper_runtime.core_task_data import json_digest
from .release_assets import release_asset

GENERATION_PANEL = release_asset("evaluation/tasks/generation32.json")


def reconstruct(source, tokenizer, panel):
    ends = list(accumulate(source.metadata.row_group(i).num_rows for i in range(source.num_row_groups)))
    previous = None
    prompts = []
    for row in panel["rows"]:
        i = row["row"]
        if type(i) is not int or i < 0 or i >= ends[-1]:
            raise ValueError("generation panel row is outside the local shard")
        group = bisect_right(ends, i)
        if group != previous:
            texts = source.read_row_group(group, columns=["text"]).column(0)
            previous = group
        offset = ends[group - 1] if group else 0
        body = texts[i - offset].as_py().strip()
        prefix = [tokenizer.get_bos_token_id(), *tokenizer.encode(body)][: panel["prefix_tokens"]]
        prompts.append({"id": row["id"], "prefix_ids": prefix})
    if json_digest(prompts) != panel["prompts_sha256"]:
        raise ValueError("generation prompt IDs or tokenization differ from the study")
    return prompts


def study_prompts(path, tokenizer):
    reference = reference_manifest()
    if tokenizer_identity(tokenizer) != reference["tokenizer_sha256"]:
        raise ValueError("the study prompts require the paper tokenizer")
    panel = json.loads(GENERATION_PANEL.read_bytes())
    with Path(path).open("rb") as source:
        if stream_sha256(source) != reference["shard_sha256"]:
            raise ValueError("the local shard differs from the held-out study shard")
        source.seek(0)
        return reconstruct(pq.ParquetFile(source), tokenizer, panel)
