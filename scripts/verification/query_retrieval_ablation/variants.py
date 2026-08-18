"""Freeze Query Variants Before Retrieval."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from scripts.verification.query_decomposition.base import QueryDecomposer
from scripts.verification.query_translation.base import Translator
from .inputs import AblationQuery
from .models import QueryVariant


def freeze_query_variants(
    queries: Sequence[AblationQuery],
    translator: Translator,
    decomposer: QueryDecomposer,
    output_path: Path,
) -> list[QueryVariant]:
    """Translate and decompose queries, saving to a manifest."""
    variants = []
    
    # If file exists, load it to avoid re-running models.
    if output_path.is_file():
        try:
            payload = json.loads(output_path.read_text(encoding="utf-8"))
            for data in payload:
                variants.append(QueryVariant.from_dict(data))
            # Verify we have all queries
            cached_ids = {v.query_id for v in variants}
            if all(q.query_id in cached_ids for q in queries):
                return variants
        except json.JSONDecodeError:
            pass

    for q in queries:
        en_full = ""
        translation_success = False
        keyword_query = ""
        decomposition_success = False
        
        # 1. Translate
        try:
            en_full = translator.translate(q.query_vi)
            translation_success = True
        except Exception:
            translation_success = False

        # 2. Decompose
        if translation_success and en_full:
            try:
                # decomposer returns (concepts, keyword_query, raw_output)
                _, keyword_query, _ = decomposer.decompose(en_full)
                decomposition_success = True
            except Exception:
                decomposition_success = False

        # Try to get schema version from decomposer module, fallback to "v1"
        try:
            from scripts.verification.query_decomposition.qwen import SCHEMA_VERSION
            qwen_schema_version = SCHEMA_VERSION
        except ImportError:
            qwen_schema_version = "v1"

        variant = QueryVariant(
            query_id=q.query_id,
            query_vi=q.query_vi,
            query_en_full=en_full,
            keyword_query=keyword_query,
            translation_model=translator.name,
            qwen_model=decomposer.name,
            qwen_schema_version=qwen_schema_version,
            translation_success=translation_success,
            decomposition_success=decomposition_success,
        )
        variants.append(variant)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [v.to_dict() for v in variants]
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    
    return variants
