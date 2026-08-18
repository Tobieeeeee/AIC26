"""Input loading for the ablation benchmark."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class AblationQuery:
    query_id: str
    query_vi: str
    video_id: str
    start_frame: int
    end_frame: int


def load_ablation_queries(gt_path: Path) -> list[AblationQuery]:
    """Load query + GT from JSON.
    
    Expected format:
    {
      "queries": [
        {
          "query_id": "q1",
          "query_vi": "Tiếng Việt...",
          "video_id": "L21_V001",
          "start_frame": 100,
          "end_frame": 200
        }
      ]
    }
    """
    path = Path(gt_path)
    if not path.is_file():
        raise FileNotFoundError(f"GT file not found: {path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in GT file: {e}") from e

    queries = []
    for row in payload.get("queries", []):
        if "query_vi" not in row:
            raise ValueError(f"Missing 'query_vi' in query {row.get('query_id')}")
        queries.append(
            AblationQuery(
                query_id=row["query_id"],
                query_vi=row["query_vi"],
                video_id=row["video_id"],
                start_frame=row["start_frame"],
                end_frame=row["end_frame"],
            )
        )
    return queries
