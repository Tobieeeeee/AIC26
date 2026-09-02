from __future__ import annotations

import json
from pathlib import Path

import pytest

from mvr.operations.enrichment_version import (
    EnrichmentVersionError,
    activate_enrichment_version,
    apply_enrichment_pointer,
    resolve_active_enrichment_version,
    validate_enrichment_version,
)
from mvr.operations.enrichment_catalog import read_enrichment_catalog
from mvr.operations.r2_layout import (
    VERSIONS_PREFIX,
    enrichment_version_prefix,
    version_key,
)


def test_version_key_builds_an_immutable_prefix_outside_the_corpus_tree() -> None:
    key = version_key("enrichment", "enrich-batch1-v1", "batch=1", "collection=L21")

    assert key == (
        "data/versions/enrichment/enrich-batch1-v1/batch=1/collection=L21"
    )
    assert key.startswith(f"{VERSIONS_PREFIX}/")
    assert not key.startswith("data/corpus/")


def test_enrichment_version_prefix_is_the_run_scoped_root() -> None:
    assert enrichment_version_prefix("enrich-batch1-v1") == (
        "data/versions/enrichment/enrich-batch1-v1"
    )


@pytest.mark.parametrize("version", ["", "   ", "a/b", ".", "..", "v/"])
def test_version_key_rejects_a_version_that_is_not_one_path_segment(version: str) -> None:
    with pytest.raises(ValueError):
        version_key("enrichment", version)


@pytest.mark.parametrize("feature", ["", "   ", "a/b", ".."])
def test_version_key_rejects_a_feature_that_is_not_one_path_segment(feature: str) -> None:
    with pytest.raises(ValueError):
        version_key(feature, "enrich-batch1-v1")


def test_manifest_without_a_pointer_resolves_to_no_active_version() -> None:
    assert resolve_active_enrichment_version({"collections": {}}) is None
    assert resolve_active_enrichment_version({"active_enrichment_version": None}) is None


def test_pointer_resolves_to_the_activated_version() -> None:
    manifest = {"active_enrichment_version": " enrich-batch1-v1 "}

    assert resolve_active_enrichment_version(manifest) == "enrich-batch1-v1"


@pytest.mark.parametrize("pointer", [123, "", "   ", "a/b", ".."])
def test_malformed_pointer_is_rejected_instead_of_silently_ignored(pointer: object) -> None:
    with pytest.raises(EnrichmentVersionError):
        resolve_active_enrichment_version({"active_enrichment_version": pointer})


def test_applying_the_pointer_records_the_version_prefix_without_mutating_the_input() -> None:
    manifest = {"schema_version": "1.0", "downstream_status": "REBUILD_REQUIRED"}

    updated = apply_enrichment_pointer(
        manifest, version="enrich-batch1-v1", activated_at="2026-09-02T00:00:00Z"
    )

    assert manifest == {"schema_version": "1.0", "downstream_status": "REBUILD_REQUIRED"}
    assert updated["active_enrichment_version"] == "enrich-batch1-v1"
    assert updated["enrichment"]["prefix"] == "data/versions/enrichment/enrich-batch1-v1"
    assert updated["enrichment"]["activated_at"] == "2026-09-02T00:00:00Z"
    assert updated["downstream_status"] == "ENRICHED"


def test_applying_the_pointer_records_the_previous_version_so_rollback_is_one_put() -> None:
    first = apply_enrichment_pointer(
        {}, version="enrich-batch1-v1", activated_at="2026-09-02T00:00:00Z"
    )

    second = apply_enrichment_pointer(
        first, version="enrich-batch1-v2", activated_at="2026-09-03T00:00:00Z"
    )
    rolled_back = apply_enrichment_pointer(
        second, version="enrich-batch1-v1", activated_at="2026-09-04T00:00:00Z"
    )

    assert first["enrichment"]["previous_version"] is None
    assert second["enrichment"]["previous_version"] == "enrich-batch1-v1"
    assert rolled_back["active_enrichment_version"] == "enrich-batch1-v1"
    assert rolled_back["enrichment"]["previous_version"] == "enrich-batch1-v2"


class _FakeClient:
    """Minimal R2Client surface: list/download/upload over an in-memory bucket."""

    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = dict(objects)
        self.uploaded: list[str] = []
        self.listings = 0

    def list_objects(self, prefix, **kwargs):
        self.listings += 1
        return [
            {"Key": key, "Size": len(value)}
            for key, value in sorted(self.objects.items())
            if key.startswith(prefix)
        ]

    def download_file(self, key, local_path, **kwargs):
        destination = Path(local_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.objects[key])
        return destination

    def upload_file(self, local_path, key, **kwargs):
        payload = Path(local_path).read_bytes()
        self.objects[key] = payload
        self.uploaded.append(key)
        return {"key": key, "size": len(payload), "sha256": "unused"}


NEWLINE = chr(10)
PROVIDERS = ("beit3", "metaclip2", "perception_encoder")


def _mapping_rows(collection: str, videos: int, keyframes: int) -> list[dict]:
    """Keyframe rows spread evenly across a collection's videos."""
    per_video = keyframes // videos
    rows = []
    for v in range(1, videos + 1):
        video_id = f"{collection}_V{v:03d}"
        for i in range(per_video):
            rows.append(
                {
                    "keyframe_id": f"{video_id}_S{i + 1:04d}_K01",
                    "video_id": video_id,
                    "feature_file": f"{video_id}.npy",
                    "feature_index": str(i),
                    "frame_id": str(i * 10),
                    "shot_id": str(i + 1),
                    "timestamp": f"{i}.5",
                    "image_path": f"data/corpus/keyframes/{collection}/{video_id}/k{i}.jpg",
                }
            )
    return rows


def _mapping_csv(rows: list[dict]) -> bytes:
    columns = [
        "keyframe_id", "video_id", "feature_file", "feature_index",
        "frame_id", "shot_id", "timestamp", "image_path",
    ]
    lines = [",".join(columns)]
    lines.extend(",".join(str(row[c]) for c in columns) for row in rows)
    return (NEWLINE.join(lines) + NEWLINE).encode()


def _collection_objects(prefix: str, collection: str, *, videos: int, keyframes: int) -> dict:
    root = f"{prefix}/batch=1/collection={collection}"
    marker = {
        "schema_version": "2.0",
        "status": "DONE",
        "run_id": prefix.rsplit("/", maxsplit=1)[-1],
        "batch_id": "batch1",
        "collection_id": collection,
        "video_count": videos,
        "keyframe_count": keyframes,
        "providers": list(PROVIDERS),
        "features": ["objects", *PROVIDERS, "ocr", "asr"],
        "object_transport": "task05a/objects.tar.gz",
        "object_metadata": "metadata/objects/part-000.parquet",
        "ocr_transport": "task05a/ocr.tar.gz",
        "ocr_metadata": "metadata/ocr/part-000.parquet",
        "asr_transport": "task05a/asr.tar.gz",
        "asr_metadata": "metadata/asr/part-000.parquet",
    }
    objects = {f"{root}/_SUCCESS.json": json.dumps(marker).encode()}
    for feature in ("objects", "ocr", "asr"):
        objects[f"{root}/task05a/{feature}.tar.gz"] = b"archive"
        objects[f"{root}/metadata/{feature}/part-000.parquet"] = b"parquet"
    rows = _mapping_rows(collection, videos, keyframes)
    for provider in PROVIDERS:
        base = f"{root}/embeddings/{provider}"
        objects[f"{base}/manifest.json"] = b"{}"
        objects[f"{base}/feature_mapping.csv"] = _mapping_csv(rows)
        for video_id in sorted({row["video_id"] for row in rows}):
            objects[f"{base}/{video_id}.npy"] = b"vectors"
    objects[f"{root}/ocr/manifest.json"] = b"{}"
    objects[f"{root}/asr/manifest.json"] = b"{}"
    return objects


def _version_objects(
    version: str,
    collections: tuple[str, ...],
    *,
    videos: int = 2,
    keyframes: int = 4,
) -> dict[str, bytes]:
    """A structurally complete, publishable version."""
    prefix = f"data/versions/enrichment/{version}"
    objects = {
        f"{prefix}/_READY.json": json.dumps(
            {
                "schema_version": "2.0",
                "status": "READY",
                "run_id": version,
                "batch_id": "batch1",
                "collections": list(collections),
                "providers": list(PROVIDERS),
                "video_count": videos * len(collections),
                "keyframe_count": keyframes * len(collections),
            }
        ).encode()
    }
    for collection in collections:
        objects.update(
            _collection_objects(prefix, collection, videos=videos, keyframes=keyframes)
        )
    return objects


def _bucket(version: str = "enrich-v1", collections: tuple[str, ...] = ("L21",)) -> dict:
    objects = {
        "data/corpus/dataset_manifest.json": json.dumps(
            {"schema_version": "1.0", "downstream_status": "REBUILD_REQUIRED"}
        ).encode()
    }
    objects.update(_version_objects(version, collections))
    return objects


def test_activation_is_refused_when_the_version_has_no_ready_marker(tmp_path) -> None:
    objects = _bucket()
    del objects["data/versions/enrichment/enrich-v1/_READY.json"]
    client = _FakeClient(objects)

    with pytest.raises(EnrichmentVersionError, match="_READY"):
        activate_enrichment_version(
            client=client, version="enrich-v1", work_root=tmp_path
        )

    assert client.uploaded == []


def test_activation_is_refused_when_a_collection_success_marker_is_missing(
    tmp_path,
) -> None:
    objects = _bucket(collections=("L21", "L22"))
    del objects["data/versions/enrichment/enrich-v1/batch=1/collection=L22/_SUCCESS.json"]
    client = _FakeClient(objects)

    with pytest.raises(EnrichmentVersionError, match="L22"):
        activate_enrichment_version(
            client=client, version="enrich-v1", work_root=tmp_path
        )

    assert client.uploaded == []


def test_activation_swaps_the_pointer_and_backs_up_the_previous_manifest(
    tmp_path,
) -> None:
    client = _FakeClient(_bucket())

    report = activate_enrichment_version(
        client=client,
        version="enrich-v1",
        work_root=tmp_path,
        activated_at="2026-09-02T00:00:00Z",
    )

    manifest = json.loads(client.objects["data/corpus/dataset_manifest.json"])
    assert manifest["active_enrichment_version"] == "enrich-v1"
    assert manifest["enrichment"]["prefix"] == "data/versions/enrichment/enrich-v1"
    assert report["previous_version"] is None
    backups = [key for key in client.uploaded if "manifest-backups" in key]
    assert len(backups) == 1
    assert json.loads(client.objects[backups[0]])["downstream_status"] == (
        "REBUILD_REQUIRED"
    )
    assert client.uploaded == [backups[0], "data/corpus/dataset_manifest.json"]


def test_activation_copies_no_artifact_and_rolls_back_with_one_manifest_put(
    tmp_path,
) -> None:
    objects = _bucket()
    objects.update(_version_objects("enrich-v2", ("L21",)))
    client = _FakeClient(objects)
    before = set(client.objects)

    activate_enrichment_version(
        client=client, version="enrich-v1", work_root=tmp_path / "a"
    )
    activate_enrichment_version(
        client=client, version="enrich-v2", work_root=tmp_path / "b"
    )
    client.uploaded.clear()
    report = activate_enrichment_version(
        client=client, version="enrich-v1", work_root=tmp_path / "c"
    )

    manifest = json.loads(client.objects["data/corpus/dataset_manifest.json"])
    assert manifest["active_enrichment_version"] == "enrich-v1"
    assert report["previous_version"] == "enrich-v2"
    assert client.uploaded.count("data/corpus/dataset_manifest.json") == 1
    unchanged = {key for key in before if not key.endswith("dataset_manifest.json")}
    assert unchanged <= set(client.objects)
    assert all(
        client.objects[key] == objects[key]
        for key in unchanged
        if key.startswith("data/versions/enrichment/")
    )


def test_validation_refuses_a_declared_artifact_that_was_never_uploaded(tmp_path) -> None:
    objects = _bucket()
    del objects[
        "data/versions/enrichment/enrich-v1/batch=1/collection=L21"
        "/metadata/objects/part-000.parquet"
    ]
    client = _FakeClient(objects)

    with pytest.raises(EnrichmentVersionError, match="part-000.parquet"):
        validate_enrichment_version(
            client=client, version="enrich-v1", work_root=tmp_path
        )


def test_validation_refuses_an_empty_artifact(tmp_path) -> None:
    objects = _bucket()
    objects[
        "data/versions/enrichment/enrich-v1/batch=1/collection=L21"
        "/task05a/objects.tar.gz"
    ] = b""
    client = _FakeClient(objects)

    with pytest.raises(EnrichmentVersionError, match="empty"):
        validate_enrichment_version(
            client=client, version="enrich-v1", work_root=tmp_path
        )


def test_validation_refuses_a_provider_with_no_vector_file(tmp_path) -> None:
    objects = _bucket()
    for key in [k for k in objects if "/embeddings/beit3/" in k and k.endswith(".npy")]:
        del objects[key]
    client = _FakeClient(objects)

    with pytest.raises(EnrichmentVersionError, match="beit3"):
        validate_enrichment_version(
            client=client, version="enrich-v1", work_root=tmp_path
        )


def test_validation_refuses_when_collection_counts_do_not_sum_to_the_run_total(
    tmp_path,
) -> None:
    objects = _bucket(collections=("L21", "L22"))
    ready_key = "data/versions/enrichment/enrich-v1/_READY.json"
    ready = json.loads(objects[ready_key])
    ready["keyframe_count"] += 1
    objects[ready_key] = json.dumps(ready).encode()
    client = _FakeClient(objects)

    with pytest.raises(EnrichmentVersionError, match="keyframe_count"):
        validate_enrichment_version(
            client=client, version="enrich-v1", work_root=tmp_path
        )


def test_validation_reports_every_problem_at_once(tmp_path) -> None:
    objects = _bucket()
    root = "data/versions/enrichment/enrich-v1/batch=1/collection=L21"
    del objects[f"{root}/metadata/asr/part-000.parquet"]
    objects[f"{root}/task05a/ocr.tar.gz"] = b""
    client = _FakeClient(objects)

    with pytest.raises(EnrichmentVersionError) as excinfo:
        validate_enrichment_version(
            client=client, version="enrich-v1", work_root=tmp_path
        )

    message = str(excinfo.value)
    assert "metadata/asr/part-000.parquet" in message
    assert "task05a/ocr.tar.gz" in message


def test_activation_refuses_a_version_that_fails_artifact_validation(tmp_path) -> None:
    objects = _bucket()
    del objects[
        "data/versions/enrichment/enrich-v1/batch=1/collection=L21"
        "/embeddings/perception_encoder/manifest.json"
    ]
    client = _FakeClient(objects)

    with pytest.raises(EnrichmentVersionError):
        activate_enrichment_version(
            client=client, version="enrich-v1", work_root=tmp_path
        )

    assert client.uploaded == []


def _rewrite_mappings(objects: dict, rows: list[dict]) -> None:
    """Replace every provider mapping in the bucket with the given rows."""
    for key in [k for k in objects if k.endswith("/feature_mapping.csv")]:
        objects[key] = _mapping_csv(rows)


def test_catalog_returns_one_entry_per_keyframe_carrying_every_provider_vector(
    tmp_path,
) -> None:
    client = _FakeClient(_bucket())

    catalog = read_enrichment_catalog(
        client=client, version="enrich-v1", work_root=tmp_path
    )

    assert len(catalog["entries"]) == 4
    entry = catalog["entries"][0]
    assert entry.video_id == "L21_V001"
    assert entry.collection_id == "L21"
    assert set(entry.vectors) == {"beit3", "metaclip2", "perception_encoder"}
    assert entry.vectors["beit3"] == ("L21_V001.npy", 0)


def test_catalog_identity_is_the_video_and_keyframe_pair(tmp_path) -> None:
    """Organizer ids repeat across videos; the pair must stay distinct."""
    objects = _bucket()
    shared = [
        {
            "keyframe_id": "034",
            "video_id": video_id,
            "feature_file": f"{video_id}.npy",
            "feature_index": "0",
            "frame_id": "10",
            "shot_id": "1",
            "timestamp": "1.0",
            "image_path": f"data/corpus/keyframes/L21/{video_id}/034.jpg",
        }
        for video_id in ("L21_V001", "L21_V002")
    ]
    _rewrite_mappings(objects, shared)
    ready_key = "data/versions/enrichment/enrich-v1/_READY.json"
    ready = json.loads(objects[ready_key])
    ready["keyframe_count"] = 2
    objects[ready_key] = json.dumps(ready).encode()
    marker_key = (
        "data/versions/enrichment/enrich-v1/batch=1/collection=L21/_SUCCESS.json"
    )
    marker = json.loads(objects[marker_key])
    marker["keyframe_count"] = 2
    objects[marker_key] = json.dumps(marker).encode()

    catalog = read_enrichment_catalog(
        client=_FakeClient(objects), version="enrich-v1", work_root=tmp_path
    )

    identities = {(e.video_id, e.keyframe_id) for e in catalog["entries"]}
    assert identities == {("L21_V001", "034"), ("L21_V002", "034")}
    assert len(catalog["entries"]) == 2


def test_catalog_refuses_a_keyframe_that_one_provider_never_embedded(tmp_path) -> None:
    objects = _bucket()
    key = (
        "data/versions/enrichment/enrich-v1/batch=1/collection=L21"
        "/embeddings/metaclip2/feature_mapping.csv"
    )
    rows = _mapping_rows("L21", 2, 4)
    objects[key] = _mapping_csv(rows[:-1])

    with pytest.raises(EnrichmentVersionError, match="metaclip2"):
        read_enrichment_catalog(
            client=_FakeClient(objects), version="enrich-v1", work_root=tmp_path
        )


def test_catalog_refuses_a_vector_file_the_mapping_declares_but_nobody_published(
    tmp_path,
) -> None:
    objects = _bucket()
    del objects[
        "data/versions/enrichment/enrich-v1/batch=1/collection=L21"
        "/embeddings/beit3/L21_V002.npy"
    ]

    with pytest.raises(EnrichmentVersionError, match="L21_V002.npy"):
        read_enrichment_catalog(
            client=_FakeClient(objects), version="enrich-v1", work_root=tmp_path
        )


def test_catalog_refuses_a_feature_index_that_is_not_a_row_number(tmp_path) -> None:
    objects = _bucket()
    rows = _mapping_rows("L21", 2, 4)
    rows[1]["feature_index"] = "not-a-number"
    _rewrite_mappings(objects, rows)

    with pytest.raises(EnrichmentVersionError, match="feature_index"):
        read_enrichment_catalog(
            client=_FakeClient(objects), version="enrich-v1", work_root=tmp_path
        )
