from __future__ import annotations

import json
from pathlib import Path

import pytest

from mvr.operations.enrichment_version import (
    EnrichmentVersionError,
    activate_enrichment_version,
    apply_enrichment_pointer,
    resolve_active_enrichment_version,
)
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


def _version_objects(version: str, collections: tuple[str, ...]) -> dict[str, bytes]:
    prefix = f"data/versions/enrichment/{version}"
    objects = {
        f"{prefix}/_READY.json": json.dumps(
            {
                "status": "READY",
                "run_id": version,
                "batch_id": "batch1",
                "collections": list(collections),
            }
        ).encode()
    }
    for collection in collections:
        objects[f"{prefix}/batch=1/collection={collection}/_SUCCESS.json"] = (
            json.dumps({"status": "DONE"}).encode()
        )
        objects[f"{prefix}/batch=1/collection={collection}/task05a/objects.tar.gz"] = b"x"
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
