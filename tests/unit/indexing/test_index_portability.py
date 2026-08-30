"""An index that can leave the machine it was built on.

Two defects motivate this file, both found by reading the real 363 MB
`artifacts/indexes/qna_organizer_all`:

1. Every `image_path` in it is an absolute Windows path under `E:\\`, and
   `answer_queries` puts that column into the review sheet the operator
   opens by eye. A teammate who clones the repo resolves none of them, and
   the whole human-verification step dies on their machine.

2. The index stores `frame_id` (the keyframe's own id) but never
   `frame_idx` (the competition frame number), so every submission row is
   resolved through a separate `keyframe_mapping.csv`. Nothing binds the
   two files together: hand someone the index without that exact CSV and
   the run completes and emits nothing.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest


MAPPING_HEADER = "keyframe_id,video_id,shot_id,ordinal,frame_idx,timestamp,image_path\n"


def _mapping(tmp_path: Path, *rows: str, name: str = "keyframe_mapping.csv") -> Path:
    path = tmp_path / name
    path.write_text(MAPPING_HEADER + "".join(rows), encoding="utf-8")
    return path


# -- 1. image paths survive leaving the machine -------------------------


def test_a_path_under_the_image_root_is_stored_relative_to_it() -> None:
    """What gets written into metadata.json, and therefore what a
    teammate receives."""
    from mvr.indexing.faiss_index import relative_image_path

    stored = relative_image_path(
        Path(r"E:\AIC26_data\Keyframes_L21\keyframes\L21_V001\001.jpg"),
        Path(r"E:\AIC26_data"),
    )

    assert stored == "Keyframes_L21/keyframes/L21_V001/001.jpg"


def test_a_stored_path_uses_forward_slashes_so_linux_can_read_it() -> None:
    """The index is built on Windows and consumed on Kaggle, which is
    Linux. A backslash there is a filename character, not a separator."""
    from mvr.indexing.faiss_index import relative_image_path

    stored = relative_image_path(
        Path(r"E:\root\a\b\c.jpg"), Path(r"E:\root")
    )

    assert "\\" not in stored


def test_a_path_outside_the_image_root_is_refused_rather_than_stored() -> None:
    """Silently falling back to the absolute path would put one
    unresolvable row in an otherwise portable index -- the failure would
    surface later, on someone else's machine."""
    from mvr.indexing.faiss_index import relative_image_path

    with pytest.raises(ValueError, match="outside"):
        relative_image_path(Path(r"D:\elsewhere\001.jpg"), Path(r"E:\AIC26_data"))


def test_a_relative_path_resolves_against_the_root_recorded_at_build_time() -> None:
    from mvr.indexing.faiss_index import resolve_image_path

    resolved = resolve_image_path(
        {"image_path": "Keyframes_L21/keyframes/L21_V001/001.jpg"},
        {"image_root": r"E:\AIC26_data"},
    )

    assert resolved == Path(r"E:\AIC26_data") / "Keyframes_L21/keyframes/L21_V001/001.jpg"


def test_an_override_root_wins_so_the_index_works_on_another_machine() -> None:
    """This is the whole point: the same index, read on a machine where
    the corpus sits somewhere else entirely."""
    from mvr.indexing.faiss_index import resolve_image_path

    resolved = resolve_image_path(
        {"image_path": "Keyframes_L21/keyframes/L21_V001/001.jpg"},
        {"image_root": r"E:\AIC26_data"},
        image_root=Path("/kaggle/input/aic26"),
    )

    assert resolved == Path("/kaggle/input/aic26/Keyframes_L21/keyframes/L21_V001/001.jpg")


def test_an_index_built_before_this_change_keeps_its_absolute_paths() -> None:
    """The real 363 MB index has no image_root and absolute paths. It must
    keep working untouched -- rebuilding it is a four-minute job nobody
    should be forced into by a library change."""
    from mvr.indexing.faiss_index import resolve_image_path

    absolute = r"E:\AIC26_data\Keyframes_L21\keyframes\L21_V001\001.jpg"

    resolved = resolve_image_path({"image_path": absolute}, {})

    assert resolved == Path(absolute)


def test_an_override_cannot_rewrite_a_legacy_absolute_path() -> None:
    """Joining a root onto an absolute path yields nonsense. Better to
    hand back what was stored and let the caller see a real missing
    file than to invent a path that never existed."""
    from mvr.indexing.faiss_index import resolve_image_path

    absolute = r"E:\AIC26_data\Keyframes_L21\keyframes\L21_V001\001.jpg"

    resolved = resolve_image_path(
        {"image_path": absolute}, {}, image_root=Path("/kaggle/input/aic26")
    )

    assert resolved == Path(absolute)


# -- 2. the index is bound to the mapping it was built from -------------


def test_a_fingerprint_records_both_the_checksum_and_the_row_count() -> None:
    """The checksum decides; the row count is what makes the error
    message actionable to a person."""
    from mvr.indexing.faiss_index import mapping_fingerprint

    csv_path = _mapping(
        Path(pytest.importorskip("tempfile").mkdtemp()),
        "001,L21_V001,1,1,0,0.0,a.jpg\n",
        "002,L21_V001,2,1,90,3.6,b.jpg\n",
    )

    fingerprint = mapping_fingerprint(csv_path)

    assert set(fingerprint) == {"mapping_sha256", "mapping_row_count"}
    assert fingerprint["mapping_row_count"] == 2
    assert len(fingerprint["mapping_sha256"]) == 64


def test_the_mapping_the_index_was_built_from_verifies_clean(tmp_path: Path) -> None:
    from mvr.indexing.faiss_index import mapping_fingerprint, verify_mapping_binding

    csv_path = _mapping(tmp_path, "001,L21_V001,1,1,0,0.0,a.jpg\n")

    verify_mapping_binding(mapping_fingerprint(csv_path), csv_path)


def test_a_mapping_edited_after_the_index_was_built_is_rejected(tmp_path: Path) -> None:
    """The defect this exists to catch. Today the frame resolver simply
    finds nothing for the changed rows and drops them, so the run
    succeeds and the submission is short -- with no error anywhere."""
    from mvr.indexing.faiss_index import (
        IndexMappingMismatch,
        mapping_fingerprint,
        verify_mapping_binding,
    )

    csv_path = _mapping(tmp_path, "001,L21_V001,1,1,0,0.0,a.jpg\n")
    recorded = mapping_fingerprint(csv_path)
    _mapping(tmp_path, "001,L21_V001,1,1,9999,0.0,a.jpg\n")

    with pytest.raises(IndexMappingMismatch):
        verify_mapping_binding(recorded, csv_path)


def test_the_mismatch_message_names_both_row_counts(tmp_path: Path) -> None:
    """`checksum differs` sends a person nowhere. `built over 2 rows, this
    file has 1` tells them which file they grabbed."""
    from mvr.indexing.faiss_index import (
        IndexMappingMismatch,
        mapping_fingerprint,
        verify_mapping_binding,
    )

    built_from = _mapping(
        tmp_path,
        "001,L21_V001,1,1,0,0.0,a.jpg\n",
        "002,L21_V001,2,1,90,3.6,b.jpg\n",
        name="built.csv",
    )
    handed_over = _mapping(tmp_path, "001,L21_V001,1,1,0,0.0,a.jpg\n", name="other.csv")

    with pytest.raises(IndexMappingMismatch) as caught:
        verify_mapping_binding(mapping_fingerprint(built_from), handed_over)

    assert "2" in str(caught.value) and "1" in str(caught.value)


def test_an_index_with_no_recorded_mapping_warns_rather_than_blocking(
    tmp_path: Path,
) -> None:
    """Every index built so far lacks the binding. Raising would break
    them all; staying silent would hide that the check did not run."""
    from mvr.indexing.faiss_index import verify_mapping_binding

    csv_path = _mapping(tmp_path, "001,L21_V001,1,1,0,0.0,a.jpg\n")

    with pytest.warns(UserWarning, match="not bound"):
        verify_mapping_binding({}, csv_path)


def test_a_missing_mapping_file_is_an_error_not_a_warning(tmp_path: Path) -> None:
    from mvr.indexing.faiss_index import verify_mapping_binding

    with pytest.raises(FileNotFoundError):
        verify_mapping_binding({}, tmp_path / "absent.csv")


# -- the two halves meet on a real round trip ---------------------------


def test_a_portable_index_round_trips_through_disk(tmp_path: Path) -> None:
    """Both fields survive save/load, so a teammate reading the index gets
    the root it was built against and the mapping it is bound to."""
    pytest.importorskip("faiss")
    import numpy as np

    from mvr.indexing.faiss_index import (
        FaissIndex,
        mapping_fingerprint,
        resolve_image_path,
    )

    csv_path = _mapping(tmp_path, "001,L21_V001,1,1,0,0.0,a.jpg\n")
    index = FaissIndex.build(
        np.array([[1.0, 0.0]], dtype=np.float32),
        [
            {
                "frame_id": "001",
                "video_id": "L21_V001",
                "timestamp": 0.0,
                "image_path": "Keyframes_L21/keyframes/L21_V001/001.jpg",
            }
        ],
        manifest={
            "embedding_model": "m",
            "pretrained": "p",
            "image_root": r"E:\AIC26_data",
            **mapping_fingerprint(csv_path),
        },
    )
    index.save(tmp_path / "idx")

    loaded = FaissIndex.load(tmp_path / "idx")

    assert resolve_image_path(
        loaded.metadata[0], loaded.manifest, image_root=Path("/elsewhere")
    ) == Path("/elsewhere/Keyframes_L21/keyframes/L21_V001/001.jpg")
    assert loaded.manifest["mapping_sha256"] == mapping_fingerprint(csv_path)["mapping_sha256"]


def test_the_manifest_written_to_disk_is_still_plain_json(tmp_path: Path) -> None:
    """A Path object in the manifest would make save() raise. image_root
    arrives as a Path from the builders, so it must be text by then."""
    pytest.importorskip("faiss")
    import numpy as np

    from mvr.indexing.faiss_index import FaissIndex

    FaissIndex.build(
        np.array([[1.0, 0.0]], dtype=np.float32),
        [{"frame_id": "1", "video_id": "v", "timestamp": 0.0, "image_path": "a.jpg"}],
        manifest={"embedding_model": "m", "pretrained": "p", "image_root": Path("E:/x")},
    ).save(tmp_path)

    written = json.loads((tmp_path / "manifest.json").read_text(encoding="utf-8"))

    assert written["image_root"] == "E:/x"


# -- the check runs where the two files are first put together ----------


def _tiny_index(tmp_path: Path, mapping_csv: Path | None):
    """A one-vector index, optionally bound to a mapping."""
    import numpy as np

    from mvr.indexing.faiss_index import FaissIndex, mapping_fingerprint

    manifest: dict = {"embedding_model": "m", "pretrained": "p"}
    if mapping_csv is not None:
        manifest.update(mapping_fingerprint(mapping_csv))
    index = FaissIndex.build(
        np.array([[1.0, 0.0]], dtype=np.float32),
        [{"frame_id": "1", "video_id": "v", "timestamp": 0.0, "image_path": "a.jpg"}],
        manifest=manifest,
    )
    index.save(tmp_path / "idx")
    return tmp_path / "idx"


def test_loading_with_the_mapping_it_was_built_from_succeeds(tmp_path: Path) -> None:
    pytest.importorskip("faiss")
    from mvr.indexing.faiss_index import FaissIndex

    csv_path = _mapping(tmp_path, "001,L21_V001,1,1,0,0.0,a.jpg\n")
    index_dir = _tiny_index(tmp_path, csv_path)

    assert FaissIndex.load(index_dir, mapping_csv=csv_path).index.ntotal == 1


def test_loading_with_the_wrong_mapping_fails_at_load_not_at_submission(
    tmp_path: Path,
) -> None:
    """Caught here, it names the problem. Caught nowhere -- which is
    today -- the resolver drops the unmatched rows and the run writes a
    short submission that looks fine."""
    pytest.importorskip("faiss")
    from mvr.indexing.faiss_index import FaissIndex, IndexMappingMismatch

    built_from = _mapping(tmp_path, "001,L21_V001,1,1,0,0.0,a.jpg\n", name="built.csv")
    index_dir = _tiny_index(tmp_path, built_from)
    other = _mapping(
        tmp_path,
        "001,L21_V001,1,1,0,0.0,a.jpg\n",
        "002,L21_V002,1,1,7,0.3,b.jpg\n",
        name="other.csv",
    )

    with pytest.raises(IndexMappingMismatch):
        FaissIndex.load(index_dir, mapping_csv=other)


def test_loading_without_naming_a_mapping_stays_silent(tmp_path: Path) -> None:
    """`score_verified_set` and the API load the index alone. They must
    not start warning about a file they never asked for."""
    pytest.importorskip("faiss")
    from mvr.indexing.faiss_index import FaissIndex

    index_dir = _tiny_index(tmp_path, None)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert FaissIndex.load(index_dir).index.ntotal == 1
