import json
import csv
from pathlib import Path

def validate_ocr_output(mapping_csv: Path, output_dir: Path) -> bool:
    print(f"Validating OCR output in {output_dir}")

    # Load mapping, keyed on (video_id, keyframe_id).
    #
    # A keyframe_id alone does not identify a keyframe. The organizer's
    # 166,550 keyframes carry only 632 distinct ids -- "001" exists in ~850
    # videos -- so keying on the id alone reported a duplicate at row 309 of
    # the real mapping (L21_V001, then L21_V002) and refused to validate
    # anything. It is the same composite key the Elasticsearch _id and the
    # retrieval fusion already use.
    keyframe_mappings = {}
    with mapping_csv.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            identity = (row["video_id"], row["keyframe_id"])

            if identity in keyframe_mappings:
                video_id, keyframe_id = identity
                print(f"ERROR: Duplicate keyframe in mapping: {video_id}/{keyframe_id}")
                return False

            keyframe_mappings[identity] = {
                "video_id": row["video_id"],
                "shot_id": int(row["shot_id"]),
                "frame_idx": int(row["frame_idx"]),
                "timestamp": float(row["timestamp"]),
            }

    # Check manifest
    manifest_path = output_dir / "manifest.json"
    if not manifest_path.exists():
        print("ERROR: manifest.json missing")
        return False

    with manifest_path.open("r", encoding="utf-8") as f:
        manifest = json.load(f)

    json_count = 0

    for video_dir in output_dir.iterdir():
        if not video_dir.is_dir():
            continue

        video_id = video_dir.name

        for json_file in video_dir.glob("*.json"):
            json_count += 1
            keyframe_id = json_file.stem

            if (video_id, keyframe_id) not in keyframe_mappings:
                print(f"ERROR: Keyframe {video_id}/{keyframe_id} not in mapping CSV")
                return False

            expected = keyframe_mappings[(video_id, keyframe_id)]
            # The old "belongs to another video" check lived here because the
            # lookup was ambiguous. Under the composite key it cannot fail:
            # a JSON filed under the wrong video now misses the lookup above
            # instead, and says so naming both.

            with json_file.open("r", encoding="utf-8") as f:
                data = json.load(f)

            if data["keyframe_id"] != keyframe_id:
                print(f"ERROR: {json_file} has wrong keyframe_id")
                return False
            if data["video_id"] != video_id:
                print(f"ERROR: {json_file} has wrong video_id")
                return False
            if data["shot_id"] != expected["shot_id"]:
                print(f"ERROR: {json_file} has wrong shot_id")
                return False
            if data["anchor_frame_idx"] != expected["frame_idx"]:
                print(f"ERROR: {json_file} has wrong anchor_frame_idx")
                return False
            if abs(data["anchor_timestamp"] - expected["timestamp"]) > 1e-6:
                print(f"ERROR: {json_file} has wrong anchor_timestamp")
                return False
            for t in data.get("texts", []):
                if t.get("video_id") != expected["video_id"]:
                    print(f"ERROR: {json_file} text has wrong video_id")
                    return False
                if t.get("shot_id") != expected["shot_id"]:
                    print(f"ERROR: {json_file} text has wrong shot_id")
                    return False
                if t.get("frame_idx", -1) < 0:
                    print(f"ERROR: {json_file} text has invalid frame_idx")
                    return False
                if t.get("timestamp", -1) < 0:
                    print(f"ERROR: {json_file} text has invalid timestamp")
                    return False
                if not t.get("text"):
                    print(f"ERROR: {json_file} text has empty text")
                    return False
                if not (0 <= t.get("conf", -1) <= 1):
                    print(f"ERROR: {json_file} text has invalid conf {t.get('conf')}")
                    return False
                if not (0 <= t.get("det_conf", -1) <= 1):
                    print(f"ERROR: {json_file} text has invalid det_conf {t.get('det_conf')}")
                    return False
                if not (0 <= t.get("rec_conf", -1) <= 1):
                    print(f"ERROR: {json_file} text has invalid rec_conf {t.get('rec_conf')}")
                    return False
                if abs(t["conf"] - t["rec_conf"]) > 1e-6:
                    print(f"ERROR: {json_file} conf != rec_conf")
                    return False
                bbox = t.get("bbox", [])
                if (
                    len(bbox) != 4
                    or not all(isinstance(x, (int, float)) for x in bbox)
                ):
                    print(f"ERROR: {json_file} text has invalid bbox {bbox}")
                    return False

                x1, y1, x2, y2 = bbox
                if x1 > x2 or y1 > y2:
                    print(f"ERROR: {json_file} text has invalid bbox ordering {bbox}")
                    return False

                support_count = t.get("support_count", 0)
                if not isinstance(support_count, int) or not (1 <= support_count <= 3):
                    print(
                        f"ERROR: {json_file} text has invalid "
                        f"support_count {support_count}"
                    )
                    return False

    # Now a true row count. Keyed on keyframe_id alone this was the number of
    # *distinct ids* -- 632 for the organizer's 166,550 keyframes -- so the
    # completeness check was comparing against the wrong number as well.
    if json_count != len(keyframe_mappings):
        print(f"ERROR: JSON count {json_count} != canonical keyframe count {len(keyframe_mappings)}")
        return False

    print(f"SUCCESS: Validated {json_count} OCR JSON files.")
    return True

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mapping-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if not validate_ocr_output(args.mapping_csv, args.output_dir):
        exit(1)
