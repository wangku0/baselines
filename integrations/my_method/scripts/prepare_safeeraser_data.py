"""Build a paired subset matching SafeEraser's deterministic violence sample."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path


RETAIN_FIELDS = (
    "UnharmPair_text1",
    "UnharmPair_text2",
    "UnharmPair_image1",
    "UnharmPair_image2",
)


def load_list(path: Path) -> list[dict]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError(f"Expected a JSON list: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def core_unsafe_equal(left: dict, right: dict) -> bool:
    left_pairs = left.get("unsafe_pairs") or []
    right_pairs = right.get("unsafe_pairs") or []
    if len(left_pairs) != len(right_pairs):
        return False
    return all(
        a.get("question") == b.get("question")
        and a.get("model_response") == b.get("model_response")
        for a, b in zip(left_pairs, right_pairs)
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser()
    parser.add_argument("--safe-data", type=Path, default=repo_root / "dataset/all_train.json")
    parser.add_argument("--paired-data", type=Path, default=repo_root / "dataset/paired/all_train.json")
    parser.add_argument("--image-root", type=Path, default=repo_root / "dataset")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "integrations/my_method/outputs/data",
    )
    parser.add_argument("--seed", type=int, default=233)
    parser.add_argument("--records", type=int, default=50)
    args = parser.parse_args()

    safe = load_list(args.safe_data)
    paired = load_list(args.paired_data)
    violence = [
        row for row in safe
        if str(row.get("category", "")).strip().lower() == "violence"
    ]
    random.Random(args.seed).shuffle(violence)
    half = violence[: max(1, len(violence) // 2)]
    selected = half[: args.records]
    if len(selected) != args.records:
        raise ValueError(f"Requested {args.records} records, selected {len(selected)}")

    paired_by_id = {row.get("image_id"): row for row in paired}
    if len(paired_by_id) != len(paired):
        raise ValueError("Paired dataset contains duplicate image_id values")

    paired_selected = []
    errors = []
    for row in selected:
        image_id = row.get("image_id")
        match = paired_by_id.get(image_id)
        if match is None:
            errors.append(f"missing paired record: {image_id}")
            continue
        if len(row.get("unsafe_pairs") or []) != 4:
            errors.append(f"unsafe pair count is not 4: {image_id}")
        if len(match.get("safeNb_pairs") or []) != 4:
            errors.append(f"safeNb pair count is not 4: {image_id}")
        if not core_unsafe_equal(row, match):
            errors.append(f"unsafe question/response mismatch: {image_id}")
        if any(match.get(key) != row.get(key) for key in RETAIN_FIELDS):
            errors.append(f"retain fields mismatch: {image_id}")
        if not (args.image_root / str(image_id)).is_file():
            errors.append(f"missing image: {image_id}")
        paired_selected.append(match)
    if errors:
        raise ValueError("Dataset validation failed:\n- " + "\n- ".join(errors[:20]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paired_path = args.output_dir / f"violence_{args.records}_paired_train.json"
    eval_path = args.output_dir / f"violence_{args.records}_eval.json"
    ids_path = args.output_dir / f"violence_{args.records}_image_ids.json"
    paired_path.write_text(json.dumps(paired_selected, ensure_ascii=False, indent=2), encoding="utf-8")
    eval_path.write_text(json.dumps(selected, ensure_ascii=False, indent=2), encoding="utf-8")
    ids_path.write_text(
        json.dumps([row["image_id"] for row in selected], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    manifest = {
        "seed": args.seed,
        "violence_records": len(violence),
        "half_records": len(half),
        "selected_records": len(selected),
        "unsafe_pairs": sum(len(row["unsafe_pairs"]) for row in paired_selected),
        "safeNb_pairs": sum(len(row["safeNb_pairs"]) for row in paired_selected),
        "retain_pairs": len(selected) * len(RETAIN_FIELDS),
        "source_sha256": {
            "safe": sha256(args.safe_data),
            "paired": sha256(args.paired_data),
        },
        "outputs": {
            "paired_train": str(paired_path),
            "safeeraser_eval": str(eval_path),
            "image_ids": str(ids_path),
        },
    }
    manifest_path = args.output_dir / f"violence_{args.records}_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
