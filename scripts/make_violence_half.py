"""Create a deterministic 50% subset of the violence training records."""

import argparse
import json
import random
from pathlib import Path


def is_violence(item):
    explicit_category = str(item.get("category", "")).strip().lower()
    if explicit_category:
        return explicit_category == "violence"
    image_path = Path(str(item.get("image_path") or item.get("image_id", "")))
    return any(part.lower() == "violence" for part in image_path.parts)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="dataset/all_train.json")
    parser.add_argument("--output", default="dataset/violence_half_train.json")
    parser.add_argument("--seed", type=int, default=233)
    args = parser.parse_args()

    with Path(args.input).open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError("The input dataset must be a JSON list")

    violence = [item for item in data if is_violence(item)]
    if not violence:
        raise ValueError(
            "No violence records found. Check the category field and image_path layout."
        )

    random.Random(args.seed).shuffle(violence)
    # Keep exactly floor(N / 2), but retain one record for a one-item category.
    selected = violence[: max(1, len(violence) // 2)]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(selected, handle, ensure_ascii=False, indent=2)

    print(f"violence records: {len(violence)}")
    print(f"selected records: {len(selected)}")
    print(f"saved to: {output_path}")


if __name__ == "__main__":
    main()
