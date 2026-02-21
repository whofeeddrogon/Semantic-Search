"""
ingestion.py — CLI script to ingest JSON or CSV files into Qdrant.

Usage:
    python ingestion.py --file sample_data.json --text-field text
    python ingestion.py --file data.csv --text-field content
"""

import argparse
import json
import sys

import pandas as pd
from tqdm import tqdm

import config
import model_loader
import database_manager


def load_data(file_path: str, text_field: str) -> list[dict]:
    """Load records from a JSON or CSV file. Each record must have `text_field`."""
    if file_path.endswith(".json"):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            records = []
            for item in data:
                if text_field not in item:
                    raise ValueError(f"Field '{text_field}' not found in record.")
                records.append(item)
        else:
            raise ValueError("JSON file must contain a list of objects.")
    elif file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
        if text_field not in df.columns:
            raise ValueError(f"Column '{text_field}' not found. "
                             f"Available: {list(df.columns)}")
        records = df.dropna(subset=[text_field]).to_dict(orient="records")
    else:
        raise ValueError("Unsupported file type. Use .json or .csv")

    return records


def ingest(file_path: str, text_field: str, batch_size: int | None = None) -> int:
    """
    Full ingestion pipeline: load → encode → upsert.

    Returns the number of documents ingested.
    """
    bs = batch_size or config.BATCH_SIZE

    # 1 — Load
    print(f"[ingestion] Loading data from {file_path} (field='{text_field}')…")
    records = load_data(file_path, text_field)
    total = len(records)
    print(f"[ingestion] Found {total} documents.")

    # 2 — Ensure collection
    database_manager.ensure_collection()

    # 3 — Batch encode & upsert
    for start in tqdm(range(0, total, bs), desc="Ingesting"):
        batch_records = records[start : start + bs]
        batch_texts = [r[text_field] for r in batch_records]
        vectors = model_loader.encode(batch_texts, batch_size=bs)
        # Store ALL fields as payload (not just text)
        database_manager.upsert_batch(vectors, batch_records)

    count = database_manager.collection_count()
    print(f"[ingestion] Done. Collection now has {count} points.")
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest data into Qdrant")
    parser.add_argument("--file", required=True, help="Path to JSON or CSV file")
    parser.add_argument("--text-field", required=True,
                        help="Name of the text field/column")
    parser.add_argument("--batch-size", type=int, default=None,
                        help=f"Override default batch size ({config.BATCH_SIZE})")
    args = parser.parse_args()

    try:
        ingest(args.file, args.text_field, args.batch_size)
    except Exception as exc:
        print(f"[ingestion] ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
