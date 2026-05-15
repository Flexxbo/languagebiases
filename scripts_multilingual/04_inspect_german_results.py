import csv
import pickle
from pathlib import Path

RAW_DIR = Path("results_multilingual/raw/de_coffee_machines_social_proof")
OUT_DIR = Path("results_multilingual/csv")
OUT_DIR.mkdir(parents=True, exist_ok=True)

rows = []

for path in RAW_DIR.rglob("*.pickle"):
    condition = path.parent.name

    with path.open("rb") as f:
        data = pickle.load(f)

    for run_idx, run in enumerate(data):
        source_file, products_shuffled, prompt, llm_response, alignment = run

        rows.append({
            "file": str(path),
            "condition": condition,
            "run_idx": run_idx,
            "n_products": len(products_shuffled),
            "alignment": str(alignment),
            "llm_response": llm_response,
            "prompt": prompt,
        })

out_path = OUT_DIR / "de_german_pilot_raw_review.csv"

with out_path.open("w", newline="", encoding="utf-8-sig") as f:
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "file",
            "condition",
            "run_idx",
            "n_products",
            "alignment",
            "llm_response",
            "prompt",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} rows to {out_path}")