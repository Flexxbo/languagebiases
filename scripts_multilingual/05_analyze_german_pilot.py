import csv
import pickle
import re
from pathlib import Path
from statistics import mean


RAW_DIR = Path("results_multilingual/raw/de_coffee_machines_social_proof")
OUT_DIR = Path("results_multilingual/csv")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PER_PRODUCT_CSV = OUT_DIR / "de_german_pilot_deltas.csv"
AGGREGATE_CSV = OUT_DIR / "de_german_pilot_aggregate.csv"

N_TARGETS = 10

BASELINE_CONDITIONS = {
    "control_original",
    "neutral_paraphrase",
}

TARGET_CONDITIONS = {
    "social_proof_append_generated",
    "social_proof_rewrite_generated",
}

COMPARISONS = [
    {
        "comparison": "neutral_vs_control",
        "baseline_condition": "control_original",
        "attack_condition": "neutral_paraphrase",
    },
    {
        "comparison": "append_vs_control",
        "baseline_condition": "control_original",
        "attack_condition": "social_proof_append_generated",
    },
    {
        "comparison": "rewrite_vs_neutral",
        "baseline_condition": "neutral_paraphrase",
        "attack_condition": "social_proof_rewrite_generated",
    },
]


def load_pickle(path: Path):
    with path.open("rb") as f:
        return pickle.load(f)


def metrics_for_target(runs, target_idx: int) -> dict:
    positions = []

    for run in runs:
        alignment = run[4]
        hit_positions = [
            item["llm_output"] + 1
            for item in alignment
            if item["se_output"] == target_idx
        ]

        if hit_positions:
            positions.append(hit_positions[0])

    n = len(runs)
    recommended = len(positions)
    rate = recommended / n if n else 0
    avg_pos = mean(positions) if positions else None
    mrr = mean([(1 / p) for p in positions] + [0] * (n - recommended)) if n else 0

    return {
        "runs": n,
        "recommended": recommended,
        "rate": rate,
        "avg_pos": avg_pos,
        "mrr": mrr,
    }


def get_single_condition_file(condition: str) -> Path:
    condition_dir = RAW_DIR / condition
    files = sorted(condition_dir.glob("*.pickle"))

    if not files:
        raise FileNotFoundError(f"No pickle files found for condition: {condition}")

    return files[0]


def get_target_condition_file(condition: str, target_idx: int) -> Path:
    condition_dir = RAW_DIR / condition
    files = sorted(condition_dir.glob("*.pickle"))

    for path in files:
        match = re.search(r"_(\d+)\.pickle$", path.name)
        if match and int(match.group(1)) == target_idx:
            return path

    raise FileNotFoundError(f"No pickle file found for condition={condition}, target={target_idx}")


def get_runs_for_condition(condition: str, target_idx: int):
    if condition in BASELINE_CONDITIONS:
        return load_pickle(get_single_condition_file(condition))

    if condition in TARGET_CONDITIONS:
        return load_pickle(get_target_condition_file(condition, target_idx))

    raise ValueError(f"Unknown condition: {condition}")


def fmt_pos(x):
    return "N/A" if x is None else f"{x:.2f}"


def delta_pos(base_pos, attack_pos):
    if base_pos is None or attack_pos is None:
        return None
    return attack_pos - base_pos


def main():
    per_product_rows = []
    aggregate_rows = []

    print("\nGERMAN PILOT: PER-PRODUCT DELTAS")
    print("=" * 130)
    print(
        f"{'comparison':24s} {'target':>6s} {'runs':>6s} "
        f"{'b_rate':>8s} {'a_rate':>8s} {'d_rate':>8s} "
        f"{'b_pos':>8s} {'a_pos':>8s} {'d_pos':>8s} "
        f"{'b_mrr':>8s} {'a_mrr':>8s} {'d_mrr':>8s}"
    )

    for comp in COMPARISONS:
        comparison = comp["comparison"]
        baseline_condition = comp["baseline_condition"]
        attack_condition = comp["attack_condition"]

        delta_rates = []
        delta_mrrs = []
        delta_positions = []

        rate_up = 0
        mrr_up = 0
        pos_better = 0
        n = 0

        for target_idx in range(N_TARGETS):
            baseline_runs = get_runs_for_condition(baseline_condition, target_idx)
            attack_runs = get_runs_for_condition(attack_condition, target_idx)

            b = metrics_for_target(baseline_runs, target_idx)
            a = metrics_for_target(attack_runs, target_idx)

            d_rate = a["rate"] - b["rate"]
            d_mrr = a["mrr"] - b["mrr"]
            d_pos = delta_pos(b["avg_pos"], a["avg_pos"])

            delta_rates.append(d_rate)
            delta_mrrs.append(d_mrr)
            if d_pos is not None:
                delta_positions.append(d_pos)

            rate_up += int(d_rate > 0)
            mrr_up += int(d_mrr > 0)
            pos_better += int(d_pos is not None and d_pos < 0)
            n += 1

            print(
                f"{comparison:24s} {target_idx:6d} {a['runs']:6d} "
                f"{b['rate']:8.2f} {a['rate']:8.2f} {d_rate:8.2f} "
                f"{fmt_pos(b['avg_pos']):>8s} {fmt_pos(a['avg_pos']):>8s} "
                f"{('N/A' if d_pos is None else f'{d_pos:.2f}'):>8s} "
                f"{b['mrr']:8.3f} {a['mrr']:8.3f} {d_mrr:8.3f}"
            )

            per_product_rows.append({
                "comparison": comparison,
                "baseline_condition": baseline_condition,
                "attack_condition": attack_condition,
                "target_idx": target_idx,
                "runs": a["runs"],
                "baseline_recommended": b["recommended"],
                "attack_recommended": a["recommended"],
                "baseline_rate": b["rate"],
                "attack_rate": a["rate"],
                "delta_rate": d_rate,
                "baseline_avg_pos": b["avg_pos"],
                "attack_avg_pos": a["avg_pos"],
                "delta_pos": d_pos,
                "baseline_mrr": b["mrr"],
                "attack_mrr": a["mrr"],
                "delta_mrr": d_mrr,
                "rate_up": d_rate > 0,
                "pos_better": d_pos is not None and d_pos < 0,
                "mrr_up": d_mrr > 0,
            })

        mean_d_rate = mean(delta_rates) if delta_rates else 0
        mean_d_mrr = mean(delta_mrrs) if delta_mrrs else 0
        mean_d_pos = mean(delta_positions) if delta_positions else None

        aggregate_rows.append({
            "comparison": comparison,
            "baseline_condition": baseline_condition,
            "attack_condition": attack_condition,
            "n_targets": n,
            "mean_delta_rate": mean_d_rate,
            "mean_delta_pos": mean_d_pos,
            "mean_delta_mrr": mean_d_mrr,
            "rate_up": rate_up,
            "pos_better": pos_better,
            "mrr_up": mrr_up,
        })

    print("\nGERMAN PILOT: AGGREGATE SUMMARY")
    print("=" * 130)
    print(
        f"{'comparison':24s} {'n':>4s} {'mean_d_rate':>12s} "
        f"{'mean_d_pos':>12s} {'mean_d_mrr':>12s} "
        f"{'rate_up':>8s} {'pos_better':>11s} {'mrr_up':>8s}"
    )

    for row in aggregate_rows:
        mean_d_pos = row["mean_delta_pos"]
        print(
            f"{row['comparison']:24s} {row['n_targets']:4d} "
            f"{row['mean_delta_rate']:12.3f} "
            f"{('N/A' if mean_d_pos is None else f'{mean_d_pos:.3f}'):>12s} "
            f"{row['mean_delta_mrr']:12.3f} "
            f"{row['rate_up']:8d} {row['pos_better']:11d} {row['mrr_up']:8d}"
        )

    per_product_fields = [
        "comparison",
        "baseline_condition",
        "attack_condition",
        "target_idx",
        "runs",
        "baseline_recommended",
        "attack_recommended",
        "baseline_rate",
        "attack_rate",
        "delta_rate",
        "baseline_avg_pos",
        "attack_avg_pos",
        "delta_pos",
        "baseline_mrr",
        "attack_mrr",
        "delta_mrr",
        "rate_up",
        "pos_better",
        "mrr_up",
    ]

    with PER_PRODUCT_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=per_product_fields)
        writer.writeheader()
        writer.writerows(per_product_rows)

    aggregate_fields = [
        "comparison",
        "baseline_condition",
        "attack_condition",
        "n_targets",
        "mean_delta_rate",
        "mean_delta_pos",
        "mean_delta_mrr",
        "rate_up",
        "pos_better",
        "mrr_up",
    ]

    with AGGREGATE_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=aggregate_fields)
        writer.writeheader()
        writer.writerows(aggregate_rows)

    print(f"\nWrote per-product CSV: {PER_PRODUCT_CSV}")
    print(f"Wrote aggregate CSV:   {AGGREGATE_CSV}")


if __name__ == "__main__":
    main()