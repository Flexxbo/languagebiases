import csv
import os
import pickle
from datetime import datetime
from statistics import mean

BASE = "outputs_rank_optimizer"
MODEL = "llama3.1-8b"
CATALOG = "coffee_machines"
QUERY_TYPE = "abstract"

EXPORT_DIR = "csv_exports"

ATTACKS = [
    "social_proof_baseline",
    "social_proof",
    # später optional:
    # "exclusivity_baseline",
    # "exclusivity",
]


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def metrics_for_target(runs, target_idx):
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


def get_path(attack, target_idx):
    return os.path.join(
        BASE,
        attack,
        f"experiment_{CATALOG}_{QUERY_TYPE}_{MODEL}_{attack}_{target_idx}.pickle"
    )


def fmt_pos(x):
    return "N/A" if x is None else f"{x:.2f}"


def delta_pos(control_pos, attack_pos):
    if control_pos is None or attack_pos is None:
        return None
    return attack_pos - control_pos


def get_export_path():
    os.makedirs(EXPORT_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    export_file = f"deltas_{CATALOG}_{QUERY_TYPE}_{MODEL}_{timestamp}.csv"

    return os.path.join(EXPORT_DIR, export_file)


def main():
    export_path = get_export_path()
    csv_rows = []

    print("\nPER-PRODUCT DELTAS AGAINST CONTROL")
    print("=" * 120)
    print(
        f"{'attack':25s} {'target':>6s} {'runs':>6s} "
        f"{'c_rate':>8s} {'a_rate':>8s} {'d_rate':>8s} "
        f"{'c_pos':>8s} {'a_pos':>8s} {'d_pos':>8s} "
        f"{'c_mrr':>8s} {'a_mrr':>8s} {'d_mrr':>8s}"
    )

    aggregate = {}

    for attack in ATTACKS:
        aggregate[attack] = {
            "delta_rate": [],
            "delta_mrr": [],
            "delta_pos": [],
            "rate_up": 0,
            "mrr_up": 0,
            "pos_up": 0,
            "n": 0,
        }

        for target_idx in range(10):
            control_path = get_path("control_attack_baseline", target_idx)
            attack_path = get_path(attack, target_idx)

            if not os.path.exists(control_path):
                # fallback: if your control run only produced _0 file, use that
                control_path = get_path("control_attack_baseline", 0)

            if not os.path.exists(control_path) or not os.path.exists(attack_path):
                print(f"Missing file for attack={attack}, target={target_idx}")
                continue

            control_runs = load_pickle(control_path)
            attack_runs = load_pickle(attack_path)

            c = metrics_for_target(control_runs, target_idx)
            a = metrics_for_target(attack_runs, target_idx)

            d_rate = a["rate"] - c["rate"]
            d_mrr = a["mrr"] - c["mrr"]
            d_pos = delta_pos(c["avg_pos"], a["avg_pos"])

            aggregate[attack]["delta_rate"].append(d_rate)
            aggregate[attack]["delta_mrr"].append(d_mrr)

            if d_pos is not None:
                aggregate[attack]["delta_pos"].append(d_pos)

            aggregate[attack]["rate_up"] += int(d_rate > 0)
            aggregate[attack]["mrr_up"] += int(d_mrr > 0)
            aggregate[attack]["pos_up"] += int(d_pos is not None and d_pos < 0)
            aggregate[attack]["n"] += 1

            print(
                f"{attack:25s} {target_idx:6d} {a['runs']:6d} "
                f"{c['rate']:8.2f} {a['rate']:8.2f} {d_rate:8.2f} "
                f"{fmt_pos(c['avg_pos']):>8s} {fmt_pos(a['avg_pos']):>8s} "
                f"{('N/A' if d_pos is None else f'{d_pos:.2f}'):>8s} "
                f"{c['mrr']:8.3f} {a['mrr']:8.3f} {d_mrr:8.3f}"
            )

            csv_rows.append({
                "row_type": "per_product",
                "catalog": CATALOG,
                "query_type": QUERY_TYPE,
                "model": MODEL,
                "attack": attack,
                "target_idx": target_idx,
                "runs": a["runs"],
                "control_recommended": c["recommended"],
                "attack_recommended": a["recommended"],
                "control_rate": c["rate"],
                "attack_rate": a["rate"],
                "delta_rate": d_rate,
                "control_avg_pos": c["avg_pos"],
                "attack_avg_pos": a["avg_pos"],
                "delta_pos": d_pos,
                "control_mrr": c["mrr"],
                "attack_mrr": a["mrr"],
                "delta_mrr": d_mrr,
                "rate_up": d_rate > 0,
                "pos_better": d_pos is not None and d_pos < 0,
                "mrr_up": d_mrr > 0,
            })

    print("\nAGGREGATE SUMMARY")
    print("=" * 120)
    print(
        f"{'attack':25s} {'n':>4s} {'mean_d_rate':>12s} "
        f"{'mean_d_pos':>12s} {'mean_d_mrr':>12s} "
        f"{'rate_up':>8s} {'pos_better':>11s} {'mrr_up':>8s}"
    )

    for attack, vals in aggregate.items():
        n = vals["n"]
        mean_d_rate = mean(vals["delta_rate"]) if vals["delta_rate"] else 0
        mean_d_mrr = mean(vals["delta_mrr"]) if vals["delta_mrr"] else 0
        mean_d_pos = mean(vals["delta_pos"]) if vals["delta_pos"] else None

        print(
            f"{attack:25s} {n:4d} {mean_d_rate:12.3f} "
            f"{('N/A' if mean_d_pos is None else f'{mean_d_pos:.3f}'):>12s} "
            f"{mean_d_mrr:12.3f} "
            f"{vals['rate_up']:8d} {vals['pos_up']:11d} {vals['mrr_up']:8d}"
        )

        csv_rows.append({
            "row_type": "aggregate",
            "catalog": CATALOG,
            "query_type": QUERY_TYPE,
            "model": MODEL,
            "attack": attack,
            "target_idx": "",
            "runs": n,
            "control_recommended": "",
            "attack_recommended": "",
            "control_rate": "",
            "attack_rate": "",
            "delta_rate": mean_d_rate,
            "control_avg_pos": "",
            "attack_avg_pos": "",
            "delta_pos": mean_d_pos,
            "control_mrr": "",
            "attack_mrr": "",
            "delta_mrr": mean_d_mrr,
            "rate_up": vals["rate_up"],
            "pos_better": vals["pos_up"],
            "mrr_up": vals["mrr_up"],
        })

    fieldnames = [
        "row_type",
        "catalog",
        "query_type",
        "model",
        "attack",
        "target_idx",
        "runs",
        "control_recommended",
        "attack_recommended",
        "control_rate",
        "attack_rate",
        "delta_rate",
        "control_avg_pos",
        "attack_avg_pos",
        "delta_pos",
        "control_mrr",
        "attack_mrr",
        "delta_mrr",
        "rate_up",
        "pos_better",
        "mrr_up",
    ]

    with open(export_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\nCSV export written to: {export_path}")


if __name__ == "__main__":
    main()