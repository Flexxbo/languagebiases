import glob
import os
import pickle
import re
from statistics import mean

BASE = "outputs_rank_optimizer"

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

    rate = len(positions) / len(runs) if runs else 0
    avg_pos = mean(positions) if positions else None
    mrr = mean([(1 / p) for p in positions] + [0] * (len(runs) - len(positions))) if runs else 0

    return {
        "runs": len(runs),
        "recommended": len(positions),
        "rate": rate,
        "avg_pos": avg_pos,
        "mrr": mrr,
    }

def extract_target_idx(path):
    match = re.search(r"_(\d+)\.pickle$", path)
    return int(match.group(1)) if match else None

def print_metrics(label, path, target_idx):
    runs = load_pickle(path)
    m = metrics_for_target(runs, target_idx)

    avg_pos = "N/A" if m["avg_pos"] is None else f"{m['avg_pos']:.2f}"

    print(
        f"{label:25s} target={target_idx} "
        f"runs={m['runs']} "
        f"recommended={m['recommended']} "
        f"rate={m['rate']:.2f} "
        f"avg_pos={avg_pos} "
        f"mrr={m['mrr']:.3f}"
    )

# Control: one control file can be used as baseline for target 0 and 1
control_files = glob.glob(os.path.join(BASE, "control_attack_baseline", "*.pickle"))
if control_files:
    control_path = control_files[0]
    for target_idx in [0, 1]:
        print_metrics("control", control_path, target_idx)

for attack in ["social_proof_baseline", "social_proof"]:
    files = sorted(glob.glob(os.path.join(BASE, attack, "*.pickle")))
    for path in files:
        target_idx = extract_target_idx(path)
        print_metrics(attack, path, target_idx)