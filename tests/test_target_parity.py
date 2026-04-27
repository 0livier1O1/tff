import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# ── Configure these ──────────────────────────────────────────────────────────
RUN_A = ROOT / "artifacts" / "exp_synthetic_50s_500d_sgd_MABS"
RUN_B = ROOT / "artifacts" / "exp_synthetic_200s_5000d_sgd_BOSS"
# ─────────────────────────────────────────────────────────────────────────────

for run in [RUN_A, RUN_B]:
    if not run.exists():
        print(f"Directory not found: {run}")
        sys.exit(1)

seeds_a = {p.parent.name for p in RUN_A.glob("seed_*/target_tensor.npz")}
seeds_b = {p.parent.name for p in RUN_B.glob("seed_*/target_tensor.npz")}
shared  = sorted(seeds_a & seeds_b)

print(f"A: {RUN_A.name}")
print(f"B: {RUN_B.name}")
print(f"Shared seeds: {shared}\n")

for seed in shared:
    a = np.load(RUN_A / seed / "target_tensor.npz")["data"]
    b = np.load(RUN_B / seed / "target_tensor.npz")["data"]
    equal = a.shape == b.shape and np.array_equal(a, b)
    print(f"{seed}: shape {a.shape} vs {b.shape}  equal={equal}"
          + (f"  max|diff|={np.max(np.abs(a-b)):.2e}" if a.shape == b.shape else ""))
