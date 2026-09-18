# GPU Health Checker

A pre-flight health check for NVIDIA GPU nodes. Run it before a scheduler
(Slurm prolog, Kubernetes init container, cron) hands a node to a job, so a
degraded node gets caught before it silently wastes a training run instead
of after.

## Checks

| Check | Source | Fails on |
|---|---|---|
| `gpu_temp` | `nvidia-smi` | per-model limit for A100/H100; 90°C generic default otherwise (incl. B200 — see below) |
| `throttle` | `nvidia-smi` | driver reports active HW/SW thermal slowdown (not inferred from temperature) |
| `ecc` | `nvidia-smi` | any uncorrected ECC error (WARN on corrected) |
| `gpu_memory` | `nvidia-smi` | never fails — WARN only, ≥ 90% used |
| `nvlink` | `nvidia-smi nvlink --status` | any inactive link |
| `fabric_manager` | `systemctl` | service not active (NVSwitch systems only) |
| `dcgm` | `dcgmi diag -j` | any test reporting Fail/Warn, attributed to a specific GPU |
| `infiniband` | `ibstat` | any port not `Active` |
| `disk` | stdlib | free space < 5% (WARN < 15%) |
| `cpu_load` | stdlib | load/core ≥ 3.0 (WARN ≥ 1.5) |

Any check whose underlying tool isn't installed reports `SKIP`, not `FAIL` —
a node with no InfiniBand HCA shouldn't fail its health check over it.

`gpu_temp` thresholds are best-effort, not verified against real hardware —
confirm against your actual datasheet before trusting them. Confidence
varies by model: A100's 85°C is a widely-repeated figure (moderate-good
confidence); H100's 88°C is a conservative pick from a range of ~83-90°C
cited across sources (moderate); B200 has no number here at all — it's new
enough that nothing here was worth standing behind, so it uses the generic
90°C default and the check's output says so explicitly (`default-unverified-
for-model`) rather than presenting a guess as fact. Any other GPU model
falls back to that same default.

## Usage

```
python3 gpu_health_checker.py                                  # text report
python3 gpu_health_checker.py --json                            # machine-readable
python3 gpu_health_checker.py --checks gpu_temp,ecc,disk         # subset
python3 gpu_health_checker.py --dcgm-run-level 2 --disk-path /mnt/scratch
```

**Exit codes:** `0` all PASS/WARN/SKIP · `1` at least one FAIL · `2` the checker itself crashed (bad args, unexpected exception).

## Requirements

Python 3.7+, stdlib only — no pip install. Each check degrades gracefully
if its tool is missing:

- `nvidia-smi` — GPU temp/ECC/memory/NVLink
- `dcgmi` (NVIDIA DCGM) — diagnostics
- `ibstat` (OFED/rdma-core) — InfiniBand
- `systemctl` + `nv-fabricmanager` — Fabric Manager (NVSwitch systems only)

