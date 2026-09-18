# GPU Health Checker

A pre-flight health check for NVIDIA GPU nodes. Run it before a scheduler
(Slurm prolog, Kubernetes init container, cron) hands a node to a job, so a
degraded node gets caught before it silently wastes a training run instead
of after.

## Checks

| Check | Source | Fails on |
|---|---|---|
| `gpu_temp` | `nvidia-smi` | temp ≥ 90°C (WARN ≥ 80°C) |
| `ecc` | `nvidia-smi` | any uncorrected ECC error (WARN on corrected) |
| `gpu_memory` | `nvidia-smi` | never fails — WARN only, ≥ 90% used |
| `nvlink` | `nvidia-smi nvlink --status` | any inactive link |
| `fabric_manager` | `systemctl` | service not active (NVSwitch systems only) |
| `dcgm` | `dcgmi diag` | any Fail/Warn row in the report |
| `infiniband` | `ibstat` | any port not `Active` |
| `disk` | stdlib | free space < 5% (WARN < 15%) |
| `cpu_load` | stdlib | load/core ≥ 3.0 (WARN ≥ 1.5) |

Any check whose underlying tool isn't installed reports `SKIP`, not `FAIL` —
a node with no InfiniBand HCA shouldn't fail its health check over it.

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

