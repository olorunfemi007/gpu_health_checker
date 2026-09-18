#!/usr/bin/env python3
"""GPU node pre-flight health checker.

Runs a battery of checks against an NVIDIA GPU node -- the kind of checks
you want to pass BEFORE a scheduler hands the node to an expensive multi-GPU
job, not the kind you discover you needed after the job has been silently
degraded for six hours.

Checks: GPU temperature, thermal throttle state, ECC errors, NVLink state,
GPU memory, Fabric Manager, DCGM diagnostics, InfiniBand state, disk
capacity, CPU load.

Usage:
    python3 gpu_health_checker.py
    python3 gpu_health_checker.py --json
    python3 gpu_health_checker.py --checks gpu_temp,ecc,disk
    python3 gpu_health_checker.py --dcgm-run-level 2 --disk-path /mnt/scratch

Exit codes:
    0  every check PASSed or WARNed (or was SKIPped -- tool not installed)
    1  at least one check FAILed
    2  the checker itself crashed (bad args, unexpected exception)
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Callable


class Status(Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"
    SKIP = "SKIP"


@dataclass
class CheckResult:
    name: str
    status: Status
    detail: str
    duration_s: float = 0.0


# Thresholds -- tune these per GPU generation / fleet SLA. What's "hot" for
# an air-cooled A100 in a warm datacenter is not what's hot for a
# liquid-cooled H100. GPU_TEMP_WARN_C/FAIL_C are the generic fallback used
# for any GPU not in GPU_TEMP_THRESHOLDS_C below.
GPU_TEMP_WARN_C = 80
GPU_TEMP_FAIL_C = 90

# Per-model overrides, substring-matched against nvidia-smi's `name` field
# (so board variants like SXM/PCIe/NVL and memory-size suffixes all match on
# the family name). These are BEST-EFFORT numbers, not verified against real
# hardware -- confirm against the actual datasheet for your SKU. Confidence
# varies a lot by model, which is exactly why it's tracked per entry instead
# of presented as one uniform table:
#   - A100: 85C max operating temp is a widely-repeated datasheet figure
#     across SXM4/PCIe variants. Moderate-good confidence.
#   - H100: published max operating temps I've seen range ~83-90C across
#     SXM5/PCIe/NVL variants and sources -- real spread, not one clean
#     number. 88 is a deliberately conservative pick, not a citation.
#   - B200: Blackwell is new enough that there's no number here I'd stand
#     behind. `None` means "recognized model, no verified threshold yet" --
#     it falls back to the generic default and says so in the check output,
#     rather than silently guessing.
GPU_TEMP_THRESHOLDS_C: dict[str, tuple[int, int] | None] = {
    "A100": (75, 85),
    "H100": (78, 88),
    "B200": None,
}
GPU_MEM_USED_WARN_PCT = 90
DISK_FREE_WARN_PCT = 15
DISK_FREE_FAIL_PCT = 5
CPU_LOAD_PER_CORE_WARN = 1.5
CPU_LOAD_PER_CORE_FAIL = 3.0
DCGM_DIAG_DEFAULT_LEVEL = "1"  # 1=quick(~sec) 2=medium(~2m) 3=long(~30m+)
DCGM_DIAG_TIMEOUT_S = {"1": 60, "2": 180, "3": 2400}


def run_cmd(cmd: list[str], timeout: int = 30) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, check=False)


def binary_missing(name: str) -> bool:
    return shutil.which(name) is None


def query_nvidia_smi(fields: str, name: str) -> tuple[list[list[str]], CheckResult | None]:
    """Run `nvidia-smi --query-gpu=<fields>` and return parsed CSV rows.

    Returns (rows, None) on success, or ([], CheckResult) with the terminal
    result the caller should return immediately on failure -- including the
    "nvidia-smi runs fine but lists zero GPUs" case, which is a real failure
    mode (a GPU that's fallen off the PCIe bus just disappears) and not an
    empty-but-healthy result.
    """
    if binary_missing("nvidia-smi"):
        return [], CheckResult(name, Status.SKIP, "nvidia-smi not found on PATH")

    result = run_cmd(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"])
    if result.returncode != 0:
        return [], CheckResult(name, Status.FAIL, f"nvidia-smi failed: {result.stderr.strip()}")

    rows = [[cell.strip() for cell in line.split(",")] for line in result.stdout.strip().splitlines()]
    if not rows:
        return [], CheckResult(name, Status.FAIL, "nvidia-smi reported zero GPUs")
    return rows, None


def resolve_temp_thresholds(model: str) -> tuple[int, int, bool]:
    """Return (warn_c, fail_c, is_model_specific) for a GPU model name.

    Substring-matches `model` against GPU_TEMP_THRESHOLDS_C. Falls back to
    the generic GPU_TEMP_WARN_C/FAIL_C both when the model isn't in the
    table at all, and when it's in the table but mapped to None (a model
    that's recognized but has no verified threshold yet) -- both cases
    genuinely have no verified number, so both get the same honest fallback.
    """
    for key, thresholds in GPU_TEMP_THRESHOLDS_C.items():
        if key in model and thresholds is not None:
            return thresholds[0], thresholds[1], True
    return GPU_TEMP_WARN_C, GPU_TEMP_FAIL_C, False


# ---------------------------------------------------------------- checks --

def check_gpu_temperature() -> CheckResult:
    name = "gpu_temperature"
    rows, early = query_nvidia_smi("index,name,temperature.gpu", name)
    if early:
        return early

    worst = Status.PASS
    details = []
    for idx, model, temp in rows:
        temp_c = int(temp)
        warn_c, fail_c, is_model_specific = resolve_temp_thresholds(model)
        if temp_c >= fail_c:
            worst = Status.FAIL
        elif temp_c >= warn_c and worst != Status.FAIL:
            worst = Status.WARN
        basis = f"fail>={fail_c}C" if is_model_specific else f"fail>={fail_c}C,default-unverified-for-model"
        details.append(f"gpu{idx}={temp_c}C({model},{basis})")
    return CheckResult(name, worst, ", ".join(details))


def check_thermal_throttle() -> CheckResult:
    name = "thermal_throttle"
    # These two fields are read directly from the driver, not inferred from
    # a temperature number -- they answer "is this GPU's clock currently
    # being held back for heat" as reported by the hardware itself, the
    # same signal nvidia-smi -q shows under "Clocks Throttle Reasons" as
    # "HW Thermal Slowdown" / "SW Thermal Slowdown". Independent of, and a
    # stronger signal than, the threshold-based gpu_temp check above.
    rows, early = query_nvidia_smi(
        "index,clocks_throttle_reasons.hw_thermal_slowdown,clocks_throttle_reasons.sw_thermal_slowdown", name
    )
    if early:
        return early

    throttled = []
    details = []
    for idx, hw_thermal, sw_thermal in rows:
        reasons = [r for r, active in (("hw", hw_thermal), ("sw", sw_thermal)) if active == "Active"]
        if reasons:
            throttled.append(f"gpu{idx}({'+'.join(reasons)})")
        details.append(f"gpu{idx}=hw:{hw_thermal},sw:{sw_thermal}")

    if throttled:
        return CheckResult(name, Status.FAIL, f"actively thermal-throttling: {', '.join(throttled)}")
    return CheckResult(name, Status.PASS, ", ".join(details))


def check_ecc_errors() -> CheckResult:
    name = "ecc_errors"
    rows, early = query_nvidia_smi(
        "index,ecc.errors.corrected.aggregate.total,ecc.errors.uncorrected.aggregate.total", name
    )
    if early:
        return early

    worst = Status.PASS
    details = []
    for idx, corrected, uncorrected in rows:
        if corrected == "[N/A]" or uncorrected == "[N/A]":
            details.append(f"gpu{idx}=ecc_unsupported")
            continue
        corrected_n, uncorrected_n = int(corrected), int(uncorrected)
        # Corrected ECC errors are memory doing its job -- a low background
        # rate is normal (cosmic-ray-induced bit flips happen). Uncorrected
        # errors mean memory failed at its job and results downstream of
        # them cannot be trusted.
        if uncorrected_n > 0:
            worst = Status.FAIL
        elif corrected_n > 0 and worst != Status.FAIL:
            worst = Status.WARN
        details.append(f"gpu{idx}=corrected:{corrected_n},uncorrected:{uncorrected_n}")
    return CheckResult(name, worst, ", ".join(details))


def check_gpu_memory() -> CheckResult:
    name = "gpu_memory"
    rows, early = query_nvidia_smi("index,memory.used,memory.total", name)
    if early:
        return early

    worst = Status.PASS
    details = []
    for idx, used, total in rows:
        used_mb, total_mb = int(used), int(total)
        pct = (used_mb / total_mb * 100) if total_mb else 0.0
        # High utilization is only a WARN, never a FAIL: a busy, healthy GPU
        # is supposed to be full of memory. This check exists to catch
        # leaked/zombie allocations on a node that's supposed to be idle,
        # not to police normal training memory usage.
        if pct >= GPU_MEM_USED_WARN_PCT:
            worst = Status.WARN
        details.append(f"gpu{idx}={used_mb}/{total_mb}MiB({pct:.0f}%)")
    return CheckResult(name, worst, ", ".join(details))


def check_nvlink_state() -> CheckResult:
    name = "nvlink_state"
    if binary_missing("nvidia-smi"):
        return CheckResult(name, Status.SKIP, "nvidia-smi not found on PATH")

    result = run_cmd(["nvidia-smi", "nvlink", "--status"])
    if result.returncode != 0:
        return CheckResult(name, Status.SKIP, "NVLink not present or unsupported on this node")

    total_links, inactive_links = 0, 0
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if line.startswith("Link "):
            total_links += 1
            if "inactive" in line.lower():
                inactive_links += 1

    if total_links == 0:
        return CheckResult(name, Status.SKIP, "no NVLink links reported")
    if inactive_links > 0:
        return CheckResult(name, Status.FAIL, f"{inactive_links}/{total_links} NVLink links inactive")
    return CheckResult(name, Status.PASS, f"{total_links}/{total_links} NVLink links active")


def check_fabric_manager() -> CheckResult:
    name = "fabric_manager"
    # Fabric Manager only exists on NVSwitch boards (DGX/HGX-class systems).
    # If the daemon was never installed, there is no fabric for it to manage
    # on this node -- that's a SKIP, not a FAIL.
    if binary_missing("nv-fabricmanager"):
        return CheckResult(name, Status.SKIP, "Fabric Manager not installed (no NVSwitch on this node)")
    if binary_missing("systemctl"):
        return CheckResult(name, Status.SKIP, "systemctl unavailable; cannot check Fabric Manager service")

    result = run_cmd(["systemctl", "is-active", "nvidia-fabricmanager"])
    state = result.stdout.strip() or "unknown"
    if state == "active":
        return CheckResult(name, Status.PASS, "nvidia-fabricmanager service active")
    return CheckResult(name, Status.FAIL, f"nvidia-fabricmanager service is '{state}', expected 'active'")


def parse_dcgm_diag_json(report: dict) -> list[tuple[str, str, str]]:
    """Walk a `dcgmi diag -j` report and return (gpu_id, test_name, status) triples.

    DCGM's diagnostic JSON schema has changed across releases, so this
    accepts the shape rather than assuming a single rigid structure will
    always match. Returns [] if nothing recognizable is found -- the caller
    must treat that as "could not parse", not "everything passed".
    """
    diag = report.get("DCGM GPU Diagnostic") or report.get("DCGM Diagnostic") or report
    categories = diag.get("test_categories") if isinstance(diag, dict) else None
    if not isinstance(categories, list):
        return []

    triples = []
    for category in categories:
        if not isinstance(category, dict):
            continue
        for test in category.get("tests", []):
            if not isinstance(test, dict):
                continue
            test_name = test.get("name", "unknown_test")
            results = test.get("results")
            if isinstance(results, list) and results:
                for r in results:
                    if isinstance(r, dict) and "status" in r:
                        triples.append((str(r.get("gpu_id", "?")), test_name, str(r["status"])))
            elif "status" in test:
                triples.append((str(test.get("gpu_id", "?")), test_name, str(test["status"])))
    return triples


def check_dcgm_diagnostics(run_level: str) -> CheckResult:
    name = "dcgm_diagnostics"
    if binary_missing("dcgmi"):
        return CheckResult(name, Status.SKIP, "dcgmi not found on PATH (DCGM not installed)")

    timeout = DCGM_DIAG_TIMEOUT_S.get(run_level, 300)
    result = run_cmd(["dcgmi", "diag", "-r", run_level, "-j"], timeout=timeout)
    output = result.stdout
    if result.returncode != 0 and not output:
        return CheckResult(name, Status.FAIL, f"dcgmi diag failed to run: {result.stderr.strip()}")

    # dcgmi diag's process exit code does not reliably reflect whether any
    # individual test failed -- you have to read the report. Prefer -j's
    # structured JSON so a failure can be attributed to a specific test and
    # GPU; if a DCGM version emits a JSON shape this doesn't recognize (the
    # schema has changed across releases), fall back to a coarse text scan
    # of the same output instead of silently reporting PASS.
    triples: list[tuple[str, str, str]] = []
    try:
        triples = parse_dcgm_diag_json(json.loads(output))
    except (json.JSONDecodeError, AttributeError):
        triples = []

    if triples:
        failing = [f"gpu{gpu}:{test}" for gpu, test, status in triples if status.lower() == "fail"]
        warning = [f"gpu{gpu}:{test}" for gpu, test, status in triples if status.lower() == "warn"]
        if failing:
            return CheckResult(name, Status.FAIL, "; ".join(failing[:8]))
        if warning:
            return CheckResult(name, Status.WARN, "; ".join(warning[:8]))
        return CheckResult(name, Status.PASS, f"dcgmi diag -r {run_level}: {len(triples)} tests passed")

    fail_lines = [l.strip() for l in output.splitlines() if re.search(r"\bFail\b", l)]
    if fail_lines:
        return CheckResult(name, Status.FAIL, "; ".join(fail_lines[:5]))
    warn_lines = [l.strip() for l in output.splitlines() if re.search(r"\bWarn\b", l)]
    if warn_lines:
        return CheckResult(name, Status.WARN, "; ".join(warn_lines[:5]))
    return CheckResult(name, Status.PASS, f"dcgmi diag -r {run_level} passed")


def check_infiniband_state() -> CheckResult:
    name = "infiniband_state"
    if binary_missing("ibstat"):
        return CheckResult(name, Status.SKIP, "ibstat not found (no InfiniBand stack installed)")

    result = run_cmd(["ibstat"])
    if result.returncode != 0:
        return CheckResult(name, Status.FAIL, f"ibstat failed: {result.stderr.strip()}")
    if not result.stdout.strip():
        return CheckResult(name, Status.SKIP, "no InfiniBand HCAs reported")

    down_ports = []
    current_ca, current_port = None, None
    for raw_line in result.stdout.splitlines():
        line = raw_line.strip()
        if line.startswith("CA '"):
            current_ca = line.split("'")[1]
        elif line.startswith("Port "):
            current_port = line.split()[1].rstrip(":")
        elif line.startswith("State:") and current_port:
            state = line.split(":", 1)[1].strip()
            if state != "Active":
                down_ports.append(f"{current_ca}/port{current_port}={state}")

    if down_ports:
        return CheckResult(name, Status.FAIL, "; ".join(down_ports))
    return CheckResult(name, Status.PASS, "all IB ports Active")


def check_disk_capacity(path: str) -> CheckResult:
    name = "disk_capacity"
    try:
        total, _used, free = shutil.disk_usage(path)
    except OSError as e:
        return CheckResult(name, Status.FAIL, f"could not stat {path}: {e}")

    free_pct = (free / total * 100) if total else 0.0
    detail = f"{free / 1e9:.1f}GB free of {total / 1e9:.1f}GB ({free_pct:.1f}% free) on {path}"
    if free_pct < DISK_FREE_FAIL_PCT:
        return CheckResult(name, Status.FAIL, detail)
    if free_pct < DISK_FREE_WARN_PCT:
        return CheckResult(name, Status.WARN, detail)
    return CheckResult(name, Status.PASS, detail)


def check_cpu_load() -> CheckResult:
    name = "cpu_load"
    try:
        load_1, load_5, load_15 = os.getloadavg()
    except (OSError, AttributeError):
        return CheckResult(name, Status.SKIP, "getloadavg unsupported on this platform")

    cpu_count = os.cpu_count() or 1
    per_core = load_1 / cpu_count
    detail = f"load avg 1m/5m/15m = {load_1:.2f}/{load_5:.2f}/{load_15:.2f} ({cpu_count} cores, {per_core:.2f}/core)"
    if per_core >= CPU_LOAD_PER_CORE_FAIL:
        return CheckResult(name, Status.FAIL, detail)
    if per_core >= CPU_LOAD_PER_CORE_WARN:
        return CheckResult(name, Status.WARN, detail)
    return CheckResult(name, Status.PASS, detail)


CHECKS: dict[str, Callable[[], CheckResult]] = {
    "gpu_temp": check_gpu_temperature,
    "throttle": check_thermal_throttle,
    "ecc": check_ecc_errors,
    "gpu_memory": check_gpu_memory,
    "nvlink": check_nvlink_state,
    "fabric_manager": check_fabric_manager,
    "dcgm": lambda: check_dcgm_diagnostics(DCGM_DIAG_DEFAULT_LEVEL),
    "infiniband": check_infiniband_state,
    "disk": lambda: check_disk_capacity("/"),
    "cpu_load": check_cpu_load,
}


def build_registry(dcgm_run_level: str, disk_path: str) -> dict[str, Callable[[], CheckResult]]:
    registry = dict(CHECKS)
    registry["dcgm"] = lambda: check_dcgm_diagnostics(dcgm_run_level)
    registry["disk"] = lambda: check_disk_capacity(disk_path)
    return registry


def run_all_checks(registry: dict[str, Callable[[], CheckResult]], names: list[str]) -> list[CheckResult]:
    results = []
    for check_name in names:
        start = time.monotonic()
        try:
            result = registry[check_name]()
        except subprocess.TimeoutExpired:
            result = CheckResult(check_name, Status.FAIL, "check timed out")
        except Exception as e:  # a single check crashing shouldn't crash the run
            result = CheckResult(check_name, Status.FAIL, f"checker error: {e}")
        result.duration_s = round(time.monotonic() - start, 2)
        results.append(result)
    return results


def print_report(results: list[CheckResult]) -> None:
    name_width = max(len(r.name) for r in results)
    for r in results:
        print(f"[{r.status.value:<4}] {r.name:<{name_width}}  {r.detail}  ({r.duration_s}s)")
    fails = sum(1 for r in results if r.status == Status.FAIL)
    warns = sum(1 for r in results if r.status == Status.WARN)
    check_word = "check" if len(results) == 1 else "checks"
    print(f"\nsummary: {len(results)} {check_word}, {fails} FAIL, {warns} WARN")


def main() -> int:
    parser = argparse.ArgumentParser(description="NVIDIA GPU node pre-flight health checker")
    parser.add_argument("--json", action="store_true", help="emit JSON instead of a text report")
    parser.add_argument(
        "--checks", type=str, default=None,
        help=f"comma-separated subset of checks to run: {','.join(CHECKS)}",
    )
    parser.add_argument(
        "--dcgm-run-level", type=str, default=DCGM_DIAG_DEFAULT_LEVEL, choices=["1", "2", "3"],
        help="dcgmi diag -r level: 1=quick(~sec) 2=medium(~2m) 3=long(~30m+)",
    )
    parser.add_argument(
        "--disk-path", type=str, default="/",
        help="path to check free space on (point this at your checkpoint/scratch volume)",
    )
    args = parser.parse_args()

    selected = [c.strip() for c in args.checks.split(",")] if args.checks else list(CHECKS.keys())
    unknown = set(selected) - set(CHECKS)
    if unknown:
        parser.error(f"unknown check(s): {', '.join(sorted(unknown))}")

    try:
        registry = build_registry(args.dcgm_run_level, args.disk_path)
        results = run_all_checks(registry, selected)

        if args.json:
            payload = {
                "timestamp_unix": time.time(),
                "checks": [{**asdict(r), "status": r.status.value} for r in results],
            }
            print(json.dumps(payload, indent=2))
        else:
            print_report(results)

        return 1 if any(r.status == Status.FAIL for r in results) else 0
    except Exception as e:
        print(f"gpu_health_checker: unexpected error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
