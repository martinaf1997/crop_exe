# -*- coding: utf-8 -*-
"""
QA of the QA - generates plan variants with simulated delivery errors.

Reads a root folder that holds one subfolder per patient. Each subfolder must
contain exactly one RT Plan DICOM file. For every plan the program writes the
15 error variants and their truncation logs back into the same subfolder.

Usage:
    qa_of_the_qa.exe "D:\\study\\patients"
    qa_of_the_qa.exe            (asks for the folder)
"""

import argparse
import csv
import math
import re
import sys
import traceback
from pathlib import Path

import pydicom
from pydicom.uid import generate_uid

# ---------------------------------------------------------------- constants

MIN_GAP = 0.5      # minimum gap between opposing leaves (Eclipse, MLC_HD)
SPAN = 150.0       # maximum leaf span within one bank
N = 60             # leaf pairs
LEAF_MIN_B = -201.0   # bank B travel limits
LEAF_MAX_B = 200.0
LEAF_MIN_A = -200.0   # bank A travel limits
LEAF_MAX_A = 201.0

RT_PLAN_SOP_CLASS = "1.2.840.10008.5.1.4.1.1.481.5"
VARIANT_PATTERN = re.compile(r"_E\d+$", re.IGNORECASE)
LABELS = [f"E{i}" for i in range(1, 16)]

# ---------------------------------------------------------------- helpers


def new_collimator_angle(value, delta):
    """Rotate the collimator, wrapping around 0-359.9."""
    return round((float(value) + delta) % 360, 1)


def enforce_span(pos, moved, beam, icp, log):
    """Pull the shifted leaves back within SPAN and record every truncation.
    moved = indices 0-119 actually modified in this control point."""
    for off in (0, N):
        bank = pos[off:off + N]
        moved_here = [i for i in moved if off <= i < off + N]
        if not moved_here or max(bank) - min(bank) <= SPAN:
            continue
        label = "B" if off == 0 else "A"
        # push back the leaves that are retracted too far
        lower = math.ceil((max(pos[off:off + N]) - SPAN) * 10 - 1e-6) / 10
        for i in moved_here:
            if pos[i] < lower:
                log.append((beam, icp, i - off + 1, label, "span", pos[i], lower))
                pos[i] = lower
        # pull back the leaves that are extended too far
        upper = math.floor((min(pos[off:off + N]) + SPAN) * 10 + 1e-6) / 10
        for i in moved_here:
            if pos[i] > upper:
                log.append((beam, icp, i - off + 1, label, "span", pos[i], upper))
                pos[i] = upper
    return pos


def save_log(log, out_dir, stem, label):
    """Write the truncation log next to the variant."""
    path = out_dir / f"{stem}_{label}_log.csv"
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["beam", "control_point", "leaf_pair", "bank", "reason",
                    "nominal", "applied"])
        w.writerows(log)
    return len(log)


def save_variant(ds, out_dir, stem, label):
    """Assign fresh UIDs, set the label and write the variant."""
    new_uid = generate_uid()
    ds.SOPInstanceUID = new_uid
    ds.file_meta.MediaStorageSOPInstanceUID = new_uid
    ds.SeriesInstanceUID = generate_uid()
    ds.RTPlanLabel = label
    ds.save_as(out_dir / f"{stem}_{label}.dcm")


def get_mlc(control_point):
    """Return the MLCX device of a control point."""
    mlc = control_point.BeamLimitingDevicePositionSequence[2]
    assert mlc.RTBeamLimitingDeviceType == "MLCX", \
        f"expected MLCX at index 2, found {mlc.RTBeamLimitingDeviceType}"
    return mlc


def shift_banks(ds, shift):
    """Move both banks away from the axis by `shift` mm (negative = closing)."""
    for b in ds.BeamSequence:
        for c in b.ControlPointSequence:
            mlc = get_mlc(c)
            for p in enumerate(mlc.LeafJawPositions):
                if p[0] < N:
                    new_p = round(p[1] - shift, 1)
                    mlc.LeafJawPositions[p[0]] = max(new_p, LEAF_MIN_B)
                else:
                    new_p = round(p[1] + shift, 1)
                    mlc.LeafJawPositions[p[0]] = min(new_p, LEAF_MAX_A)


def close_banks(ds, closure, log):
    """Move both banks towards the axis by `closure` mm, honouring MIN_GAP."""
    for b in ds.BeamSequence:
        for icp, c in enumerate(b.ControlPointSequence):
            mlc = get_mlc(c)
            pos = [float(x) for x in mlc.LeafJawPositions]
            moved = []
            for i in range(N):
                x1 = pos[i]        # bank B (X1), indices 0-59
                x2 = pos[i + N]    # bank A (X2), indices 60-119
                gap = round(x2 - x1, 1)
                if gap <= MIN_GAP:
                    continue
                if gap - 2 * closure >= MIN_GAP:
                    dx = closure
                else:
                    dx = math.floor((gap - MIN_GAP) / 2 * 10 + 1e-6) / 10
                    log.append((b.BeamName, icp, i + 1, "B+A", "gap", closure, dx))
                if dx == 0:
                    continue
                pos[i] = round(x1 + dx, 1)
                pos[i + N] = round(x2 - dx, 1)
                moved += [i, i + N]
            pos = enforce_span(pos, moved, b.BeamName, icp, log)
            mlc.LeafJawPositions = pos


def retract_central(ds, retract, log):
    """Retract the leaves of the central 1 cm (pairs 29-32) by `retract` mm."""
    central_b = [28, 29, 30, 31]
    central_a = [i + N for i in central_b]
    for b in ds.BeamSequence:
        for icp, c in enumerate(b.ControlPointSequence):
            mlc = get_mlc(c)
            pos = [float(x) for x in mlc.LeafJawPositions]
            for i in central_b:
                pos[i] = max(round(pos[i] - retract, 1), LEAF_MIN_B)
            for i in central_a:
                pos[i] = min(round(pos[i] + retract, 1), LEAF_MAX_A)
            pos = enforce_span(pos, central_b + central_a, b.BeamName, icp, log)
            mlc.LeafJawPositions = pos


def rotate_collimator(ds, delta):
    """Rotate the collimator of every control point that carries an angle."""
    for b in ds.BeamSequence:
        for c in b.ControlPointSequence:
            if "BeamLimitingDeviceAngle" in c:
                c.BeamLimitingDeviceAngle = new_collimator_angle(
                    c.BeamLimitingDeviceAngle, delta)


def scale_output(ds, factor):
    """Scale the delivered MU of every beam."""
    for fg in ds.FractionGroupSequence:
        for r in fg.ReferencedBeamSequence:
            r.BeamMeterset = f"{float(r.BeamMeterset) * factor:.5f}"

# ---------------------------------------------------------------- variants


def process_plan(plan_path, out_dir):
    """Build the 15 variants of one plan. Returns {label: truncations}."""
    stem = plan_path.stem
    counts = {}

    # E1 - collimator angle -1 deg
    ds = pydicom.dcmread(plan_path)
    rotate_collimator(ds, -1)
    save_variant(ds, out_dir, stem, "E1")
    counts["E1"] = 0

    # E2 - collimator angle +1 deg
    ds = pydicom.dcmread(plan_path)
    rotate_collimator(ds, +1)
    save_variant(ds, out_dir, stem, "E2")
    counts["E2"] = 0

    # E3 - MLC banks retracted by 0.3 mm
    ds = pydicom.dcmread(plan_path)
    shift_banks(ds, 0.3)
    save_variant(ds, out_dir, stem, "E3")
    counts["E3"] = 0

    # E4 - MLC banks retracted by 0.5 mm
    ds = pydicom.dcmread(plan_path)
    shift_banks(ds, 0.5)
    save_variant(ds, out_dir, stem, "E4")
    counts["E4"] = 0

    # E5 - MLC banks retracted by 1.0 mm
    ds = pydicom.dcmread(plan_path)
    shift_banks(ds, 1.0)
    save_variant(ds, out_dir, stem, "E5")
    counts["E5"] = 0

    # E6 - MLC banks retracted by 1.5 mm
    ds = pydicom.dcmread(plan_path)
    shift_banks(ds, 1.5)
    save_variant(ds, out_dir, stem, "E6")
    counts["E6"] = 0

    # E7 - MLC banks closed by 0.3 mm
    ds = pydicom.dcmread(plan_path)
    log = []
    close_banks(ds, 0.3, log)
    counts["E7"] = save_log(log, out_dir, stem, "E7")
    save_variant(ds, out_dir, stem, "E7")

    # E8 - MLC banks closed by 0.5 mm
    ds = pydicom.dcmread(plan_path)
    log = []
    close_banks(ds, 0.5, log)
    counts["E8"] = save_log(log, out_dir, stem, "E8")
    save_variant(ds, out_dir, stem, "E8")

    # E9 - MLC banks closed by 1.0 mm
    ds = pydicom.dcmread(plan_path)
    log = []
    close_banks(ds, 1.0, log)
    counts["E9"] = save_log(log, out_dir, stem, "E9")
    save_variant(ds, out_dir, stem, "E9")

    # E10 - central 1 cm of MLC leaves retracted by 2 mm
    ds = pydicom.dcmread(plan_path)
    log = []
    retract_central(ds, 2, log)
    counts["E10"] = save_log(log, out_dir, stem, "E10")
    save_variant(ds, out_dir, stem, "E10")

    # E11 - central 1 cm of MLC leaves retracted by 5 mm
    ds = pydicom.dcmread(plan_path)
    log = []
    retract_central(ds, 5, log)
    counts["E11"] = save_log(log, out_dir, stem, "E11")
    save_variant(ds, out_dir, stem, "E11")

    # E12 - output +2%
    ds = pydicom.dcmread(plan_path)
    scale_output(ds, 1.02)
    save_variant(ds, out_dir, stem, "E12")
    counts["E12"] = 0

    # E13 - output -2%
    ds = pydicom.dcmread(plan_path)
    scale_output(ds, 0.98)
    save_variant(ds, out_dir, stem, "E13")
    counts["E13"] = 0

    # E14 - output +2%, collimator +1 deg, MLC banks retracted by 1.0 mm
    ds = pydicom.dcmread(plan_path)
    rotate_collimator(ds, +1)
    scale_output(ds, 1.02)
    shift_banks(ds, 1.0)
    save_variant(ds, out_dir, stem, "E14")
    counts["E14"] = 0

    # E15 - output -2%, collimator -1 deg, MLC banks closed by 1.0 mm
    ds = pydicom.dcmread(plan_path)
    log = []
    rotate_collimator(ds, -1)
    scale_output(ds, 0.98)
    close_banks(ds, 1.0, log)
    counts["E15"] = save_log(log, out_dir, stem, "E15")
    save_variant(ds, out_dir, stem, "E15")

    return counts

# ---------------------------------------------------------------- folders


def find_plan(folder):
    """Return the single source RT Plan of a patient folder.
    Variants written by an earlier run are ignored."""
    candidates = []
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.suffix.lower() != ".dcm":
            continue
        if VARIANT_PATTERN.search(path.stem):
            continue
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True,
                                 specific_tags=["SOPClassUID"])
        except Exception:
            continue
        if str(ds.get("SOPClassUID", "")) == RT_PLAN_SOP_CLASS:
            candidates.append(path)
    if not candidates:
        raise FileNotFoundError("no RT Plan file found")
    if len(candidates) > 1:
        raise ValueError("more than one RT Plan: "
                         + ", ".join(p.name for p in candidates))
    return candidates[0]


def run(root):
    """Process every patient subfolder of `root`."""
    folders = sorted(p for p in root.iterdir() if p.is_dir())
    if not folders:
        print(f"No patient subfolder found in {root}")
        return 1

    summary = []
    n_ok = n_fail = 0
    for folder in folders:
        print(f"\n[{folder.name}]")
        try:
            plan = find_plan(folder)
        except Exception as exc:
            print(f"  SKIPPED: {exc}")
            summary.append([folder.name, "", "", "skipped", str(exc)])
            n_fail += 1
            continue

        print(f"  plan: {plan.name}")
        try:
            counts = process_plan(plan, folder)
        except Exception as exc:
            print(f"  FAILED: {exc}")
            traceback.print_exc()
            summary.append([folder.name, plan.name, "", "failed", str(exc)])
            n_fail += 1
            continue

        for label in LABELS:
            summary.append([folder.name, plan.name, label, "ok", counts[label]])
        truncated = {k: v for k, v in counts.items() if v}
        print(f"  15 variants written | truncations: {truncated or 'none'}")
        n_ok += 1

    out = root / "qa_summary.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["patient_folder", "plan_file", "variant", "status",
                    "truncations_or_error"])
        w.writerows(summary)

    print(f"\nDone: {n_ok} patients processed, {n_fail} skipped or failed.")
    print(f"Summary: {out}")
    return 0 if n_fail == 0 else 2


def main():
    parser = argparse.ArgumentParser(
        description="Generate RT Plan variants with simulated delivery errors.")
    parser.add_argument("root", nargs="?",
                        help="folder containing one subfolder per patient")
    args = parser.parse_args()

    interactive = args.root is None
    root = args.root
    if interactive:
        root = input("Folder containing the patient subfolders: ").strip().strip('"')

    code = 1
    try:
        path = Path(root).expanduser().resolve()
        if not path.is_dir():
            print(f"Not a folder: {path}")
        else:
            code = run(path)
    except Exception:
        traceback.print_exc()
    if interactive:
        input("\nPress Enter to close...")
    return code


if __name__ == "__main__":
    sys.exit(main())
