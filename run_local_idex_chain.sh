#!/usr/bin/env bash
# Run a local IDEX L1A -> L1B -> L2A processing chain.
#
# The script deliberately fails before processing if an input is not already
# present under DATA_ROOT. This prevents imap_cli from downloading a missing
# dependency while testing local products.

set -Eeuo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
DATA_ROOT=${IMAP_DATA_DIR:-"$SCRIPT_DIR/data/imap"}
START_DATE=${IDEX_START_DATE:-20260719}
LOCAL_MAJOR_VERSION=${IDEX_LOCAL_MAJOR_VERSION:-2}
LOCAL_MINOR_VERSION=${IDEX_LOCAL_MINOR_VERSION:-0}
WORK_DIR=${IDEX_LOCAL_WORK_DIR:-"$SCRIPT_DIR/.local_dependencies/$START_DATE"}

L1A_TEMPLATE=${L1A_TEMPLATE:-"$DATA_ROOT/dependency/idex/l1a/2026/07/imap_idex_l1a_all-70eae46c-eb87af55_20260719_v001.0004.json"}
L1B_TEMPLATE=${L1B_TEMPLATE:-"$DATA_ROOT/dependency/idex/l1b/2026/07/imap_idex_l1b_sci-10days-3d1c847d-5eded07e_20260719_v001.0009.json"}
L2A_TEMPLATE=${L2A_TEMPLATE:-"$DATA_ROOT/dependency/idex/l2a/2026/07/imap_idex_l2a_sci-10days-11542dff-5eded07e_20260719_v001.0009.json"}

L1A_DEPENDENCY="$WORK_DIR/l1a.json"
L1B_DEPENDENCY="$WORK_DIR/l1b.json"
L2A_DEPENDENCY="$WORK_DIR/l2a.json"
L1A_PRODUCT="$DATA_ROOT/idex/l1a/2026/07/imap_idex_l1a_sci-10days_${START_DATE}_v${LOCAL_MAJOR_VERSION}.$(printf '%04d' "$LOCAL_MINOR_VERSION").cdf"
L1B_PRODUCT="$DATA_ROOT/idex/l1b/2026/07/imap_idex_l1b_sci-10days_${START_DATE}_v${LOCAL_MAJOR_VERSION}.$(printf '%04d' "$LOCAL_MINOR_VERSION").cdf"

die() {
    echo "ERROR: $*" >&2
    exit 1
}

command -v imap_cli >/dev/null 2>&1 || die "imap_cli is not on PATH. Activate the IDEX environment first."
command -v python >/dev/null 2>&1 || die "python is not on PATH. Activate the IDEX environment first."

[[ -d "$DATA_ROOT" ]] || die "IMAP data directory does not exist: $DATA_ROOT"
[[ -f "$L1A_TEMPLATE" ]] || die "Missing L1A dependency template: $L1A_TEMPLATE"
[[ -f "$L1B_TEMPLATE" ]] || die "Missing L1B dependency template: $L1B_TEMPLATE"
[[ -f "$L2A_TEMPLATE" ]] || die "Missing L2A dependency template: $L2A_TEMPLATE"

mapfile -t TEMPLATE_FILES < <(
    python - "$L1A_TEMPLATE" "$L1B_TEMPLATE" "$L2A_TEMPLATE" <<'PY'
import json
import sys

for index, filename in enumerate(sys.argv[1:]):
    with open(filename) as stream:
        document = json.load(stream)
    for dependency in document.get("dependency", []):
        # L1B and L2A science inputs are replaced by the local predecessor.
        if index > 0 and dependency.get("type") == "science":
            continue
        for item in dependency.get("files", []):
            print(item)
PY
)

for filename in "${TEMPLATE_FILES[@]}"; do
    [[ -n "$filename" ]] || continue
    if [[ -z "$(find "$DATA_ROOT" -type f -name "$filename" -print -quit)" ]]; then
        die "Required local dependency is missing: $filename"
    fi
done

rm -rf -- "$WORK_DIR"
mkdir -p -- "$WORK_DIR"
cp -- "$L1A_TEMPLATE" "$WORK_DIR/template_l1a.json"
cp -- "$L1B_TEMPLATE" "$WORK_DIR/template_l1b.json"
cp -- "$L2A_TEMPLATE" "$WORK_DIR/template_l2a.json"
L1A_TEMPLATE="$WORK_DIR/template_l1a.json"
L1B_TEMPLATE="$WORK_DIR/template_l1b.json"
L2A_TEMPLATE="$WORK_DIR/template_l2a.json"

echo "Removing generated IDEX L1A/L1B/L2A products for $START_DATE"
for level in l1a l1b l2a; do
    level_dir="$DATA_ROOT/idex/$level"
    [[ -d "$level_dir" ]] && find "$level_dir" -type f -name "*${START_DATE}*" -delete
done

rewrite_dependency() {
    local template=$1
    local output=$2
    local replacement=${3:-}
    python - "$template" "$output" "$replacement" "$LOCAL_MAJOR_VERSION" "$LOCAL_MINOR_VERSION" <<'PY'
import json
import sys

template, output, replacement, major, minor = sys.argv[1:]
with open(template) as stream:
    document = json.load(stream)

if replacement:
    science_entries = [
        entry for entry in document.get("dependency", []) if entry.get("type") == "science"
    ]
    if len(science_entries) != 1:
        raise SystemExit(
            f"Expected exactly one science dependency in {template}; found {len(science_entries)}"
        )
    science_entries[0]["files"] = [replacement]

for versions in document.get("version", {}).values():
    versions["major_version"] = int(major)
    versions["minor_version"] = int(minor)

with open(output, "w") as stream:
    json.dump(document, stream, indent=2)
    stream.write("\n")
PY
}

rewrite_dependency "$L1A_TEMPLATE" "$L1A_DEPENDENCY"

echo "Processing L1A locally"
imap_cli --instrument idex --data-level l1a --descriptor all \
    --start-date "$START_DATE" --version v000 --dependency "$L1A_DEPENDENCY"

[[ -f "$L1A_PRODUCT" ]] || die "Expected L1A product was not created: $L1A_PRODUCT"

rewrite_dependency "$L1B_TEMPLATE" "$L1B_DEPENDENCY" "$(basename "$L1A_PRODUCT")"

echo "Processing L1B locally"
imap_cli --instrument idex --data-level l1b --descriptor sci-10days \
    --start-date "$START_DATE" --version v000 --dependency "$L1B_DEPENDENCY"

[[ -f "$L1B_PRODUCT" ]] || die "Expected L1B product was not created: $L1B_PRODUCT"

python - "$L1A_PRODUCT" "$L1B_PRODUCT" <<'PY'
import sys

import numpy as np
from cdflib import CDF

flag_names = (
    "science_event_flag",
    "noise_capture_flag",
    "pulser_flag",
    "dust_hit_flag",
)
l1a = CDF(sys.argv[1])
l1b = CDF(sys.argv[2])
for name in flag_names:
    if name not in l1a.cdf_info().zVariables or name not in l1b.cdf_info().zVariables:
        raise SystemExit(f"Missing {name} in L1A or L1B product")
    if not np.array_equal(l1a.varget(name), l1b.varget(name)):
        raise SystemExit(f"L1A/L1B values differ for {name}")
print("Verified all four event flags are preserved from L1A to L1B")
PY

rewrite_dependency "$L2A_TEMPLATE" "$L2A_DEPENDENCY" "$(basename "$L1B_PRODUCT")"

echo "Processing L2A locally"
imap_cli --instrument idex --data-level l2a --descriptor sci-10days \
    --start-date "$START_DATE" --version v000 --dependency "$L2A_DEPENDENCY"

echo "Local IDEX processing chain completed for $START_DATE"
echo "L1A: $L1A_PRODUCT"
echo "L1B: $L1B_PRODUCT"
echo "L2A: $DATA_ROOT/idex/l2a/2026/07/imap_idex_l2a_sci-10days_${START_DATE}_v${LOCAL_MAJOR_VERSION}.$(printf '%04d' "$LOCAL_MINOR_VERSION").cdf"
