#!/usr/bin/env bash
# Sync analyzable_output/ and display_videos/ folders from the capture/processing
# PC to the analysis PC over the LAN, skipping raw video and intermediate steps.
#
# Run this FROM the capture/processing PC (push) so it can read source data
# directly off local disk. Safe to re-run repeatedly: rsync skips files that
# are already up to date on the destination, so only new/changed sessions
# take real time.
#
# By default syncs every session under SOURCE_ROOT. To sync only a subset
# (e.g. a date range or "EO3 and earlier" for a specific analysis task), pass
# SESSION_LIST pointing to a file with one session folder name per line —
# generate it with python_code/batch_processing/select_sessions.py, e.g.:
#
#   python -m python_code.batch_processing.select_sessions \
#     --day-label-at-or-before EO3 > /tmp/moseq_sessions.txt
#   SESSION_LIST=/tmp/moseq_sessions.txt REMOTE_HOST=analysis-pc \
#     ./scripts/sync_recordings_to_analysis_pc.sh
#
# Usage:
#   REMOTE_HOST=analysis-pc REMOTE_USER=scholl-lab ./scripts/sync_recordings_to_analysis_pc.sh
#
# One-time setup for passwordless, non-interactive runs:
#   ssh-copy-id "$REMOTE_USER@$REMOTE_HOST"

set -euo pipefail

SOURCE_ROOT="${SOURCE_ROOT:-/home/scholl-lab/ferret_recordings}"
REMOTE_USER="${REMOTE_USER:-scholab}"
REMOTE_HOST="${REMOTE_HOST:?Set REMOTE_HOST to the analysis PCs hostname or IP}"
REMOTE_ROOT="${REMOTE_ROOT:-/mnt/data/ferret_recordings}"

# Optional: path to a file with one session folder name per line (see header
# comment above). Leave unset to sync every session under SOURCE_ROOT.
SESSION_LIST="${SESSION_LIST:-}"

# Skip the preflight space check (e.g. if `rsync --stats` parsing ever breaks
# on a system with a different rsync version/locale) with SKIP_SPACE_CHECK=1.
SKIP_SPACE_CHECK="${SKIP_SPACE_CHECK:-0}"

# Gigabit LAN is the bottleneck, not CPU: skip -z (compression) since video is
# already compressed and compression would just burn CPU for no speedup.
# Set COMPRESS=1 if you're ever crossing a slower link (e.g. VPN, WiFi).
COMPRESS_FLAG=""
if [[ "${COMPRESS:-0}" == "1" ]]; then
  COMPRESS_FLAG="-z"
fi

# Filters shared between the dry-run size estimate and the real transfer, so
# they can never drift out of sync with each other.
FILTER_ARGS=()
FILTER_FILE=""
if [[ -n "$SESSION_LIST" ]]; then
  if [[ ! -f "$SESSION_LIST" ]]; then
    echo "ERROR: SESSION_LIST file not found: $SESSION_LIST" >&2
    exit 1
  fi
  FILTER_FILE="$(mktemp)"
  trap 'rm -f "$FILTER_FILE"' EXIT
  while IFS= read -r session_name; do
    [[ -z "$session_name" ]] && continue
    printf '+ /%s/\n' "$session_name" >> "$FILTER_FILE"
  done < "$SESSION_LIST"
  # Anchored: excludes any top-level session dir not explicitly included above,
  # before the general recursive rule below gets a chance to pull it in.
  printf -- '- /*/\n' >> "$FILTER_FILE"
  {
    printf '+ */\n'
    printf '+ **/analyzable_output/***\n'
    printf '+ **/display_videos/***\n'
    printf -- '- *\n'
  } >> "$FILTER_FILE"
  FILTER_ARGS=(--filter="merge $FILTER_FILE")
else
  FILTER_ARGS=(
    --include='*/'
    --include='**/analyzable_output/***'
    --include='**/display_videos/***'
    --exclude='*'
  )
fi

if [[ "$SKIP_SPACE_CHECK" != "1" ]]; then
  echo "Checking available space on ${REMOTE_HOST}..."

  ssh "${REMOTE_USER}@${REMOTE_HOST}" "mkdir -p '${REMOTE_ROOT}'"

  needed_bytes=$(
    rsync -an --stats "${FILTER_ARGS[@]}" -e ssh \
      "${SOURCE_ROOT%/}/" \
      "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_ROOT%/}/" \
    | grep 'Total transferred file size:' \
    | grep -oE '[0-9,]+' | head -1 | tr -d ','
  )

  available_bytes=$(
    ssh "${REMOTE_USER}@${REMOTE_HOST}" "df -B1 --output=avail '${REMOTE_ROOT}'" \
    | tail -1 | tr -d ' '
  )

  if [[ -z "$needed_bytes" || -z "$available_bytes" ]]; then
    echo "Warning: couldn't determine transfer size or free space; skipping preflight check." >&2
  else
    needed_gb=$(( needed_bytes / 1024 / 1024 / 1024 ))
    available_gb=$(( available_bytes / 1024 / 1024 / 1024 ))
    echo "Need ~${needed_gb} GB, ${available_gb} GB available on destination."

    # Require some headroom rather than cutting it exactly to the byte.
    if (( needed_bytes > available_bytes - 5*1024*1024*1024 )); then
      echo "ERROR: Not enough free space on ${REMOTE_HOST}:${REMOTE_ROOT} (need ~${needed_gb} GB + 5 GB headroom, have ${available_gb} GB). Aborting before transfer." >&2
      exit 1
    fi
  fi
fi

rsync -a $COMPRESS_FLAG \
  --info=progress2 \
  --partial \
  --prune-empty-dirs \
  "${FILTER_ARGS[@]}" \
  -e ssh \
  "${SOURCE_ROOT%/}/" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_ROOT%/}/"
