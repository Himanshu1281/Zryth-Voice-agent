#!/usr/bin/env bash
# ==============================================================================
# update.sh — safely update the live voice agent on the VPS to a branch.
# ------------------------------------------------------------------------------
#   sudo bash /opt/voice-agent/deploy/update.sh              # dev_test2
#   sudo bash /opt/voice-agent/deploy/update.sh main
#   sudo bash /opt/voice-agent/deploy/update.sh --dry-run    # check only, change nothing
#
# Safe by design:
#   1. The new commit is checked in a separate folder first (compile + imports +
#      unit tests if pytest is installed). If that fails, the live code is untouched.
#   2. It waits until no call is in progress before restarting (never drops a call).
#   3. If the service isn't healthy 20 s after the restart, it rolls back to the
#      previous commit and restarts that.
# Needs /opt/voice-agent to be a git clone with no local edits.
# ==============================================================================
set -euo pipefail

APP_DIR="${APP_DIR:-/opt/voice-agent}"
SERVICE="voice-agent"
PY="${PY:-${APP_DIR}/.venv/bin/python}"
CHECK_DIR="${CHECK_DIR:-/tmp/voice-agent-check}"
MAX_WAIT_S=1800   # give up waiting for calls to end after 30 min

DRY_RUN=0
BRANCH="dev_test2"
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    *) BRANCH="$arg" ;;
  esac
done

say()  { echo -e "\n\033[1;36m==>\033[0m $*"; }
fail() { echo -e "\n\033[1;31mABORT:\033[0m $*  (live system unchanged)"; exit 1; }

cd "$APP_DIR"
[ -d .git ] || fail "$APP_DIR is not a git clone"
git diff --quiet && git diff --cached --quiet || fail "local edits in $APP_DIR; commit or stash them first"

OLD=$(git rev-parse HEAD)
say "Live commit: $(git log --oneline -1)"

say "Fetching origin/$BRANCH…"
git fetch -q origin "$BRANCH"
NEW=$(git rev-parse "origin/$BRANCH")
if [ "$OLD" = "$NEW" ]; then
  say "Already on $(git log --oneline -1 "$NEW"). Nothing to do."
  exit 0
fi
echo "New commit:  $(git log --oneline -1 "$NEW")"
echo "Changes:"
git log --oneline "$OLD..$NEW" | sed 's/^/  /'
if ! git diff --quiet "$OLD" "$NEW" -- requirements.txt; then
  fail "requirements.txt changed: install the new packages first (pip install -r requirements.txt), then rerun"
fi

# --- 1. Check the new code in a separate folder ---------------------------------
say "Checking the new code in $CHECK_DIR (live code not touched)…"
git worktree remove --force "$CHECK_DIR" 2>/dev/null || rm -rf "$CHECK_DIR"
git worktree add -q --detach "$CHECK_DIR" "$NEW"
cleanup() { git -C "$APP_DIR" worktree remove --force "$CHECK_DIR" 2>/dev/null || true; }
trap cleanup EXIT
# Each step chained with &&: inside "( ... ) ||" bash ignores set -e, so a failed
# compile would otherwise be skipped over.
(
  cd "$CHECK_DIR" &&
  ln -s "$APP_DIR/.env" .env &&
  "$PY" -m compileall -q ./*.py deploy scripts tests &&
  "$PY" -c "import agent, maya, tools, vendor, contact_flow, intents, replies, prompts" &&
  if "$PY" -c "import pytest" 2>/dev/null; then
    CI=1 "$PY" -m pytest tests/unit -q -p no:cacheprovider
  else
    echo "  (pytest not installed on this server: skipped unit tests; run them locally before pushing)"
  fi
) || fail "the new commit failed its checks"
echo "  New code OK."

if [ "$DRY_RUN" = 1 ]; then
  say "Dry run: checks passed. Nothing was changed."
  exit 0
fi

# --- 2. Wait until no call is in progress ---------------------------------------
say "Waiting until no call is in progress…"
waited=0
while :; do
  # From the checked-out new commit: the live (old) code may not have this helper yet.
  n=$("$PY" "$CHECK_DIR/deploy/active_calls.py" | tail -1)
  [ "$n" = "-1" ] && fail "couldn't reach Supabase to check for calls in progress"
  [ "$n" = "0" ] && break
  [ "$waited" -ge "$MAX_WAIT_S" ] && fail "calls still in progress after ${MAX_WAIT_S}s; try again later"
  echo "  $n call(s) in progress, checking again in 15 s…"
  sleep 15; waited=$((waited + 15))
done

# --- 3. Switch, restart, verify; roll back on failure ---------------------------
healthy() {
  sleep 20
  systemctl is-active --quiet "$SERVICE" || return 1
  # A crash loop shows up as several starts in the last 20 s.
  starts=$(journalctl -u "$SERVICE" --since "-20s" -q | grep -c "Started " || true)
  [ "$starts" -le 1 ]
}

say "Switching live code to $(git log --oneline -1 "$NEW") and restarting…"
git checkout -q --detach "$NEW"
systemctl restart "$SERVICE"
if healthy; then
  say "Done. Live: $(git log --oneline -1)"
  echo "  Rollback if ever needed:  cd $APP_DIR && git checkout --detach $OLD && systemctl restart $SERVICE"
  exit 0
fi

echo -e "\n\033[1;31mService unhealthy after restart; rolling back to $OLD\033[0m"
journalctl -u "$SERVICE" --since "-60s" -q | tail -30
git checkout -q --detach "$OLD"
systemctl restart "$SERVICE"
if healthy; then
  say "Rolled back. Live again: $(git log --oneline -1)"
else
  echo -e "\033[1;31mRollback also unhealthy: check  journalctl -u $SERVICE -f\033[0m"
fi
exit 1
