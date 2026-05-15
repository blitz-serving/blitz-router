#!/usr/bin/env bash
set -euo pipefail

AE_ROOT="${AE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
BLITZ_ROUTER_BRANCH="${BLITZ_ROUTER_BRANCH:-osdi26-ae-workflow}"
METRICS_TEST_RUNNER_BRANCH="${METRICS_TEST_RUNNER_BRANCH:-ae}"
YAULLM_BRANCH="${YAULLM_BRANCH:-lmetric/step-reporter-v2}"
XMETRIC_PLOTS_BRANCH="${XMETRIC_PLOTS_BRANCH:-ae}"

clone_or_update() {
  local url="$1"
  local dir="$2"
  local branch="$3"
  local recursive="${4:-false}"

  if [ -d "$AE_ROOT/$dir/.git" ]; then
    local current
    current="$(git -C "$AE_ROOT/$dir" branch --show-current || true)"
    if [ "$current" = "$branch" ]; then
      echo "[ae] $dir already exists on expected branch $branch"
    else
      echo "[ae] $dir already exists on branch '${current:-detached}', expected '$branch'"
      echo "[ae] leaving current checkout untouched; switch manually after saving local changes if needed"
    fi
    return 0
  fi

  if [ "$recursive" = "true" ]; then
    git clone --recursive --branch "$branch" "$url" "$AE_ROOT/$dir"
  else
    git clone --branch "$branch" "$url" "$AE_ROOT/$dir"
  fi
}

echo "[ae] AE_ROOT=$AE_ROOT"
mkdir -p "$AE_ROOT"

clone_or_update "git@github.com:blitz-serving/blitz-router.git" "blitz-router" "$BLITZ_ROUTER_BRANCH" "true"
clone_or_update "git@github.com:blitz-serving/MetricsTestRunner.git" "MetricsTestRunner" "$METRICS_TEST_RUNNER_BRANCH" "false"
clone_or_update "git@github.com:blitz-serving/yaullm.git" "yaullm" "$YAULLM_BRANCH" "false"
clone_or_update "git@github.com:blitz-serving/xmetric-plots.git" "xmetric-plots" "$XMETRIC_PLOTS_BRANCH" "false"

if [ -d "$AE_ROOT/blitz-router/.git" ]; then
  git -C "$AE_ROOT/blitz-router" submodule update --init --recursive
fi

if command -v git-lfs >/dev/null 2>&1; then
  for repo in MetricsTestRunner xmetric-plots; do
    if [ -d "$AE_ROOT/$repo/.git" ]; then
      git -C "$AE_ROOT/$repo" lfs install --local >/dev/null 2>&1 || true
      git -C "$AE_ROOT/$repo" lfs pull || true
    fi
  done
else
  echo "[ae] git-lfs is not installed; large archived data files may remain as LFS pointers"
fi

cat <<EOF
[ae] Repository preparation complete.

Expected tree:
  $AE_ROOT/blitz-router          router + request-sim submodule; branch $BLITZ_ROUTER_BRANCH
  $AE_ROOT/MetricsTestRunner     experiment orchestration scripts; branch $METRICS_TEST_RUNNER_BRANCH
  $AE_ROOT/yaullm                patched vLLM engine; branch $YAULLM_BRANCH
  $AE_ROOT/xmetric-plots         AE plotting scripts; branch $XMETRIC_PLOTS_BRANCH
EOF
