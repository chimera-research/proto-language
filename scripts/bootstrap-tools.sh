#!/usr/bin/env bash
set -euo pipefail

language_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
tools_root="$language_root/../diablo-tools"
tools_revision=9ce3734a1e102fbe6c8757294507dfef235fd787

if [[ -e "$tools_root" || -L "$tools_root" ]]; then
  if [[ ! -d "$tools_root/.git" && ! -f "$tools_root/.git" ]]; then
    printf 'Expected an independent Git checkout at %s.\n' "$tools_root" >&2
    exit 1
  fi
  printf 'Using existing tools checkout at %s.\n' "$tools_root"
else
  git clone --branch codex/independent-diablo-packages https://github.com/chimera-research/proto-tools.git "$tools_root"
  git -C "$tools_root" checkout --detach "$tools_revision"
fi
