#!/bin/zsh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$ROOT/../../../../../.." && pwd)"
if [[ -n "${MLX_PREFIX:-}" ]]; then
  MLX="$MLX_PREFIX"
else
  VENV="${VIRTUAL_ENV:-$REPO_ROOT/.venv}"
  candidates=("$VENV"/lib/python*/site-packages/mlx(N))
  if (( ${#candidates} != 1 )); then
    print -u2 -- "Expected one MLX installation under $VENV; set MLX_PREFIX explicitly."
    exit 1
  fi
  MLX="${candidates[1]}"
fi
if [[ ! -r "$MLX/include/mlx/array.h" || ! -r "$MLX/lib/libmlx.dylib" ]]; then
  print -u2 -- "MLX_PREFIX must contain include/mlx/array.h and lib/libmlx.dylib: $MLX"
  exit 1
fi
OUT="${1:-$ROOT/libqwen38_engine.dylib}"
clang++ -std=c++20 -O3 -fPIC -shared \
  -I"$MLX/include" -I"$ROOT" \
  -L"$MLX/lib" -Wl,-rpath,"$MLX/lib" -lmlx \
  -o "$OUT" "$ROOT/qwen38_engine.cpp" "$ROOT/qwen38_c_api.cpp"
echo "built $OUT"
