#!/bin/zsh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
MLX="${MLX_PREFIX:-/Users/dcazares/sglang/.venv-mps/lib/python3.11/site-packages/mlx}"
OUT="${1:-$ROOT/libqwen38_engine.dylib}"
clang++ -std=c++20 -O3 -fPIC -shared \
  -I"$MLX/include" \
  -I"$ROOT" \
  -L"$MLX/lib" \
  -Wl,-rpath,"$MLX/lib" \
  -lmlx \
  -o "$OUT" \
  "$ROOT/qwen38_engine.cpp" \
  "$ROOT/qwen38_c_api.cpp"
echo "built $OUT"
