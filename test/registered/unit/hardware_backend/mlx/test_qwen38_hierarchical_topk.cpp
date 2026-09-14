#include "mlx/mlx.h"
#include "qwen38_engine.h"
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <vector>
namespace mx = mlx::core;
namespace native = sglang::mlx_qwen38;
void Require(bool value, const char *message) {
  if (!value)
    throw std::runtime_error(message);
}
void Check(int rows, int vocab, int k, bool tied, bool rank_three) {
  std::vector<float> values(static_cast<std::size_t>(rows) * vocab);
  std::uint32_t state = 42;
  for (auto &value : values) {
    state = state * 1664525U + 1013904223U;
    value = tied ? static_cast<float>(state % 7)
                 : static_cast<float>(state >> 8) / 65536.0f - 128.0f;
  }
  const mx::Shape shape =
      rank_three ? mx::Shape{1, rows, vocab} : mx::Shape{rows, vocab};
  const auto logits = mx::astype(mx::array(values.data(), shape, mx::float32),
                                 native::activation_dtype());
  setenv("SGLANG_MLX_NATIVE_HIERARCHICAL_TOPK", "0", 1);
  const auto control_ids = native::sampling_topk_indices(logits, k);
  const auto control_values =
      mx::sort(mx::take_along_axis(logits, control_ids, -1), -1);
  mx::eval(logits, control_values);
  for (const char *block : {"256", "512", "1024", "2048"}) {
    setenv("SGLANG_MLX_NATIVE_HIERARCHICAL_TOPK", block, 1);
    const auto ids = native::sampling_topk_indices(logits, k);
    auto expected_shape = shape;
    expected_shape.back() = k;
    Require(ids.shape() == expected_shape, "wrong top-k shape");
    const auto integer_ids = mx::astype(mx::reshape(ids, {rows, k}), mx::int32);
    const auto selected = mx::sort(mx::take_along_axis(logits, ids, -1), -1);
    const auto equal = mx::all(mx::equal(selected, control_values));
    mx::eval(integer_ids, equal);
    Require(equal.item<bool>(), "hierarchical top-k scores differ");
    const auto *indices = integer_ids.data<std::int32_t>();
    for (int row = 0; row < rows; ++row) {
      std::vector<int> selected_ids(indices + row * k, indices + (row + 1) * k);
      std::sort(selected_ids.begin(), selected_ids.end());
      Require(selected_ids.front() >= 0 && selected_ids.back() < vocab,
              "top-k index outside vocabulary");
      Require(std::adjacent_find(selected_ids.begin(), selected_ids.end()) ==
                  selected_ids.end(),
              "duplicate top-k index");
    }
  }
}
int main() {
  try {
    for (bool tied : {false, true}) {
      for (int rows : {1, 2, 3}) {
        Check(rows, 248320, 20, tied, true);
        Check(rows, 4096, 20, tied, false);
        Check(rows, 1024, 1, tied, true);
        Check(rows, 1000, 20, tied, false);
        Check(rows, 32, 20, tied, true);
      }
    }
    Check(1, 1024, 300, false, true);
    Check(1, 32, 32, true, false);
    const auto reject = [](const auto &function) {
      try {
        function();
      } catch (const std::runtime_error &) {
        return;
      }
      throw std::runtime_error("invalid top-k input accepted");
    };
    setenv("SGLANG_MLX_NATIVE_HIERARCHICAL_TOPK", "512", 1);
    const auto logits = mx::zeros({1, 32}, mx::float32);
    reject([&] { native::sampling_topk_indices(logits, 0); });
    reject([&] { native::sampling_topk_indices(logits, 33); });
    reject([&] { native::sampling_topk_indices(mx::array(0.0f), 1); });
    for (const char *value : {"512tail", "", "-1", "17", "99999999999999"}) {
      setenv("SGLANG_MLX_NATIVE_HIERARCHICAL_TOPK", value, 1);
      reject([&] { native::sampling_topk_indices(logits, 20); });
    }
    std::cout
        << "hierarchical_topk=pass scores=exact indices=valid_unique "
           "tied_cutoffs=checked small_and_nondivisible_fallbacks=checked\n";
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
