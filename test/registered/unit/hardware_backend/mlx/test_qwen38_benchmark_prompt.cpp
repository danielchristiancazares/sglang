#define main qwen38_native_benchmark_main
#include "benchmark/mac/bench_qwen38_native.cpp"
#undef main

#include <unistd.h>

namespace {

void Require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

class PromptFixture {
 public:
  PromptFixture() {
    std::string pattern =
        (std::filesystem::temp_directory_path() / "qwen38-prompt-XXXXXX").string();
    const char* directory = mkdtemp(pattern.data());
    if (directory == nullptr) {
      throw std::runtime_error("cannot create prompt fixture");
    }
    directory_ = directory;
  }

  ~PromptFixture() {
    std::error_code ignored;
    std::filesystem::remove_all(directory_, ignored);
  }

  std::filesystem::path Write(const std::string& contents) const {
    const auto path = directory_ / "prompt ids.txt";
    std::ofstream output(path);
    output << contents;
    output.close();
    Require(static_cast<bool>(output), "cannot write prompt fixture");
    return path;
  }

  void Reject(const std::string& contents) const {
    try {
      (void)ReadPromptIds(Write(contents), 3, 248320);
    } catch (const std::runtime_error&) {
      return;
    }
    throw std::runtime_error("malformed benchmark prompt was accepted");
  }

 private:
  std::filesystem::path directory_;
};

}  // namespace

int main() {
  try {
    Require(ParseTokenCount("0", "warmup", 0) == 0, "zero warmup rejected");
    for (const std::string text : {"0", "-1", "1x", "2147483648"}) {
      bool rejected = false;
      try {
        (void)ParseTokenCount(text, "positive count");
      } catch (const std::runtime_error&) {
        rejected = true;
      }
      Require(rejected, "invalid token count was accepted");
    }
    PromptFixture fixture;
    Require(
        ReadPromptIds(fixture.Write(" 0\n9\t248319 \n"), 3, 248320) ==
            std::vector<std::int32_t>{0, 9, 248319},
        "valid boundary IDs changed");
    for (const std::string text : {
             "", "0 1", "0 1 2 3", "0 -1 2", "0 248320 2",
             "0 2147483648 2", "0 1x 2", "0 +1 2", "0 1 2 garbage"}) {
      fixture.Reject(text);
    }
    const auto missing = fixture.Write("0 1 2");
    std::filesystem::remove(missing);
    bool rejected = false;
    try {
      (void)ReadPromptIds(missing, 3, 248320);
    } catch (const std::runtime_error&) {
      rejected = true;
    }
    Require(rejected, "missing prompt file was accepted");
    std::cout << "benchmark prompt validation passed\n";
    return 0;
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
