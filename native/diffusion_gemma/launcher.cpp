// SPDX-License-Identifier: Apache-2.0
// The bootstrap loads CPython only after establishing its DLL search directory.
#include <windows.h>

#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
class LaunchFailure final : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

std::filesystem::path executable_directory() {
    std::vector<wchar_t> buffer(32768);
    switch (const auto length = GetModuleFileNameW(nullptr, buffer.data(),
                                                 static_cast<DWORD>(buffer.size()))) {
    case 0:
        throw LaunchFailure("Cannot resolve the native launcher's executable path.");
    default:
        if (length >= buffer.size()) {
            throw LaunchFailure("The native launcher's executable path exceeds the Windows limit.");
        }
        return std::filesystem::path(std::wstring(buffer.data(), length)).parent_path();
    }
}
} // namespace

int wmain(int argc, wchar_t** argv) {
    try {
        if (SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS) == 0) {
            throw LaunchFailure("Cannot establish the native DLL search policy.");
        }
        if (AddDllDirectory(DG_PYTHON_BASE) == nullptr) {
            throw LaunchFailure("Cannot locate the configured CPython 3.13 runtime.");
        }
        const auto library_path = executable_directory() / L"diffusiongemma_runtime.dll";
        const auto library = [&] {
            if (auto loaded = LoadLibraryExW(library_path.c_str(), nullptr,
                    LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS)) {
                return loaded;
            }
            throw LaunchFailure("Cannot load diffusiongemma_runtime.dll (Windows error " +
                                std::to_string(GetLastError()) + "). Rebuild for the installed CPython.");
        }();
        using Run = int (*)(int, wchar_t**);
        const auto run = [&] {
            if (auto entry = GetProcAddress(library, "diffusiongemma_main")) {
                return reinterpret_cast<Run>(entry);
            }
            throw LaunchFailure("The native runtime does not export diffusiongemma_main.");
        }();
        // CPython and CUDA own process-lifetime DLL resources. Windows releases
        // the runtime after its interpreter and all model objects have shut down.
        return run(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "DiffusionGemma launch failed: " << error.what() << '\n';
        return 1;
    }
}
