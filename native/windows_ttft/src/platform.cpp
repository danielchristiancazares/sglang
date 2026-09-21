#include <sglang/windows_ttft/launcher.hpp>

#include <array>
#include <atomic>
#include <chrono>
#include <cctype>
#include <cerrno>
#include <limits>
#include <optional>
#include <stdexcept>
#include <system_error>
#include <utility>

#ifdef _WIN32
#include <winsock2.h>
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/file.h>
#include <sys/stat.h>
#include <unistd.h>
extern char **environ;
#endif

namespace sglang::windows_ttft {
namespace {
#ifdef _WIN32
[[noreturn]] void win_error(const char *operation) {
  const auto error = GetLastError();
  throw std::runtime_error(std::string(operation) + " failed, Windows error " +
                           std::to_string(error));
}

class Handle {
public:
  explicit Handle(HANDLE value = nullptr) : value_(value) {}
  ~Handle() {
    if (value_ != nullptr && value_ != INVALID_HANDLE_VALUE) CloseHandle(value_);
  }
  Handle(const Handle &) = delete;
  Handle &operator=(const Handle &) = delete;
  Handle(Handle &&other) noexcept : value_(std::exchange(other.value_, nullptr)) {}
  [[nodiscard]] HANDLE get() const { return value_; }

private:
  HANDLE value_;
};

class Attributes {
public:
  Attributes() {
    SIZE_T size = 0;
    InitializeProcThreadAttributeList(nullptr, 2, 0, &size);
    if (size == 0) win_error("InitializeProcThreadAttributeList size");
    memory_.resize(size);
    list_ = reinterpret_cast<LPPROC_THREAD_ATTRIBUTE_LIST>(memory_.data());
    if (!InitializeProcThreadAttributeList(list_, 2, 0, &size)) {
      win_error("InitializeProcThreadAttributeList");
    }
  }
  ~Attributes() { DeleteProcThreadAttributeList(list_); }
  void set(DWORD_PTR attribute, void *value, SIZE_T size) {
    if (!UpdateProcThreadAttribute(list_, 0, attribute, value, size, nullptr, nullptr)) {
      win_error("UpdateProcThreadAttribute");
    }
  }
  [[nodiscard]] LPPROC_THREAD_ATTRIBUTE_LIST get() const { return list_; }

private:
  std::vector<std::byte> memory_;
  LPPROC_THREAD_ATTRIBUTE_LIST list_{nullptr};
};

std::atomic<bool> interrupted{false};
BOOL WINAPI on_control(DWORD event) {
  if (event == CTRL_C_EVENT || event == CTRL_BREAK_EVENT) {
    interrupted.store(true, std::memory_order_relaxed);
    return TRUE;
  }
  // On console close/logoff let Windows terminate this process. Closing the
  // non-inherited job handle then kills only its descendants.
  return FALSE;
}

class ConsoleHandler {
public:
  ConsoleHandler() {
    interrupted.store(false, std::memory_order_relaxed);
    if (!SetConsoleCtrlHandler(on_control, TRUE)) win_error("SetConsoleCtrlHandler");
  }
  ~ConsoleHandler() { SetConsoleCtrlHandler(on_control, FALSE); }
};

Handle standard_handle(DWORD kind) {
  const HANDLE original = GetStdHandle(kind);
  if (original == nullptr || original == INVALID_HANDLE_VALUE) {
    throw std::runtime_error("serve requires foreground standard handles");
  }
  HANDLE duplicate = nullptr;
  if (!DuplicateHandle(GetCurrentProcess(), original, GetCurrentProcess(),
                       &duplicate, 0, TRUE, DUPLICATE_SAME_ACCESS)) {
    win_error("DuplicateHandle standard stream");
  }
  return Handle(duplicate);
}

void check_port() {
  WSADATA data{};
  if (WSAStartup(MAKEWORD(2, 2), &data) != 0) {
    throw std::runtime_error("WSAStartup failed");
  }
  const SOCKET socket = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
  if (socket == INVALID_SOCKET) {
    WSACleanup();
    throw std::runtime_error("port preflight socket failed");
  }
  BOOL exclusive = TRUE;
  sockaddr_in address{};
  address.sin_family = AF_INET;
  address.sin_port = htons(30000);
  address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
  const bool free =
      setsockopt(socket, SOL_SOCKET, SO_EXCLUSIVEADDRUSE,
                 reinterpret_cast<const char *>(&exclusive), sizeof(exclusive)) == 0 &&
      bind(socket, reinterpret_cast<const sockaddr *>(&address), sizeof(address)) == 0;
  closesocket(socket);
  WSACleanup();
  if (!free) throw std::runtime_error("loopback port 30000 is not exclusively available");
}
#endif
} // namespace

std::wstring quote_windows_argument(std::wstring_view argument) {
  if (argument.find(L'\0') != std::wstring_view::npos) {
    throw std::invalid_argument("NUL in process argument");
  }
  // Microsoft CRT argv rules: double backslashes before the closing quote,
  // and 2N+1 before embedded quotes. No shell metacharacter interpretation.
  std::wstring result(1, L'"');
  std::size_t backslashes = 0;
  for (const wchar_t character : argument) {
    if (character == L'\\') {
      ++backslashes;
    } else {
      result.append(backslashes * (character == L'"' ? 2 : 1), L'\\');
      backslashes = 0;
      if (character == L'"') result += L'\\';
      result += character;
    }
  }
  result.append(backslashes * 2, L'\\');
  result += L'"';
  return result;
}

std::wstring utf8_to_wide(std::string_view text) {
  static_cast<void>(benchmark::utf8_code_point_count(text));
  std::wstring result;
  for (std::size_t i = 0; i < text.size();) {
    const auto lead = static_cast<unsigned char>(text[i++]);
    std::uint32_t scalar = lead;
    unsigned tail = 0;
    if (lead >= 0xF0) { scalar = lead & 7U; tail = 3; }
    else if (lead >= 0xE0) { scalar = lead & 15U; tail = 2; }
    else if (lead >= 0xC0) { scalar = lead & 31U; tail = 1; }
    for (unsigned j = 0; j < tail; ++j) {
      scalar = (scalar << 6U) | (static_cast<unsigned char>(text[i++]) & 63U);
    }
    if constexpr (sizeof(wchar_t) == 2) {
      if (scalar > 0xFFFFU) {
        scalar -= 0x10000U;
        result += static_cast<wchar_t>(0xD800U + (scalar >> 10U));
        result += static_cast<wchar_t>(0xDC00U + (scalar & 1023U));
        continue;
      }
    }
    result += static_cast<wchar_t>(scalar);
  }
  return result;
}

std::string wide_to_utf8(std::wstring_view text) {
  std::string result;
  for (std::size_t i = 0; i < text.size(); ++i) {
    auto scalar = static_cast<std::uint32_t>(text[i]);
    if constexpr (sizeof(wchar_t) == 2) {
      if (scalar >= 0xD800U && scalar <= 0xDBFFU) {
        if (++i == text.size()) throw std::invalid_argument("truncated UTF-16");
        const auto low = static_cast<std::uint32_t>(text[i]);
        if (low < 0xDC00U || low > 0xDFFFU) throw std::invalid_argument("invalid UTF-16");
        scalar = 0x10000U + ((scalar - 0xD800U) << 10U) + low - 0xDC00U;
      }
    }
    if (scalar > 0x10FFFFU || (scalar >= 0xD800U && scalar <= 0xDFFFU)) {
      throw std::invalid_argument("invalid Unicode scalar");
    }
    if (scalar < 0x80U) result += static_cast<char>(scalar);
    else if (scalar < 0x800U) {
      result += static_cast<char>(0xC0U | (scalar >> 6U));
      result += static_cast<char>(0x80U | (scalar & 63U));
    } else if (scalar < 0x10000U) {
      result += static_cast<char>(0xE0U | (scalar >> 12U));
      result += static_cast<char>(0x80U | ((scalar >> 6U) & 63U));
      result += static_cast<char>(0x80U | (scalar & 63U));
    } else {
      result += static_cast<char>(0xF0U | (scalar >> 18U));
      result += static_cast<char>(0x80U | ((scalar >> 12U) & 63U));
      result += static_cast<char>(0x80U | ((scalar >> 6U) & 63U));
      result += static_cast<char>(0x80U | (scalar & 63U));
    }
  }
  return result;
}

bool is_link(const fs::path &path) {
#ifdef _WIN32
  const auto attributes = GetFileAttributesW(path.c_str());
  if (attributes == INVALID_FILE_ATTRIBUTES) {
    const auto error = GetLastError();
    if (error == ERROR_FILE_NOT_FOUND || error == ERROR_PATH_NOT_FOUND) return false;
    win_error("GetFileAttributesW");
  }
  return (attributes & FILE_ATTRIBUTE_REPARSE_POINT) != 0;
#else
  std::error_code error;
  const auto status = fs::symlink_status(path, error);
  if (error && error != std::errc::no_such_file_or_directory) {
    throw std::system_error(error, "symlink_status");
  }
  return fs::is_symlink(status);
#endif
}

std::string file_identity(const fs::path &path) {
#ifdef _WIN32
  Handle handle(CreateFileW(path.c_str(), FILE_READ_ATTRIBUTES,
                            FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                            nullptr, OPEN_EXISTING, FILE_FLAG_OPEN_REPARSE_POINT, nullptr));
  if (handle.get() == INVALID_HANDLE_VALUE) win_error("CreateFileW identity");
  FILE_ID_INFO info{};
  if (!GetFileInformationByHandleEx(handle.get(), FileIdInfo, &info, sizeof(info))) {
    win_error("GetFileInformationByHandleEx");
  }
  std::string result = std::to_string(info.VolumeSerialNumber) + ":";
  constexpr std::string_view hex = "0123456789abcdef";
  for (const auto byte : info.FileId.Identifier) {
    result += hex[byte >> 4U];
    result += hex[byte & 15U];
  }
  return result;
#else
  struct stat info{};
  if (stat(path.c_str(), &info) != 0) {
    throw std::system_error(errno, std::generic_category(), "stat");
  }
  return std::to_string(info.st_dev) + ":" + std::to_string(info.st_ino);
#endif
}

WorkspaceLock::WorkspaceLock(const fs::path &state_root) {
  if (is_link(state_root)) throw std::runtime_error("linked state root rejected");
  fs::create_directory(state_root);
  const auto path = state_root / "workspace.lock";
  if (is_link(path)) throw std::runtime_error("linked lock rejected");
#ifdef _WIN32
  const HANDLE handle = CreateFileW(path.c_str(), GENERIC_READ | GENERIC_WRITE,
                                    0, nullptr, OPEN_ALWAYS,
                                    FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OPEN_REPARSE_POINT, nullptr);
  if (handle == INVALID_HANDLE_VALUE) {
    throw std::runtime_error("cannot lock this checkout; another launcher may own it");
  }
  handle_ = reinterpret_cast<std::intptr_t>(handle);
#else
  const int handle = open(path.c_str(), O_CREAT | O_RDWR | O_CLOEXEC | O_NOFOLLOW, 0600);
  if (handle < 0) throw std::system_error(errno, std::generic_category(), "open lock");
  if (flock(handle, LOCK_EX | LOCK_NB) != 0) {
    close(handle);
    throw std::runtime_error("cannot lock this checkout; another launcher may own it");
  }
  handle_ = handle;
#endif
}

WorkspaceLock::~WorkspaceLock() {
  if (handle_ == -1) return;
#ifdef _WIN32
  CloseHandle(reinterpret_cast<HANDLE>(handle_));
#else
  close(static_cast<int>(handle_));
#endif
}

Environment current_environment() {
  Environment result;
#ifdef _WIN32
  wchar_t *block = GetEnvironmentStringsW();
  if (block == nullptr) win_error("GetEnvironmentStringsW");
  try {
    for (const wchar_t *entry = block; *entry != L'\0';) {
      const std::wstring_view value(entry);
      const auto separator = value.find(L'=', 1); // preserve Windows "=C:" entries
      if (separator != std::wstring_view::npos) {
        auto name = wide_to_utf8(value.substr(0, separator));
        for (auto &c : name) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
        result.emplace(std::move(name), wide_to_utf8(value.substr(separator + 1)));
      }
      entry += value.size() + 1;
    }
  } catch (...) {
    FreeEnvironmentStringsW(block);
    throw;
  }
  FreeEnvironmentStringsW(block);
#else
  for (char **entry = environ; *entry != nullptr; ++entry) {
    const std::string_view value(*entry);
    const auto separator = value.find('=');
    if (separator != std::string_view::npos) {
      result.emplace(value.substr(0, separator), value.substr(separator + 1));
    }
  }
#endif
  return result;
}

void check_windows_environment(const Environment &environment) {
#ifdef _WIN32
  for (const auto &[key, expected] :
       Environment{{"MAX_JOBS", "2"}, {"DISTUTILS_USE_SDK", "1"},
                   {"TORCH_CUDA_ARCH_LIST", "12.0"}, {"FLASHINFER_CUDA_ARCH_LIST", "12.0"}}) {
    const auto value = environment.find(key);
    if (value == environment.end() || value->second != expected) {
      throw std::runtime_error(
          "run initialize_cuda_build_env.ps1 -MaxJobs 2 first; unexpected " + key);
    }
  }
  const auto cuda = environment.find("CUDA_HOME");
  if (cuda == environment.end() ||
      !fs::is_regular_file(text_path(cuda->second) / "bin/nvcc.exe")) {
    throw std::runtime_error("initialized CUDA_HOME/bin/nvcc.exe is missing");
  }
  if (SearchPathW(nullptr, L"cl.exe", nullptr, 0, nullptr, nullptr) == 0) {
    throw std::runtime_error("MSVC cl.exe is not on the initialized PATH");
  }
#else
  static_cast<void>(environment);
  throw std::runtime_error("serve is native-Windows only; plan/prepare are CPU-only");
#endif
}

int serve(const Plan &plan) {
#ifdef _WIN32
  auto environment = current_environment();
  check_windows_environment(environment);
  // Revalidate inherited overrides, then change only the child environment.
  if (selected_environment(plan.options, environment) != plan.environment) {
    throw std::runtime_error("environment changed after plan");
  }
  for (const auto &[key, value] : plan.environment) environment[key] = value;
  std::wstring block;
  for (const auto &[key, value] : environment) {
    block += utf8_to_wide(key) + L"=" + utf8_to_wide(value);
    block += L'\0';
  }
  block += L'\0';
  std::wstring command = quote_windows_argument(plan.executable.wstring());
  for (const auto &arg : plan.arguments) {
    command += L" " + quote_windows_argument(utf8_to_wide(arg));
  }
  if (command.size() >= 32767) throw std::runtime_error("Windows command line is too long");

  Handle job(CreateJobObjectW(nullptr, nullptr));
  if (job.get() == nullptr) win_error("CreateJobObjectW");
  JOBOBJECT_EXTENDED_LIMIT_INFORMATION limits{};
  limits.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
  if (!SetInformationJobObject(job.get(), JobObjectExtendedLimitInformation,
                               &limits, sizeof(limits))) {
    win_error("SetInformationJobObject");
  }
  auto input = standard_handle(STD_INPUT_HANDLE);
  auto output = standard_handle(STD_OUTPUT_HANDLE);
  auto error = standard_handle(STD_ERROR_HANDLE);
  std::array<HANDLE, 3> inherited_handles{input.get(), output.get(), error.get()};
  HANDLE job_handle = job.get();
  Attributes attributes;
  // Assign the job during creation, not after a running child can escape.
  attributes.set(PROC_THREAD_ATTRIBUTE_JOB_LIST, &job_handle, sizeof(job_handle));
  attributes.set(PROC_THREAD_ATTRIBUTE_HANDLE_LIST, inherited_handles.data(),
                  sizeof(inherited_handles));
  STARTUPINFOEXW startup{};
  startup.StartupInfo.cb = sizeof(startup);
  startup.StartupInfo.dwFlags = STARTF_USESTDHANDLES;
  startup.StartupInfo.hStdInput = input.get();
  startup.StartupInfo.hStdOutput = output.get();
  startup.StartupInfo.hStdError = error.get();
  startup.lpAttributeList = attributes.get();
  ConsoleHandler console;
  check_port();
  PROCESS_INFORMATION process{};
  if (!CreateProcessW(plan.executable.c_str(), command.data(), nullptr, nullptr, TRUE,
                      EXTENDED_STARTUPINFO_PRESENT | CREATE_UNICODE_ENVIRONMENT,
                      block.data(), plan.options.repo.c_str(),
                      &startup.StartupInfo, &process)) {
    win_error("CreateProcessW");
  }
  Handle child(process.hProcess);
  Handle thread(process.hThread);
  std::optional<std::chrono::steady_clock::time_point> stopping;
  for (;;) {
    const auto wait = WaitForSingleObject(child.get(), 250);
    if (wait == WAIT_OBJECT_0) break;
    if (wait == WAIT_FAILED) win_error("WaitForSingleObject");
    if (interrupted.load(std::memory_order_relaxed)) {
      const auto now = std::chrono::steady_clock::now();
      if (!stopping) stopping = now;
      if (now - *stopping >= std::chrono::seconds(30)) {
        if (!TerminateJobObject(job.get(), 130)) win_error("TerminateJobObject");
        if (WaitForSingleObject(child.get(), 5000) != WAIT_OBJECT_0) {
          throw std::runtime_error("owned job did not exit after termination");
        }
        return 130;
      }
    }
  }
  DWORD exit_code = 0;
  if (!GetExitCodeProcess(child.get(), &exit_code)) win_error("GetExitCodeProcess");
  return exit_code <= static_cast<DWORD>(std::numeric_limits<int>::max())
             ? static_cast<int>(exit_code) : 1;
#else
  static_cast<void>(plan);
  throw std::runtime_error("serve is native-Windows only");
#endif
}
} // namespace sglang::windows_ttft
