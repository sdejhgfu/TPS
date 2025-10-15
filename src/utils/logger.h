#pragma once

#include <string>
#include <fstream>
#include <mutex>
#include <memory>

namespace hpie {

enum class LogLevel {
    DEBUG = 0,
    INFO = 1,
    WARNING = 2,
    ERROR = 3
};

class Logger {
public:
    static void Initialize(const std::string& log_file = "", LogLevel level = LogLevel::INFO);
    static void Shutdown();
    
    static void Debug(const char* format, ...);
    static void Info(const char* format, ...);
    static void Warning(const char* format, ...);
    static void Error(const char* format, ...);
    
    static void SetLevel(LogLevel level);
    static LogLevel GetLevel() { return current_level_; }
    
    static void SetOutputToFile(bool enable) { output_to_file_ = enable; }
    static void SetOutputToConsole(bool enable) { output_to_console_ = enable; }

private:
    static LogLevel current_level_;
    static std::unique_ptr<std::ofstream> log_file_;
    static std::mutex log_mutex_;
    static bool output_to_file_;
    static bool output_to_console_;
    
    static void Log(LogLevel level, const char* format, va_list args);
    static std::string GetLevelString(LogLevel level);
    static std::string GetTimestamp();
};

} // namespace hpie
