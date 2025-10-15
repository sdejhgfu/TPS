#include "logger.h"
#include <cstdarg>
#include <iostream>
#include <chrono>
#include <iomanip>
#include <sstream>

namespace hpie {

LogLevel Logger::current_level_ = LogLevel::INFO;
std::unique_ptr<std::ofstream> Logger::log_file_ = nullptr;
std::mutex Logger::log_mutex_;
bool Logger::output_to_file_ = false;
bool Logger::output_to_console_ = true;

void Logger::Initialize(const std::string& log_file, LogLevel level) {
    std::lock_guard<std::mutex> lock(log_mutex_);
    
    current_level_ = level;
    
    if (!log_file.empty()) {
        log_file_ = std::make_unique<std::ofstream>(log_file, std::ios::app);
        output_to_file_ = true;
        
        if (!log_file_->is_open()) {
            std::cerr << "Failed to open log file: " << log_file << std::endl;
            output_to_file_ = false;
        }
    }
}

void Logger::Shutdown() {
    std::lock_guard<std::mutex> lock(log_mutex_);
    
    if (log_file_ && log_file_->is_open()) {
        log_file_->close();
    }
    log_file_.reset();
}

void Logger::Debug(const char* format, ...) {
    if (current_level_ <= LogLevel::DEBUG) {
        va_list args;
        va_start(args, format);
        Log(LogLevel::DEBUG, format, args);
        va_end(args);
    }
}

void Logger::Info(const char* format, ...) {
    if (current_level_ <= LogLevel::INFO) {
        va_list args;
        va_start(args, format);
        Log(LogLevel::INFO, format, args);
        va_end(args);
    }
}

void Logger::Warning(const char* format, ...) {
    if (current_level_ <= LogLevel::WARNING) {
        va_list args;
        va_start(args, format);
        Log(LogLevel::WARNING, format, args);
        va_end(args);
    }
}

void Logger::Error(const char* format, ...) {
    if (current_level_ <= LogLevel::ERROR) {
        va_list args;
        va_start(args, format);
        Log(LogLevel::ERROR, format, args);
        va_end(args);
    }
}

void Logger::SetLevel(LogLevel level) {
    std::lock_guard<std::mutex> lock(log_mutex_);
    current_level_ = level;
}

void Logger::Log(LogLevel level, const char* format, va_list args) {
    std::lock_guard<std::mutex> lock(log_mutex_);
    
    std::stringstream ss;
    ss << "[" << GetTimestamp() << "] "
       << "[" << GetLevelString(level) << "] ";
    
    // Format the message
    char buffer[1024];
    vsnprintf(buffer, sizeof(buffer), format, args);
    ss << buffer << std::endl;
    
    std::string message = ss.str();
    
    // Output to console
    if (output_to_console_) {
        if (level >= LogLevel::ERROR) {
            std::cerr << message;
        } else {
            std::cout << message;
        }
    }
    
    // Output to file
    if (output_to_file_ && log_file_ && log_file_->is_open()) {
        *log_file_ << message;
        log_file_->flush();
    }
}

std::string Logger::GetLevelString(LogLevel level) {
    switch (level) {
        case LogLevel::DEBUG:   return "DEBUG";
        case LogLevel::INFO:    return "INFO ";
        case LogLevel::WARNING: return "WARN ";
        case LogLevel::ERROR:   return "ERROR";
        default:                return "UNKNOWN";
    }
}

std::string Logger::GetTimestamp() {
    auto now = std::chrono::system_clock::now();
    auto time_t = std::chrono::system_clock::to_time_t(now);
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()) % 1000;
    
    std::stringstream ss;
    ss << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S");
    ss << '.' << std::setfill('0') << std::setw(3) << ms.count();
    
    return ss.str();
}

} // namespace hpie
