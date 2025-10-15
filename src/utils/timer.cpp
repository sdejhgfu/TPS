#include "timer.h"
#include "logger.h"
#include <iostream>
#include <iomanip>
#include <algorithm>

namespace hpie {

Timer::Timer() {
    Reset();
}

void Timer::Reset() {
    start_time_ = std::chrono::steady_clock::now();
    last_lap_time_ = start_time_;
    last_lap_ = std::chrono::microseconds(0);
}

std::chrono::microseconds Timer::Elapsed() const {
    auto now = std::chrono::steady_clock::now();
    return std::chrono::duration_cast<std::chrono::microseconds>(now - start_time_);
}

double Timer::ElapsedSeconds() const {
    return Elapsed().count() / 1000000.0;
}

double Timer::ElapsedMilliseconds() const {
    return Elapsed().count() / 1000.0;
}

uint64_t Timer::ElapsedNanoseconds() const {
    auto now = std::chrono::steady_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(now - start_time_);
    return duration.count();
}

std::chrono::microseconds Timer::Lap() {
    auto now = std::chrono::steady_clock::now();
    last_lap_ = std::chrono::duration_cast<std::chrono::microseconds>(now - last_lap_time_);
    last_lap_time_ = now;
    return last_lap_;
}

ScopedTimer::ScopedTimer(const std::string& name) : name_(name) {
    Logger::Debug("Starting timer: %s", name_.c_str());
}

ScopedTimer::~ScopedTimer() {
    double elapsed = timer_.ElapsedSeconds();
    Logger::Debug("Timer %s: %.3f seconds", name_.c_str(), elapsed);
}

PerformanceProfiler& PerformanceProfiler::Instance() {
    static PerformanceProfiler instance;
    return instance;
}

void PerformanceProfiler::StartTimer(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto& timer_data = timers_[name];
    if (!timer_data.is_running) {
        timer_data.start_time = std::chrono::steady_clock::now();
        timer_data.is_running = true;
    }
}

void PerformanceProfiler::EndTimer(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = timers_.find(name);
    if (it != timers_.end() && it->second.is_running) {
        auto now = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::microseconds>(
            now - it->second.start_time);
        
        it->second.total_time += elapsed;
        it->second.call_count++;
        it->second.is_running = false;
    }
}

void PerformanceProfiler::ResetTimer(const std::string& name) {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = timers_.find(name);
    if (it != timers_.end()) {
        it->second.total_time = std::chrono::microseconds(0);
        it->second.call_count = 0;
        it->second.is_running = false;
    }
}

double PerformanceProfiler::GetElapsedTime(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = timers_.find(name);
    if (it != timers_.end()) {
        return it->second.total_time.count() / 1000000.0; // Convert to seconds
    }
    return 0.0;
}

std::chrono::microseconds PerformanceProfiler::GetElapsedMicroseconds(const std::string& name) const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    auto it = timers_.find(name);
    if (it != timers_.end()) {
        return it->second.total_time;
    }
    return std::chrono::microseconds(0);
}

void PerformanceProfiler::PrintReport() const {
    std::lock_guard<std::mutex> lock(mutex_);
    
    if (timers_.empty()) {
        Logger::Info("No performance timers recorded");
        return;
    }
    
    Logger::Info("=== Performance Report ===");
    Logger::Info("%-20s %12s %12s %12s %12s", 
                 "Timer Name", "Total (s)", "Calls", "Avg (ms)", "Total (ms)");
    Logger::Info("%-20s %12s %12s %12s %12s", 
                 "----------", "---------", "-----", "--------", "----------");
    
    // Sort by total time
    std::vector<std::pair<std::string, const TimerData*>> sorted_timers;
    for (const auto& pair : timers_) {
        sorted_timers.emplace_back(pair.first, &pair.second);
    }
    
    std::sort(sorted_timers.begin(), sorted_timers.end(),
              [](const auto& a, const auto& b) {
                  return a.second->total_time > b.second->total_time;
              });
    
    for (const auto& pair : sorted_timers) {
        const std::string& name = pair.first;
        const TimerData& data = *pair.second;
        
        double total_seconds = data.total_time.count() / 1000000.0;
        double avg_ms = data.call_count > 0 ? 
            (data.total_time.count() / 1000.0) / data.call_count : 0.0;
        double total_ms = data.total_time.count() / 1000.0;
        
        Logger::Info("%-20s %12.3f %12zu %12.3f %12.3f",
                     name.c_str(), total_seconds, data.call_count, avg_ms, total_ms);
    }
    
    Logger::Info("==========================");
}

void PerformanceProfiler::ClearAllTimers() {
    std::lock_guard<std::mutex> lock(mutex_);
    timers_.clear();
}

} // namespace hpie
