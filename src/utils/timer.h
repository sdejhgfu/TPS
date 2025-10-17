#pragma once

#include <chrono>
#include <string>
#include <mutex>
#include <unordered_map>

namespace hpie {

class Timer {
public:
    Timer();
    
    void Reset();
    std::chrono::microseconds Elapsed() const;
    double ElapsedSeconds() const;
    double ElapsedMilliseconds() const;
    
    // High-resolution timing for performance measurements
    uint64_t ElapsedNanoseconds() const;
    
    // Lap timing
    std::chrono::microseconds Lap();
    std::chrono::microseconds GetLastLap() const { return last_lap_; }

private:
    std::chrono::steady_clock::time_point start_time_;
    std::chrono::steady_clock::time_point last_lap_time_;
    std::chrono::microseconds last_lap_;
};

// RAII timer for automatic measurement
class ScopedTimer {
public:
    explicit ScopedTimer(const std::string& name);
    ~ScopedTimer();

private:
    std::string name_;
    Timer timer_;
};

// Performance profiler
class PerformanceProfiler {
public:
    static PerformanceProfiler& Instance();
    
    void StartTimer(const std::string& name);
    void EndTimer(const std::string& name);
    void ResetTimer(const std::string& name);
    
    double GetElapsedTime(const std::string& name) const;
    std::chrono::microseconds GetElapsedMicroseconds(const std::string& name) const;
    
    void PrintReport() const;
    void ClearAllTimers();

private:
    PerformanceProfiler() = default;
    
    struct TimerData {
        std::chrono::steady_clock::time_point start_time;
        std::chrono::microseconds total_time;
        size_t call_count;
        bool is_running;
        
        TimerData() : total_time(0), call_count(0), is_running(false) {}
    };
    
    mutable std::mutex mutex_;
    std::unordered_map<std::string, TimerData> timers_;
};

} // namespace hpie
