#pragma once

#include <memory>
#include <mutex>
#include <exception>
#include <atomic>
#include <condition_variable>

#include "executors.hpp"

class Task : public std::enable_shared_from_this<Task> {
public:
    friend class Executor;
    virtual ~Task() = default;
    virtual void Run() = 0;

    // Task::run() completed without throwing exception
    inline bool IsCompleted() const {
        return is_completed_.load();
    }

    // Task::run() throwed exception
    inline bool IsFailed() const {
        return exception_set_.load();
    }

    // Task either completed, failed or was canceled
    inline bool IsFinished() const {
        return IsCompleted() || IsFailed();
    }

    inline std::exception_ptr GetError() {
        return exception_;
    }

    void Wait();

private:
    void SetException(std::exception_ptr exc);
    void SetCompleted();

    std::mutex state_mutex_;
    std::condition_variable wait_cv_;
    std::exception_ptr exception_;

    std::atomic<bool> is_completed_{false};
    std::atomic<bool> exception_set_{false};
};

inline void Executor::SetCompleted(Task* task) {
    task->SetCompleted();
}

inline void Executor::SetException(Task* task, std::exception_ptr exc) {
    task->SetException(exc);
}
