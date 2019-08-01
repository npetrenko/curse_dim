#include <thread_pool/task.hpp>

void Task::Wait() {
    std::unique_lock lock(state_mutex_);
    wait_cv_.wait(lock, [&] { return IsFinished(); });
}

void Task::SetException(std::exception_ptr exc) {
    {
        std::lock_guard lock{state_mutex_};
        exception_ = exc;
        exception_set_.store(true);
    }
    wait_cv_.notify_all();
}

void Task::SetCompleted() {
    {
        std::lock_guard lock{state_mutex_};
        is_completed_.store(true);
    }
    wait_cv_.notify_all();
}
