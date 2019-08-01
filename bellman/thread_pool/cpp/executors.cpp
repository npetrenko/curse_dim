#include <thread_pool/executors.hpp>
#include <thread_pool/task.hpp>
#include <thread_pool/exceptions.hpp>

#include <thread>
#include <mutex>
#include <deque>
#include <condition_variable>
#include <exception>
#include <cassert>
#include <optional>
#include <vector>

struct OnTaskFinishCallback {
    Task* task;
    void operator()(std::exception_ptr eptr) const;
};

class ThreadPool {
public:
    explicit ThreadPool(int num_threads);

    bool Run(std::shared_ptr<Task> fun,
             std::optional<OnTaskFinishCallback> on_execution = std::nullopt);

    void StartShutdown();
    void WaitShutdown();

    const std::set<std::thread::id>& GetWorkerThreadIds() const;

    ~ThreadPool();

private:
    using ElemT = std::pair<std::shared_ptr<Task>, std::optional<OnTaskFinishCallback>>;
    void PoolLoop();
    void ProcessSingleTask(ElemT&& pool_element);
    void WaitMetaReady() const;

    mutable std::mutex mut_;
    mutable std::condition_variable pool_cv_;

    // for gathering info on started threads
    mutable std::mutex meta_mut_;
    mutable std::condition_variable meta_cv_;

    std::deque<ElemT> pool_;
    std::vector<std::thread> threads_;
    std::set<std::thread::id> worker_thread_ids_;

    bool stopped_{false};
    std::atomic<bool> has_waited_for_shutdown_{false};
    int num_started_threads_{0};
    bool finished_gathering_meta_{false};
};

class ThreadPoolExecutor : public Executor {
    friend struct OnTaskFinishCallback;

public:
    ThreadPoolExecutor(int num_threads) : thread_pool_(num_threads) {
    }

    void Submit(std::shared_ptr<Task> task) override {
        OnTaskFinishCallback callback{task.get()};
        if (!thread_pool_.Run(std::move(task), std::move(callback))) {
            throw ThreadPoolExeption{};
        }
    }

    void StartShutdown() override {
        thread_pool_.StartShutdown();
    }

    void WaitShutdown() override {
        thread_pool_.WaitShutdown();
    }

    const std::set<std::thread::id>& GetWorkerThreadIds() const override {
        return thread_pool_.GetWorkerThreadIds();
    }

private:
    ThreadPool thread_pool_;
};

void OnTaskFinishCallback::operator()(std::exception_ptr eptr) const {
    if (eptr) {
        ThreadPoolExecutor::SetException(task, eptr);
    } else {
        ThreadPoolExecutor::SetCompleted(task);
    }
}

std::shared_ptr<Executor> MakeThreadPoolExecutor(int num_threads) {
    return std::make_shared<ThreadPoolExecutor>(num_threads);
}

Executor* GetGlobalExecutor() {
    static auto executor{std::make_unique<ThreadPoolExecutor>(std::thread::hardware_concurrency())};
    return executor.get();
}

ThreadPool::ThreadPool(int num_threads) {
    assert(num_threads > 0);
    for (int i = 0; i < num_threads; ++i) {
        threads_.emplace_back([this, num_threads] {
            {
                std::lock_guard lock{meta_mut_};

                worker_thread_ids_.insert(std::this_thread::get_id());
                ++num_started_threads_;
                if (num_started_threads_ == num_threads) {
                    finished_gathering_meta_ = true;
                    meta_cv_.notify_all();
                }
            }
            this->PoolLoop();
        });
    }
}

void ThreadPool::WaitMetaReady() const {
    std::unique_lock lock{meta_mut_};
    meta_cv_.wait(lock, [this] { return finished_gathering_meta_; });
}

const std::set<std::thread::id>& ThreadPool::GetWorkerThreadIds() const {
    WaitMetaReady();
    return worker_thread_ids_;
}

bool ThreadPool::Run(std::shared_ptr<Task> fun, std::optional<OnTaskFinishCallback> on_execution) {
    {
        std::lock_guard lock(mut_);
        if (stopped_) {
            return false;
        }
        pool_.emplace_back(std::move(fun), std::move(on_execution));
    }
    pool_cv_.notify_one();
    return true;
}

void ThreadPool::StartShutdown() {
    {
        std::lock_guard lock(mut_);
        stopped_ = true;
    }
    pool_cv_.notify_all();
}

void ThreadPool::WaitShutdown() {
    if (!has_waited_for_shutdown_.exchange(true)) {
        for (auto& th : threads_) {
            th.join();
        }
    }
}

ThreadPool::~ThreadPool() {
    if (!has_waited_for_shutdown_.load()) {
        StartShutdown();
        WaitShutdown();
    }
}

void ThreadPool::PoolLoop() {
    while (true) {
        ElemT current_execution;
        bool needs_notify;
        {
            std::unique_lock lock(mut_);
            pool_cv_.wait(lock, [&] { return !(pool_.empty() && !stopped_); });

            if (stopped_ && pool_.empty()) {
                return;
            }

            current_execution = std::move(pool_.front());
            pool_.pop_front();
            needs_notify = !pool_.empty();
        }
        if (needs_notify) {
            pool_cv_.notify_one();
        }
        ProcessSingleTask(std::move(current_execution));
    }
}

void ThreadPool::ProcessSingleTask(ElemT&& pool_element) {
    try {
        pool_element.first->Run();
        if (pool_element.second) {
            pool_element.second.value()(nullptr);
        }
    } catch (...) {
        if (pool_element.second) {
            pool_element.second.value()(std::current_exception());
        } else {
            throw;
        }
    }
}
