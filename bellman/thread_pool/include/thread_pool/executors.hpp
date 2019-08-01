#pragma once

#include <memory>
#include <type_traits>
#include <thread>
#include <set>

struct Unit {};

class Task;

template <class T>
class Future;

template <class T>
using FuturePtr = std::shared_ptr<Future<T>>;

class Executor {
public:
    virtual ~Executor() = default;

    virtual void Submit(std::shared_ptr<Task> task) = 0;

    virtual void StartShutdown() = 0;
    virtual void WaitShutdown() = 0;

    template <class T, class FuncT>
    FuturePtr<T> Invoke(FuncT&& fn);

    virtual const std::set<std::thread::id>& GetWorkerThreadIds() const = 0;

protected:
    inline static void SetCompleted(Task* task);
    inline static void SetException(Task* task, std::exception_ptr exc);
};

std::shared_ptr<Executor> MakeThreadPoolExecutor(int num_threads);
Executor* GetGlobalExecutor();
