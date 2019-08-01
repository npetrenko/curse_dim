#pragma once

#include <memory>

#include "executors.hpp"
#include "task.hpp"

#include <functional>

template <class T>
class Future : public Task {
public:
    friend class Executor;

    explicit Future(std::function<T()> func) : func_{std::move(func)} {
    }

    void Run() override {
        val_ = func_();
    }

    T Get() & {
        GetHelper();
        return val_;
    }

    T Get() && {
        GetHelper();
        return std::move(val_);
    }

private:
    void GetHelper() {
        this->Wait();
        if (this->IsFailed()) {
            std::rethrow_exception(this->GetError());
        }
    }
    T val_;
    std::function<T()> func_;
};

template <class T, class FuncT>
FuturePtr<T> Executor::Invoke(FuncT&& fn) {
    auto task = std::make_shared<Future<T>>(std::forward<FuncT>(fn));
    Submit(task);
    return task;
}
