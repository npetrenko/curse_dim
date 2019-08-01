#pragma once

#include <vector>
#include <algorithm>

#include "future.hpp"

class ParallelFor {
public:
    ParallelFor(size_t from, size_t to, uint8_t step_complexity,
                Executor* exec = GetGlobalExecutor())
        : from_{from}, to_{to}, exec_{exec}, step_complexity_{step_complexity} {
    }

    template <class FuncT>
    void operator()(FuncT func) && {
        // check if we are not running inside the executor
        if (NeedsFallback()) {
            FallbackLoop(std::move(func));
            return;
        }

        std::vector<std::shared_ptr<Task>> tasks;

        size_t piece_size = (to_ - from_) / std::max(step_complexity_, (uint8_t)16);
        piece_size = std::max((size_t)1, piece_size);

        tasks.reserve(1 + ((to_ - from_) / piece_size));

        for (size_t begin_ix = from_; begin_ix < to_; begin_ix += piece_size) {
            size_t end_ix = begin_ix + piece_size;
            end_ix = std::min(end_ix, to_);
            auto work = [begin_ix, end_ix, func]() {
                for (size_t i = begin_ix; i < end_ix; ++i) {
                    func(i);
                }
                return Unit{};
            };

            tasks.push_back(exec_->Invoke<Unit>(work));
        }

        for (auto& task : tasks) {
            task->Wait();
            if (task->IsFailed()) {
                std::rethrow_exception(task->GetError());
            }
        }
    }

private:
    template <class FuncT>
    void FallbackLoop(FuncT func) const {
        for (size_t i = from_; i < to_; ++i) {
            func(i);
        }
    }

    inline bool NeedsFallback() const {
	if (step_complexity_ == 0) {
	    return true;
	}
        auto tid = std::this_thread::get_id();
        if (exec_->GetWorkerThreadIds().count(tid)) {
            return true;
        }
        return false;
    }

    const size_t from_, to_;
    Executor* const exec_;
    const uint8_t step_complexity_;
};
