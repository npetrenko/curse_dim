#pragma once

#include "agent_policy.hpp"
#include "kernel.hpp"
#include "util.hpp"
#include "particle.hpp"

class IQFuncEstimate : public EnableCloneInterface<IQFuncEstimate> {
public:
    virtual FloatT ValueAtPoint(TypeErasedParticleRef point, size_t action_num) const = 0;
    virtual ConstMemoryView ValueAtIndex(size_t index) const = 0;
    virtual MemoryView ValueAtIndex(size_t index) = 0;
    virtual size_t NumActions() const = 0;
    virtual ~IQFuncEstimate() = default;
};

template <class QFuncT>
class GreedyPolicy : public EnableClone<GreedyPolicy<QFuncT>, InheritFrom<IAgentPolicy>> {
public:
    GreedyPolicy(const QFuncT& qfunc) noexcept : qfunc_estimate_(qfunc) {
    }

    size_t React(TypeErasedParticleRef state) const override {
        return ReactHelper(
            [&](size_t action_num) { return qfunc_estimate_.ValueAtPoint(state, action_num); });
    }

    size_t React(size_t state_index) const override {
        return ReactHelper(
            [this, state_index, memory_view = qfunc_estimate_.ValueAtIndex(state_index)](
                size_t action_num) { return memory_view[action_num]; });
    }

private:
    template <class Func>
    size_t ReactHelper(Func frozen_state_callback) const {
        assert(qfunc_estimate_.NumActions());
        FloatT best_val = frozen_state_callback(0);
        size_t best_action = 0;
        for (size_t i = 1; i < qfunc_estimate_.NumActions(); ++i) {
            FloatT est = frozen_state_callback(i);
            if (best_val < est) {
                best_val = est;
                best_action = i;
            }
        }

        return best_action;
    }
    const QFuncT& qfunc_estimate_;
};
