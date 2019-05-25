#pragma once

#include <src/agent_policy.hpp>
#include <src/kernel.hpp>
#include <src/util.hpp>
#include <src/particle.hpp>

template <class DerivedT>
class AbstractQFuncEstimate : private CRTPDerivedCaster<DerivedT> {
public:
    template <class StorageT>
    FloatT ValueAtPoint(const Particle<StorageT>& point, size_t action_num) const {
        return this->GetDerived()->ValueAtPoint(point, action_num);
    }

    template <class StorageT>
    FloatT& ValueAtPoint(const Particle<StorageT>& point, size_t action_num) {
        return this->GetDerived()->ValueAtPoint(point, action_num);
    }

    FloatT ValueAtIndex(size_t index, size_t action_num) const {
        return this->GetDerived()->ValueAtIndex(index, action_num);
    }

    FloatT& ValueAtIndex(size_t index, size_t action_num) {
        return this->GetDerived()->ValueAtIndex(index, action_num);
    }

    size_t NumActions() const {
        return this->GetDerived()->NumActions();
    }
};

template <class T>
class GreedyPolicy : public AbstractAgentPolicy<GreedyPolicy<T>> {
public:
    GreedyPolicy(const AbstractQFuncEstimate<T>& qfunc) noexcept : qfunc_estimate_(qfunc) {
    }

    template <class S>
    size_t React(const Particle<S>& state) const {
        struct {
            const Particle<S>& state;
            const AbstractQFuncEstimate<T>& qfunc;
            FloatT operator()(size_t action_num) {
                return qfunc.ValueAtPoint(state, action_num);
            }
        } call_helper{state, qfunc_estimate_};
        return ReactHelper(call_helper);
    }

    size_t React(size_t state_index) const {
        return ReactHelper([this, state_index](size_t action_num) {
            return qfunc_estimate_.ValueAtIndex(state_index, action_num);
        });
    }

private:
    template <class Func>
    size_t ReactHelper(Func frozen_state_callback) const {
        FloatT best_val{std::numeric_limits<FloatT>::min()};
        size_t best_action;
        for (size_t i = 0; i < qfunc_estimate_.NumActions(); ++i) {
            FloatT est = frozen_state_callback(i);
            if (best_val < est) {
                best_val = est;
                best_action = i;
            }
        }

        return best_action;
    }
    const AbstractQFuncEstimate<T>& qfunc_estimate_;
};
