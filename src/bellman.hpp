#pragma once

#include <src/agent_policy.hpp>
#include <src/kernel.hpp>
#include <src/util.hpp>
#include <src/particle.hpp>

template <class DerivedT>
class AbstractQFuncEstimate : public CRTPDerivedCaster<DerivedT> {
public:
    template <class StorageT>
    FloatT ValueAtPoint(const Particle<StorageT>& point, size_t action_num) const {
        return this->GetDerived()->ValueAtPoint(point, action_num);
    }

    template <class StorageT>
    FloatT& ValueAtPoint(const Particle<StorageT>& point, size_t action_num) {
        return this->GetDerived()->ValueAtPoint(point, action_num);
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
        FloatT best_val;
        size_t best_action;
        bool is_set{false};
        for (size_t i = 0; i < qfunc_estimate_.NumActions(); ++i) {
            FloatT est = qfunc_estimate_.ValueAtPoint(state, i);
            if (!is_set || best_val < est) {
                is_set = true;
                best_val = est;
                best_action = i;
            }
        }

        return best_action;
    }

private:
    const AbstractQFuncEstimate<T>& qfunc_estimate_;
};
