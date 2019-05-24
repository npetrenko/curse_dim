#pragma once

#include <src/types.hpp>
#include <src/particle.hpp>
#include <src/exceptions.hpp>
#include <src/agent_policy.hpp>

#include <array>
#include <memory>
#include <exception>

template <class DerivedT>
struct CRTPDerivedCaster {
    DerivedT* GetDerived() {
        return static_cast<DerivedT>(this);
    }
};

template <class DerivedT>
class AbstractKernel : public CRTPDerivedCaster<DerivedT> {
public:
    void Evolve(const Particle& from, Particle* output) {
        this->GetDerived()->Evolve(from, output);
    }
    FloatT GetTransDensity(const Particle& from, const Particle& to) {
        return this->GetDerived()->GetTransDensity(from, to);
    }
};

template <class DerivedT>
class AbstractConditionedKernel : public CRTPDerivedCaster<DerivedT> {
public:
    void EvolveConditionally(const Particle& from, Particle* output, size_t condition) {
        this->GetDerived()->EvolveConditionally(from, output, condition);
    }

    FloatT GetTransDensityConditionally(const Particle& from, const Particle& to,
                                        size_t condition) {
        return this->GetDerived()->GetTransDensityConditionally(from, to, condition);
    }
};

template <class... T>
class ActionConditionedKernel final
    : public AbstractConditionedKernel<ActionConditionedKernel<T...>> {
public:
    ActionConditionedKernel(T&&... args) : fixed_action_kernels_(std::forward<T>(args)...) {
    }

    void EvolveConditionally(const Particle& from, Particle* output, size_t action_number) {
        EvolveHelper helper{from, output};
        CallOnTupleIx(std::move(helper), fixed_action_kernels_, action_number);
    }

    FloatT GetTransDensityConditionally(const Particle& from, const Particle& to,
                                        size_t action_number) {
        FloatT result;
        DensityHelper helper{from, to, result};
        CallOnTupleIx(std::move(helper), fixed_action_kernels_, action_number);
        return result;
    }

    struct EvolveHelper {
        const Particle& from;
        Particle* to;

        template <class Ker>
        void operator()(const Ker& kernel) {
            kernel.Evolve(from, to);
        }
    };

    struct DensityHelper {
        const Particle& from;
        const Particle& to;
        FloatT& result;

        template <class Ker>
        void operator()(const Ker& kernel) {
            result = kernel.GetTransDensity(from, to);
        }
    };

private:
    std::tuple<T...> fixed_action_kernels_;
};

template <class... T>
class MDPKernel final : public AbstractKernel<MDPKernel<T...>> {
public:
    MDPKernel(ActionConditionedKernel<T...>&& action_conditioned_kernel,
              AbstractAgentPolicy* agent_policy)
        : conditioned_kernel_(std::move(action_conditioned_kernel)), agent_policy_(agent_policy) {
    }

    void Evolve(const Particle& from, Particle* output) {
        size_t action_num = agent_policy_->React(from);
        conditioned_kernel_.EvolveConditionally(from, output, action_num);
    }

    FloatT GetTransDensity(const Particle& from, const Particle& to) {
        size_t action_num = agent_policy_->React(from);
        return conditioned_kernel_.GetTransDensityConditionally(from, to, action_num);
    }

private:
    ActionConditionedKernel<T...> conditioned_kernel_;
    AbstractAgentPolicy* agent_policy_{nullptr};
};
