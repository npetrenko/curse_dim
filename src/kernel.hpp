#pragma once

#include <src/types.hpp>
#include <src/particle.hpp>
#include <src/exceptions.hpp>
#include <src/agent_policy.hpp>
#include <src/util.hpp>

#include <array>
#include <memory>
#include <exception>

template <class DerivedT>
class AbstractKernel : public CRTPDerivedCaster<DerivedT> {
public:
    template <class S1, class S2>
    void Evolve(const Particle<S1>& from, Particle<S2>* output) const {
        assert(from.GetDim() == output->GetDim());
        this->GetDerived()->Evolve(from, output);
    }

    template <class S1, class S2>
    FloatT GetTransDensity(const Particle<S1>& from, const Particle<S2>& to) const {
        assert(from.GetDim() == to.GetDim());
        return this->GetDerived()->GetTransDensity(from, to);
    }
};

template <class DerivedT>
class AbstractConditionedKernel : public CRTPDerivedCaster<DerivedT> {
public:
    template <class S1, class S2>
    void EvolveConditionally(const Particle<S1>& from, Particle<S2>* output,
                             size_t condition) const {
        this->GetDerived()->EvolveConditionally(from, output, condition);
    }

    template <class S1, class S2>
    FloatT GetTransDensityConditionally(const Particle<S1>& from, const Particle<S2>& to,
                                        size_t condition) const {
        return this->GetDerived()->GetTransDensityConditionally(from, to, condition);
    }
};

template <class... T>
class ActionConditionedKernel final
    : public AbstractConditionedKernel<ActionConditionedKernel<T...>> {
public:
    ActionConditionedKernel(T&&... args) : fixed_action_kernels_(std::move(args)...) {
    }

    ActionConditionedKernel(const T&... args) : fixed_action_kernels_(args...) {
    }

    template <class S1, class S2>
    void EvolveConditionally(const Particle<S1>& from, Particle<S2>* output,
                             size_t action_number) const {
        EvolveHelper<S1, S2> helper{from, output};
        CallOnTupleIx(std::move(helper), fixed_action_kernels_, action_number);
    }

    template <class S1, class S2>
    FloatT GetTransDensityConditionally(const Particle<S1>& from, const Particle<S2>& to,
                                        size_t action_number) const {
        FloatT result;
        DensityHelper<S1, S2> helper{from, to, result};
        CallOnTupleIx(std::move(helper), fixed_action_kernels_, action_number);
        return result;
    }

    template <class S1, class S2>
    struct EvolveHelper {
        const Particle<S1>& from;
        Particle<S2>* to;

        template <class Ker>
        void operator()(const Ker& kernel) {
            kernel.Evolve(from, to);
        }
    };

    template <class S1, class S2>
    struct DensityHelper {
        const Particle<S1>& from;
        const Particle<S2>& to;
        FloatT& result;

        template <class Ker>
        void operator()(const Ker& kernel) {
            result = kernel.GetTransDensity(from, to);
        }
    };

    static inline size_t GetDim() {
        return sizeof...(T);
    }

private:
    std::tuple<T...> fixed_action_kernels_;
};

template <class DerivedPolicy, class... T>
class MDPKernel final : public AbstractKernel<MDPKernel<T...>> {
public:
    MDPKernel(ActionConditionedKernel<T...>&& action_conditioned_kernel,
              AbstractAgentPolicy<DerivedPolicy>* agent_policy)
        : conditioned_kernel_(std::move(action_conditioned_kernel)), agent_policy_(agent_policy) {
    }

    void ResetPolicy(AbstractAgentPolicy<DerivedPolicy>* agent_policy) {
        agent_policy_ = agent_policy;
    }

    template <class S1, class S2>
    void Evolve(const Particle<S1>& from, Particle<S2>* output) const {
        size_t action_num = agent_policy_->React(from);
        conditioned_kernel_.EvolveConditionally(from, output, action_num);
    }

    template <class S1, class S2>
    FloatT GetTransDensity(const Particle<S1>& from, const Particle<S2>& to) const {
        size_t action_num = agent_policy_->React(from);
        return conditioned_kernel_.GetTransDensityConditionally(from, to, action_num);
    }

private:
    ActionConditionedKernel<T...> conditioned_kernel_;
    AbstractAgentPolicy<DerivedPolicy>* agent_policy_{nullptr};
};
