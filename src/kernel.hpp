#pragma once

#include <src/types.hpp>
#include <src/particle.hpp>
#include <src/exceptions.hpp>
#include <src/agent_policy.hpp>
#include <src/util.hpp>
#include <src/type_traits.hpp>

#include <src/abstract_kernel.hpp>

#include <array>
#include <memory>
#include <exception>
#include <functional>

template <class... T>
class ActionConditionedKernel final
    : public AbstractConditionedKernel<ActionConditionedKernel<T...>, false> {
    friend class AbstractConditionedKernel<ActionConditionedKernel<T...>, false>;

public:
    ActionConditionedKernel(T&&... args) : fixed_action_kernels_{std::move(args)...} {
    }
    ActionConditionedKernel(const T&... args) : fixed_action_kernels_{args...} {
    }

    inline size_t GetSpaceDim() const {
        size_t result = std::get<0>(fixed_action_kernels_).GetSpaceDim();
// Need to add checks
#ifndef NDEBUG
#endif
        return result;
    }

    static inline size_t GetDim() {
        return sizeof...(T);
    }

private:
    template <class S1, class S2, class RandomDeviceT>
    void EvolveConditionallyImpl(const Particle<S1>& from, Particle<S2>* output,
                                 size_t action_number, RandomDeviceT* rd) const {
        EvolveHelper<S1, S2, RandomDeviceT> helper{from, output, rd};
        CallOnTupleIx(std::move(helper), fixed_action_kernels_, action_number);
    }

    template <class S1, class S2>
    FloatT GetTransDensityConditionallyImpl(const Particle<S1>& from, const Particle<S2>& to,
                                            size_t action_number) const {
        FloatT result;
        DensityHelper<S1, S2> helper{from, to, result};
        CallOnTupleIx(std::move(helper), fixed_action_kernels_, action_number);
        return result;
    }

    template <class S1, class S2, class RandomDeviceT>
    struct EvolveHelper {
        const Particle<S1>& from;
        Particle<S2>* to;
        RandomDeviceT* rd;

        template <class Ker>
        inline void operator()(const Ker& kernel) {
            if constexpr (std::is_same_v<NullType, RandomDeviceT>) {
                kernel.Evolve(from, to);
            } else {
                kernel.Evolve(from, to, rd);
            }
        }
    };

    template <class S1, class S2>
    struct DensityHelper {
        const Particle<S1>& from;
        const Particle<S2>& to;
        FloatT& result;

        template <class Ker>
        inline void operator()(const Ker& kernel) {
            result = kernel.GetTransDensity(from, to);
        }
    };

    std::tuple<T...> fixed_action_kernels_;
};

template <class DerivedT>
class HintableKernel : public AbstractKernel<DerivedT, false> {
    using Caster = CRTPDerivedCaster<DerivedT>;

public:
    HintableKernel() = default;

    HintableKernel(std::mt19937* rd) : AbstractKernel<DerivedT, false>{rd} {
    }

    template <class S>
    auto CalculateHint(const Particle<S>& from) const {
        return Caster::GetDerived()->CalculateHintImpl(from);
    }

    template <class S1, class S2, class HintT>
    FloatT GetTransDensityWithHint(const Particle<S1>& from, const Particle<S2>& to,
                                   HintT* hint) const {
        return Caster::GetDerived()->GetTransDensityWithHintImpl(from, to, hint);
    }
};

template <class DerivedPolicy, class T>
class MDPKernel final : public HintableKernel<MDPKernel<DerivedPolicy, T>> {
    using ThisT = MDPKernel<DerivedPolicy, T>;
    friend class HintableKernel<ThisT>;
    friend class AbstractKernel<ThisT, false>;

public:
    using HintT = size_t;

    MDPKernel(const AbstractConditionedKernel<T, false>& action_conditioned_kernel,
              AbstractAgentPolicy<DerivedPolicy>* agent_policy)
        : conditioned_kernel_{type_traits::GetDeepestLevelCopy(action_conditioned_kernel)},
          agent_policy_{agent_policy} {
    }

    inline void ResetPolicy(AbstractAgentPolicy<DerivedPolicy>* agent_policy) {
        agent_policy_ = agent_policy;
    }

    inline size_t GetSpaceDim() const {
        return conditioned_kernel_.GetSpaceDim();
    }

private:
    template <class S1, class S2, class RandomDeviceT>
    void EvolveImpl(const Particle<S1>& from, Particle<S2>* output, RandomDeviceT* rd) const {
        size_t action_num = agent_policy_->React(from);
        conditioned_kernel_.EvolveConditionally(from, output, action_num, rd);
    }

    template <class S1, class S2>
    FloatT GetTransDensityImpl(const Particle<S1>& from, const Particle<S2>& to) const {
        size_t action_num = agent_policy_->React(from);
        return conditioned_kernel_.GetTransDensityConditionally(from, to, action_num);
    }

    template <class S>
    HintT CalculateHintImpl(const Particle<S>& from) const {
        return agent_policy_->React(from);
    }

    template <class S1, class S2>
    FloatT GetTransDensityWithHintImpl(const Particle<S1>& from, const Particle<S2>& to,
                                       HintT* hint) const {
        return conditioned_kernel_.GetTransDensityConditionally(from, to, *hint);
    }

    type_traits::DeepestCRTPType<AbstractConditionedKernel<T, false>> conditioned_kernel_;
    AbstractAgentPolicy<DerivedPolicy>* agent_policy_{nullptr};
};
